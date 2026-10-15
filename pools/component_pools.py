"""
Component pools and lightweight BO runtime implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import itertools
import io
import math
import os
import re

import numpy as np

def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _resolve_torch_device(params: dict[str, Any] | None = None) -> "torch.device":
    if torch is None:
        raise RuntimeError("Torch is unavailable")
    raw_device = str((params or {}).get("torch_device") or os.getenv("CHEMBO_BO_TORCH_DEVICE") or "cpu").strip()
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raw_device = "cpu"
    return torch.device(raw_device or "cpu")


def _safe_import_torch_stack():
    """Import the optional torch/botorch stack by default.

    AutoBO v3 expects GP and Deep Ensemble surrogates to be available whenever
    the local Python environment can import the stack. Set
    CHEMBO_DISABLE_TORCH_STACK=1 only when a specific runtime needs to force
    fallback-only behavior.
    """

    if _env_flag("CHEMBO_DISABLE_TORCH_STACK"):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "Torch/BoTorch stack disabled via CHEMBO_DISABLE_TORCH_STACK=1.",
        )

    try:
        import torch as _torch
        from botorch.acquisition import LogExpectedImprovement as _LogExpectedImprovement, UpperConfidenceBound as _UpperConfidenceBound
        from botorch.fit import fit_gpytorch_mll as _fit_gpytorch_mll
        from botorch.models import SingleTaskGP as _SingleTaskGP
        from botorch.models.transforms.outcome import Standardize as _Standardize
        from gpytorch.kernels import AdditiveKernel as _AdditiveKernel, MaternKernel as _MaternKernel, ProductKernel as _ProductKernel, RBFKernel as _RBFKernel, ScaleKernel as _ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood as _ExactMarginalLogLikelihood
        try:
            from .smk_kernels import WrappedSMK as _WrappedSMK
        except ImportError:  # pragma: no cover
            from pools.smk_kernels import WrappedSMK as _WrappedSMK
    except Exception as exc:  # pragma: no cover
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            f"Torch/BoTorch stack unavailable: {type(exc).__name__}: {exc}",
        )

    return (
        _torch,
        _LogExpectedImprovement,
        _UpperConfidenceBound,
        _fit_gpytorch_mll,
        _SingleTaskGP,
        _Standardize,
        _AdditiveKernel,
        _MaternKernel,
        _ProductKernel,
        _RBFKernel,
        _ScaleKernel,
        _ExactMarginalLogLikelihood,
        _WrappedSMK,
        None,
    )


(
    torch,
    LogExpectedImprovement,
    UpperConfidenceBound,
    fit_gpytorch_mll,
    SingleTaskGP,
    Standardize,
    AdditiveKernel,
    MaternKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
    ExactMarginalLogLikelihood,
    WrappedSMK,
    TORCH_STATUS_NOTE,
) = _safe_import_torch_stack()


def _safe_import_rdkit():
    """Import RDKit only when it is likely to be ABI-compatible.

    Some RDKit wheels are compiled against the NumPy 1.x C-API. Importing them
    under NumPy 2 can emit a long cascade of low-signal tracebacks and may even
    segfault the process before our Python fallback logic can help. To keep the
    core workflow usable, we skip RDKit by default on NumPy >= 2 unless the
    caller explicitly opts in via CHEMBO_ENABLE_RDKIT=1.
    """

    if _env_flag("CHEMBO_DISABLE_RDKIT"):
        return (None, None, None, None, None, "RDKit disabled via CHEMBO_DISABLE_RDKIT=1.")

    numpy_major = int(str(np.__version__).split(".", 1)[0])
    if numpy_major >= 2 and not _env_flag("CHEMBO_ENABLE_RDKIT"):
        return (
            None,
            None,
            None,
            None,
            None,
            (
                f"RDKit import skipped under NumPy {np.__version__}. "
                "Many RDKit builds are compiled against the NumPy 1.x ABI; "
                "downgrade to numpy<2 for chemistry-aware features, or set "
                "CHEMBO_ENABLE_RDKIT=1 if your RDKit build is known to support NumPy 2."
            ),
        )

    try:
        # Suppress noisy C-extension import tracebacks from optional RDKit paths.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from rdkit import Chem as _Chem
            from rdkit import DataStructs as _DataStructs
            from rdkit.Chem import Descriptors as _Descriptors
            from rdkit.Chem import rdFingerprintGenerator as _rdFingerprintGenerator
            from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
    except Exception as exc:  # pragma: no cover
        return (
            None,
            None,
            None,
            None,
            None,
            f"RDKit unavailable: {type(exc).__name__}: {exc}",
        )

    return _Chem, _Descriptors, _rdMolDescriptors, _rdFingerprintGenerator, _DataStructs, None


(
    Chem,
    Descriptors,
    rdMolDescriptors,
    rdFingerprintGenerator,
    DataStructs,
    RDKIT_STATUS_NOTE,
) = _safe_import_rdkit()


PHYSICAL_FEATURE_NAMES = [
    "mean_atomic_number",
    "mean_group",
    "mean_period",
    "mean_electronegativity",
    "mean_covalent_radius",
    "metal_fraction",
    "n_elements",
]

# Lightweight periodic-table subset for process-catalysis style problems where
# RDKit molecular features are not the right abstraction.
ELEMENT_PHYSICAL_PROPERTIES: dict[str, dict[str, float]] = {
    "Al": {"atomic_number": 13.0, "group": 13.0, "period": 3.0, "electronegativity": 1.61, "covalent_radius": 121.0, "is_metal": 1.0},
    "B": {"atomic_number": 5.0, "group": 13.0, "period": 2.0, "electronegativity": 2.04, "covalent_radius": 84.0, "is_metal": 0.0},
    "Ba": {"atomic_number": 56.0, "group": 2.0, "period": 6.0, "electronegativity": 0.89, "covalent_radius": 215.0, "is_metal": 1.0},
    "Br": {"atomic_number": 35.0, "group": 17.0, "period": 4.0, "electronegativity": 2.96, "covalent_radius": 120.0, "is_metal": 0.0},
    "C": {"atomic_number": 6.0, "group": 14.0, "period": 2.0, "electronegativity": 2.55, "covalent_radius": 76.0, "is_metal": 0.0},
    "Ca": {"atomic_number": 20.0, "group": 2.0, "period": 4.0, "electronegativity": 1.00, "covalent_radius": 176.0, "is_metal": 1.0},
    "Ce": {"atomic_number": 58.0, "group": 3.0, "period": 6.0, "electronegativity": 1.12, "covalent_radius": 204.0, "is_metal": 1.0},
    "Cl": {"atomic_number": 17.0, "group": 17.0, "period": 3.0, "electronegativity": 3.16, "covalent_radius": 102.0, "is_metal": 0.0},
    "Co": {"atomic_number": 27.0, "group": 9.0, "period": 4.0, "electronegativity": 1.88, "covalent_radius": 126.0, "is_metal": 1.0},
    "Cs": {"atomic_number": 55.0, "group": 1.0, "period": 6.0, "electronegativity": 0.79, "covalent_radius": 244.0, "is_metal": 1.0},
    "Cu": {"atomic_number": 29.0, "group": 11.0, "period": 4.0, "electronegativity": 1.90, "covalent_radius": 132.0, "is_metal": 1.0},
    "Eu": {"atomic_number": 63.0, "group": 3.0, "period": 6.0, "electronegativity": 1.20, "covalent_radius": 198.0, "is_metal": 1.0},
    "F": {"atomic_number": 9.0, "group": 17.0, "period": 2.0, "electronegativity": 3.98, "covalent_radius": 57.0, "is_metal": 0.0},
    "Fe": {"atomic_number": 26.0, "group": 8.0, "period": 4.0, "electronegativity": 1.83, "covalent_radius": 132.0, "is_metal": 1.0},
    "H": {"atomic_number": 1.0, "group": 1.0, "period": 1.0, "electronegativity": 2.20, "covalent_radius": 31.0, "is_metal": 0.0},
    "Hf": {"atomic_number": 72.0, "group": 4.0, "period": 6.0, "electronegativity": 1.30, "covalent_radius": 175.0, "is_metal": 1.0},
    "I": {"atomic_number": 53.0, "group": 17.0, "period": 5.0, "electronegativity": 2.66, "covalent_radius": 139.0, "is_metal": 0.0},
    "K": {"atomic_number": 19.0, "group": 1.0, "period": 4.0, "electronegativity": 0.82, "covalent_radius": 203.0, "is_metal": 1.0},
    "La": {"atomic_number": 57.0, "group": 3.0, "period": 6.0, "electronegativity": 1.10, "covalent_radius": 207.0, "is_metal": 1.0},
    "Li": {"atomic_number": 3.0, "group": 1.0, "period": 2.0, "electronegativity": 0.98, "covalent_radius": 128.0, "is_metal": 1.0},
    "Mg": {"atomic_number": 12.0, "group": 2.0, "period": 3.0, "electronegativity": 1.31, "covalent_radius": 141.0, "is_metal": 1.0},
    "Mn": {"atomic_number": 25.0, "group": 7.0, "period": 4.0, "electronegativity": 1.55, "covalent_radius": 139.0, "is_metal": 1.0},
    "Mo": {"atomic_number": 42.0, "group": 6.0, "period": 5.0, "electronegativity": 2.16, "covalent_radius": 154.0, "is_metal": 1.0},
    "N": {"atomic_number": 7.0, "group": 15.0, "period": 2.0, "electronegativity": 3.04, "covalent_radius": 71.0, "is_metal": 0.0},
    "Na": {"atomic_number": 11.0, "group": 1.0, "period": 3.0, "electronegativity": 0.93, "covalent_radius": 166.0, "is_metal": 1.0},
    "Nb": {"atomic_number": 41.0, "group": 5.0, "period": 5.0, "electronegativity": 1.60, "covalent_radius": 164.0, "is_metal": 1.0},
    "Nd": {"atomic_number": 60.0, "group": 3.0, "period": 6.0, "electronegativity": 1.14, "covalent_radius": 201.0, "is_metal": 1.0},
    "Ni": {"atomic_number": 28.0, "group": 10.0, "period": 4.0, "electronegativity": 1.91, "covalent_radius": 124.0, "is_metal": 1.0},
    "O": {"atomic_number": 8.0, "group": 16.0, "period": 2.0, "electronegativity": 3.44, "covalent_radius": 66.0, "is_metal": 0.0},
    "P": {"atomic_number": 15.0, "group": 15.0, "period": 3.0, "electronegativity": 2.19, "covalent_radius": 107.0, "is_metal": 0.0},
    "Pd": {"atomic_number": 46.0, "group": 10.0, "period": 5.0, "electronegativity": 2.20, "covalent_radius": 139.0, "is_metal": 1.0},
    "S": {"atomic_number": 16.0, "group": 16.0, "period": 3.0, "electronegativity": 2.58, "covalent_radius": 105.0, "is_metal": 0.0},
    "Si": {"atomic_number": 14.0, "group": 14.0, "period": 3.0, "electronegativity": 1.90, "covalent_radius": 111.0, "is_metal": 0.0},
    "Sr": {"atomic_number": 38.0, "group": 2.0, "period": 5.0, "electronegativity": 0.95, "covalent_radius": 195.0, "is_metal": 1.0},
    "Tb": {"atomic_number": 65.0, "group": 3.0, "period": 6.0, "electronegativity": 1.10, "covalent_radius": 194.0, "is_metal": 1.0},
    "Ti": {"atomic_number": 22.0, "group": 4.0, "period": 4.0, "electronegativity": 1.54, "covalent_radius": 160.0, "is_metal": 1.0},
    "V": {"atomic_number": 23.0, "group": 5.0, "period": 4.0, "electronegativity": 1.63, "covalent_radius": 153.0, "is_metal": 1.0},
    "W": {"atomic_number": 74.0, "group": 6.0, "period": 6.0, "electronegativity": 2.36, "covalent_radius": 162.0, "is_metal": 1.0},
    "Y": {"atomic_number": 39.0, "group": 3.0, "period": 5.0, "electronegativity": 1.22, "covalent_radius": 180.0, "is_metal": 1.0},
    "Zn": {"atomic_number": 30.0, "group": 12.0, "period": 4.0, "electronegativity": 1.65, "covalent_radius": 122.0, "is_metal": 1.0},
    "Zr": {"atomic_number": 40.0, "group": 4.0, "period": 5.0, "electronegativity": 1.33, "covalent_radius": 175.0, "is_metal": 1.0},
}

FORMULA_ALIASES: dict[str, str] = {
    "BEA": "Al2Si30O64",
    "ZSM-5": "Al2Si94O192",
}

@dataclass
class PoolEntry:
    key: str
    display_name: str
    description: str
    tags: dict[str, Any] = field(default_factory=dict)
    factory: Callable[..., Any] | None = None


def detect_runtime_capabilities() -> dict[str, Any]:
    rdkit_available = Chem is not None and rdFingerprintGenerator is not None and DataStructs is not None
    torch_stack_present = all(
        dependency is not None
        for dependency in (
            torch,
            SingleTaskGP,
            fit_gpytorch_mll,
            ExactMarginalLogLikelihood,
            ScaleKernel,
            MaternKernel,
            RBFKernel,
            LogExpectedImprovement,
            UpperConfidenceBound,
            Standardize,
        )
    )
    notes = []
    if RDKIT_STATUS_NOTE:
        notes.append(RDKIT_STATUS_NOTE)
    if TORCH_STATUS_NOTE:
        notes.append(TORCH_STATUS_NOTE)
    if not torch_stack_present:
        notes.append("BoTorch stack is unavailable in the current environment.")
    else:
        notes.append("BoTorch stack enabled.")
    return {
        "rdkit": rdkit_available,
        "torch_stack": bool(torch_stack_present),
        "runtime_mode": "botorch" if torch_stack_present else "fallback_only",
        "notes": notes,
    }


class BaseSurrogateModel:
    def __init__(
        self,
        search_space: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ):
        self.search_space = list(search_space or [])
        self.params = params or {}
        self.metadata: dict[str, Any] = {"notes": []}

    def fit(self, candidates: list[dict[str, Any]], y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class CoCaBOGPSurrogate(BaseSurrogateModel):
    """Gaussian Process surrogate with a CoCaBO mixed kernel."""

    def __init__(
        self,
        search_space: list[dict[str, Any]],
        kernel_name: str = "matern52",
        params: dict[str, Any] | None = None,
        kernel_params: dict[str, Any] | None = None,
        feature_spec: dict[str, Any] | None = None,
    ):
        del feature_spec
        super().__init__(search_space, params)
        self.kernel_name = str(kernel_name or "matern52")
        self.kernel_params = kernel_params or {}
        self.device = _resolve_torch_device(self.params) if torch is not None else None
        if self.device is not None:
            self.metadata["torch_device"] = str(self.device)
        raw_batch_size = self.params.get("prediction_batch_size") or os.getenv("CHEMBO_BO_PREDICT_BATCH_SIZE") or 1024
        try:
            self.prediction_batch_size = max(1, int(raw_batch_size))
        except Exception:
            self.prediction_batch_size = 1024
        self.metadata["prediction_batch_size"] = self.prediction_batch_size
        self.model = None
        self._var_specs: list[dict[str, Any]] = []
        self._cont_indices: list[int] = []
        self._cat_indices: list[int] = []
        self._cat_specs: list[dict[str, Any]] = []
        self._build_encoding_spec()

    def _build_encoding_spec(self) -> None:
        self._var_specs = []
        self._cont_indices = []
        self._cat_indices = []
        self._cat_specs = []
        offset = 0
        for variable in self.search_space:
            name = str(variable.get("name") or "")
            if not name:
                continue
            if variable.get("type", "categorical") == "continuous":
                low, high = _continuous_bounds(variable)
                self._var_specs.append(
                    {
                        "name": name,
                        "type": "continuous",
                        "low": low,
                        "high": high,
                        "scale": _continuous_scale(variable),
                        "index": offset,
                    }
                )
                self._cont_indices.append(offset)
                offset += 1
            else:
                labels = _domain_labels(variable) or ["unknown"]
                self._var_specs.append(
                    {
                        "name": name,
                        "type": "categorical",
                        "labels": labels,
                        "label_to_idx": {str(label): idx for idx, label in enumerate(labels)},
                        "n_categories": max(len(labels), 1),
                        "index": offset,
                    }
                )
                self._cat_indices.append(offset)
                self._cat_specs.append(
                    {
                        "name": name,
                        "index": offset,
                        "labels": labels,
                        "n_categories": max(len(labels), 1),
                        "encoding_scale": max(float(len(labels)) - 1.0, 1.0),
                    }
                )
                offset += 1

    def encode_candidates(self, candidates: list[dict[str, Any]]) -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("Torch is unavailable")
        rows: list[list[float]] = []
        for candidate in candidates:
            row: list[float] = []
            for spec in self._var_specs:
                value = candidate.get(spec["name"])
                if spec["type"] == "continuous":
                    row.append(
                        _normalize_continuous(
                            value,
                            float(spec["low"]),
                            float(spec["high"]),
                            spec.get("scale"),
                        )
                    )
                elif spec["type"] == "categorical":
                    idx = float(spec["label_to_idx"].get(str(value), 0))
                    row.append(idx / max(float(spec.get("n_categories", 1)) - 1.0, 1.0))
            rows.append(row)
        if not rows:
            width = 0
            for spec in self._var_specs:
                width += int(spec.get("dim", 1)) if spec.get("type") == "feature_map" else 1
            return torch.zeros((0, width), dtype=torch.double, device=self.device)
        return torch.as_tensor(rows, dtype=torch.double, device=self.device)

    def _build_kernel(self):
        if ScaleKernel is None or MaternKernel is None:
            raise RuntimeError("BoTorch kernel dependencies are unavailable")
        from pools.cocabo_kernel import CoCaBOMixedKernel

        base_kernel = _build_base_cont_kernel_for_cocabo(
            self.kernel_name,
            max(len(self._cont_indices), 1),
            self.metadata,
            self.kernel_params,
        )
        return CoCaBOMixedKernel(
            cont_dims=self._cont_indices,
            cat_dims=self._cat_indices,
            base_cont_kernel=base_kernel,
            cat_kernel_type=str(self.kernel_params.get("cat_kernel", "indicator") or "indicator"),
            cat_kernel_params=dict(self.kernel_params.get("cat_kernel_params") or {}),
            cat_specs=self._cat_specs,
        )

    def fit(self, candidates: list[dict[str, Any]], y: np.ndarray) -> None:
        if not detect_runtime_capabilities()["torch_stack"]:
            raise RuntimeError("BoTorch stack is unavailable")
        if not candidates:
            raise RuntimeError("Cannot fit CoCaBO GP without training data")
        train_X = self.encode_candidates(candidates)
        train_Y = _to_torch_column(y, device=self.device)
        covar_module = self._build_kernel()
        if hasattr(covar_module, "to"):
            covar_module = covar_module.to(self.device)

        if self.kernel_name.lower() in {"smk", "smkbo"} and self._cont_indices:
            try:
                cont_kernel = covar_module.cont_kernel
                if hasattr(cont_kernel, "initialize_from_data"):
                    cont_kernel.initialize_from_data(train_X[:, self._cont_indices], train_Y.squeeze(-1))
            except Exception as exc:
                self.metadata.setdefault("notes", []).append(f"SMK initialization skipped: {type(exc).__name__}: {exc}")

        likelihood = None
        try:
            from gpytorch.constraints import Interval
            from gpytorch.likelihoods import GaussianLikelihood

            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1.0))
        except Exception as exc:
            self.metadata.setdefault("notes", []).append(f"Learned-noise likelihood unavailable: {type(exc).__name__}: {exc}")

        gp_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "covar_module": covar_module,
            "outcome_transform": Standardize(m=1),
        }
        if likelihood is not None:
            gp_kwargs["likelihood"] = likelihood
        else:
            noise_level = max(float(self.params.get("noise_level", 1e-4)), 1e-6)
            gp_kwargs["train_Yvar"] = torch.full_like(train_Y, noise_level)
        self.model = SingleTaskGP(**gp_kwargs).to(self.device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if hasattr(mll, "to"):
            mll = mll.to(self.device)
        try:
            try:
                fit_gpytorch_mll(mll, max_attempts=3)
            except TypeError:
                fit_gpytorch_mll(mll)
        except Exception as first_exc:
            try:
                from botorch.optim.fit import fit_gpytorch_mll_torch

                try:
                    fit_gpytorch_mll_torch(mll, step_limit=200)
                except TypeError:
                    fit_gpytorch_mll_torch(mll)
                self.metadata.setdefault("notes", []).append(
                    f"GP fit recovered with torch optimizer after {type(first_exc).__name__}."
                )
            except Exception as second_exc:
                self.metadata.setdefault("notes", []).append(
                    "GP fit fell back to unoptimized hyperparameters: "
                    f"{type(first_exc).__name__}: {first_exc}; {type(second_exc).__name__}: {second_exc}"
                )
                self.metadata["fit_partial"] = True
        self.model.eval()
        self.model.likelihood.eval()

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("CoCaBO GP model must be fit before prediction")
        if not candidates:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        batch_size = max(1, int(self.prediction_batch_size or len(candidates)))
        if len(candidates) <= batch_size:
            batches = [candidates]
        else:
            batches = [candidates[index : index + batch_size] for index in range(0, len(candidates), batch_size)]
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in batches:
                posterior = self.model.posterior(self.encode_candidates(batch))
                means.append(posterior.mean.squeeze(-1).detach().cpu().numpy())
                variance = posterior.variance.squeeze(-1).clamp_min(1e-12)
                stds.append(variance.sqrt().detach().cpu().numpy())
        mean = np.concatenate(means, axis=0) if len(means) > 1 else means[0]
        std = np.concatenate(stds, axis=0) if len(stds) > 1 else stds[0]
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


class CatBoostSurrogate(BaseSurrogateModel):
    """CatBoost surrogate using RMSEWithUncertainty and native categorical input."""

    def __init__(
        self,
        search_space: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
        feature_spec: dict[str, Any] | None = None,
    ):
        del feature_spec
        super().__init__(search_space, params)
        self._encoding_spec: list[dict[str, Any]] = []
        self._feature_names: list[str] = []
        self._cat_feature_indices: list[int] = []
        self._model = None
        self._build_encoding_spec()

    def _build_encoding_spec(self) -> None:
        self._encoding_spec = []
        self._feature_names = []
        self._cat_feature_indices = []
        for variable in self.search_space:
            name = str(variable.get("name") or f"x{len(self._encoding_spec)}")
            if variable.get("type", "categorical") == "continuous":
                low, high = _continuous_bounds(variable)
                self._encoding_spec.append(
                    {
                        "name": name,
                        "type": "continuous",
                        "low": low,
                        "high": high,
                        "scale": _continuous_scale(variable),
                    }
                )
                self._feature_names.append(name)
                continue
            self._encoding_spec.append({"name": name, "type": "categorical"})
            self._cat_feature_indices.append(len(self._feature_names))
            self._feature_names.append(name)

    def _to_feature_rows(self, candidates: list[dict[str, Any]]) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for candidate in candidates:
            row: list[Any] = []
            for spec in self._encoding_spec:
                value = candidate.get(spec["name"])
                if spec["type"] == "continuous":
                    row.append(
                        _normalize_continuous(
                            value,
                            float(spec["low"]),
                            float(spec["high"]),
                            spec.get("scale"),
                        )
                    )
                else:
                    row.append(str(value) if value is not None else "")
            rows.append(row)
        return rows

    def fit(self, candidates: list[dict[str, Any]], y: np.ndarray) -> None:
        try:
            import catboost as cb
        except ImportError as exc:
            raise RuntimeError("CatBoost is not installed. Install catboost>=0.26 to enable this surrogate.") from exc

        if not candidates:
            raise RuntimeError("Cannot fit CatBoost without training data")
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n_obs = len(y_arr)
        depth = int(self.params.get("depth", min(4, max(2, int(np.log2(n_obs + 1))))))
        iterations = int(self.params.get("iterations", min(200, max(80, 5 * n_obs))))
        resolved_params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "l2_leaf_reg": float(self.params.get("l2_leaf_reg", 3.0)),
            "loss_function": "RMSEWithUncertainty",
            "bootstrap_type": str(self.params.get("bootstrap_type", "Bayesian")),
            "bagging_temperature": float(self.params.get("bagging_temperature", 1.0)),
            "random_seed": int(self.params.get("random_seed", 42)),
            "verbose": False,
            "allow_writing_files": False,
        }
        pool = cb.Pool(
            data=self._to_feature_rows(candidates),
            label=y_arr,
            cat_features=self._cat_feature_indices,
            feature_names=self._feature_names,
        )
        self._model = cb.CatBoostRegressor(**resolved_params)
        self._model.fit(pool)

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("CatBoost model must be fit before prediction")
        try:
            import catboost as cb
        except ImportError as exc:
            raise RuntimeError("CatBoost is not installed. Install catboost>=0.26 to enable this surrogate.") from exc
        pool = cb.Pool(
            data=self._to_feature_rows(candidates),
            cat_features=self._cat_feature_indices,
            feature_names=self._feature_names,
        )
        preds = np.asarray(
            self._model.predict(pool, prediction_type="RMSEWithUncertainty"),
            dtype=float,
        )
        if preds.ndim == 1:
            mean = preds
            sigma = np.full_like(mean, 1e-3, dtype=float)
        else:
            # CatBoost returns (n_candidates, 2), including the ambiguous 2x2 case.
            if preds.shape != (len(candidates), 2):
                raise ValueError(f"Unexpected CatBoost prediction shape: {preds.shape}")
            mean = preds[:, 0]
            variance = np.maximum(preds[:, 1], 1e-12)
            sigma = np.sqrt(variance)
        return np.asarray(mean, dtype=float), np.maximum(np.asarray(sigma, dtype=float), 1e-6)


class _SmallNN(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int):
        if torch is None:
            raise RuntimeError("PyTorch is required for DeepEnsembleSurrogate")
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden1)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden1), int(hidden2)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden2), 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)


class DeepEnsembleSurrogate(BaseSurrogateModel):
    """Small neural-network deep ensemble with chemistry-aware per-variable features."""

    def __init__(
        self,
        search_space: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
        feature_spec: dict[str, Any] | None = None,
    ):
        super().__init__(search_space, params)
        self.feature_spec = feature_spec or {}
        self.n_models = int(self.params.get("n_models", 5))
        self.hidden1 = int(self.params.get("hidden1", 64))
        self.hidden2 = int(self.params.get("hidden2", 32))
        self.n_epochs = int(self.params.get("n_epochs", 200))
        self.learning_rate = float(self.params.get("learning_rate", 1e-3))
        self.weight_decay = float(self.params.get("weight_decay", 1e-3))
        self.random_seed = int(self.params.get("random_seed", 42))
        self.device = _resolve_torch_device(self.params) if torch is not None else None
        if self.device is not None:
            self.metadata["torch_device"] = str(self.device)
        self._encoding_spec: list[dict[str, Any]] = []
        self._models: list[Any] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def _build_encoding_spec(self) -> None:
        from pools.deep_ensemble_features import compute_rdkit_features_for_variable

        variable_features = self.feature_spec.get("variable_features") or {}
        spec: list[dict[str, Any]] = []
        for variable in self.search_space:
            name = str(variable.get("name") or "")
            if variable.get("type", "categorical") == "continuous":
                low, high = _continuous_bounds(variable)
                spec.append(
                    {
                        "name": name,
                        "type": "continuous",
                        "low": low,
                        "high": high,
                        "scale": _continuous_scale(variable),
                        "dim": 1,
                    }
                )
                continue

            feature_map = None
            feature_entry = variable_features.get(name) or {}
            desc_names = feature_entry.get("descriptor_names", []) if isinstance(feature_entry, dict) else []
            if desc_names and variable.get("smiles_map"):
                feature_map = compute_rdkit_features_for_variable(
                    variable,
                    list(desc_names),
                    (Chem, Descriptors, rdMolDescriptors),
                )
            if not feature_map:
                physical_map = {
                    label: vector
                    for label in (_domain_labels(variable) or ["unknown"])
                    for vector in [_physical_feature_vector_from_label(label)]
                    if vector is not None
                }
                if physical_map and len(physical_map) >= 0.8 * len(_domain_labels(variable) or ["unknown"]):
                    feature_map = physical_map
            if feature_map:
                dim = len(next(iter(feature_map.values())))
                spec.append({"name": name, "type": "feature_map", "feature_map": feature_map, "dim": dim})
            else:
                labels = _domain_labels(variable) or ["unknown"]
                spec.append(
                    {
                        "name": name,
                        "type": "one_hot_cat",
                        "label_to_idx": {str(label): idx for idx, label in enumerate(labels)},
                        "n_categories": max(len(labels), 1),
                        "dim": max(len(labels), 1),
                    }
                )
        self._encoding_spec = spec

    def _encode_candidates(self, candidates: list[dict[str, Any]]) -> np.ndarray:
        if not self._encoding_spec:
            self._build_encoding_spec()
        rows: list[list[float]] = []
        for candidate in candidates:
            row: list[float] = []
            for spec in self._encoding_spec:
                value = candidate.get(spec["name"])
                if spec["type"] == "continuous":
                    row.append(
                        _normalize_continuous(
                            value,
                            float(spec["low"]),
                            float(spec["high"]),
                            spec.get("scale"),
                        )
                    )
                elif spec["type"] == "feature_map":
                    vector = spec["feature_map"].get(str(value))
                    if vector is None:
                        vector = np.zeros(int(spec["dim"]), dtype=float)
                    row.extend(np.asarray(vector, dtype=float).reshape(-1).tolist())
                else:
                    vector = np.zeros(int(spec["n_categories"]), dtype=float)
                    idx = spec["label_to_idx"].get(str(value))
                    if idx is not None:
                        vector[int(idx)] = 1.0
                    row.extend(vector.tolist())
            rows.append(row)
        if not rows:
            return np.zeros((0, sum(int(item["dim"]) for item in self._encoding_spec)), dtype=float)
        return np.asarray(rows, dtype=float)

    def fit(self, candidates: list[dict[str, Any]], y: np.ndarray) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for DeepEnsembleSurrogate")
        if not candidates:
            raise RuntimeError("Cannot fit DeepEnsemble without training data")
        X_np = self._encode_candidates(candidates)
        y_np = np.asarray(y, dtype=float).reshape(-1)
        self._feature_mean = np.mean(X_np, axis=0)
        self._feature_std = np.std(X_np, axis=0)
        self._feature_std[self._feature_std < 1e-8] = 1.0
        X_norm = (X_np - self._feature_mean) / self._feature_std
        self._models = []
        rng = np.random.default_rng(self.random_seed)
        for model_index in range(max(self.n_models, 1)):
            torch.manual_seed(self.random_seed + 137 * model_index)
            model = _SmallNN(X_norm.shape[1], self.hidden1, self.hidden2).to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            sample_indices = rng.choice(len(y_np), size=len(y_np), replace=len(y_np) > 1)
            X_t = torch.as_tensor(X_norm[sample_indices], dtype=torch.float32, device=self.device)
            y_t = torch.as_tensor(y_np[sample_indices], dtype=torch.float32, device=self.device)
            loss_fn = torch.nn.MSELoss()
            model.train()
            for _ in range(max(self.n_epochs, 1)):
                optimizer.zero_grad()
                loss = loss_fn(model(X_t), y_t)
                loss.backward()
                optimizer.step()
            model.eval()
            self._models.append(model)

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError("DeepEnsemble model must be fit before prediction")
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("DeepEnsemble feature normalization is missing")
        X_np = self._encode_candidates(candidates)
        X_norm = (X_np - self._feature_mean) / self._feature_std
        X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=self.device)
        predictions = []
        with torch.no_grad():
            for model in self._models:
                predictions.append(model(X_t).detach().cpu().numpy())
        pred_stack = np.stack(predictions, axis=0)
        mean = np.mean(pred_stack, axis=0)
        sigma = np.sqrt(np.maximum(np.var(pred_stack, axis=0), 1e-6))
        return np.asarray(mean, dtype=float), np.asarray(sigma, dtype=float)


class AcquisitionFunction:
    def __init__(self, key: str, params: dict[str, Any] | None = None):
        self.key = key
        self.params = params or {}
        self.metadata: dict[str, Any] = {"notes": []}

    def score(
        self,
        surrogate: BaseSurrogateModel,
        X: np.ndarray,
        best_f: float | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not isinstance(surrogate, CoCaBOGPSurrogate) or surrogate.model is None:
            raise RuntimeError("BoTorch acquisition scoring requires a fit BoTorch GP surrogate")
        X_tensor = _to_torch_matrix(X, device=getattr(surrogate, "device", None)).unsqueeze(-2)

        if self.key in {"log_ei", "qlog_ei"}:
            if self.key == "qlog_ei":
                self.metadata.setdefault("notes", []).append("Batch q-LogEI is ranked with pointwise BoTorch LogEI in Phase 1.")
            scorer = LogExpectedImprovement(
                model=surrogate.model,
                best_f=float(best_f if best_f is not None else 0.0),
            )
            with torch.no_grad():
                return scorer(X_tensor).detach().cpu().numpy().reshape(-1)
        if self.key == "ucb":
            beta = float(self.params.get("beta", 0.4))
            scorer = UpperConfidenceBound(model=surrogate.model, beta=beta)
            with torch.no_grad():
                return scorer(X_tensor).detach().cpu().numpy().reshape(-1)
        if self.key == "ts":
            with torch.no_grad():
                posterior = surrogate.model.posterior(_to_torch_matrix(X, device=getattr(surrogate, "device", None)))
                sample = posterior.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(-1)
            return sample.detach().cpu().numpy().reshape(-1)
        raise RuntimeError(f"Unsupported acquisition function: {self.key}")


def _algorithm_profile(
    *,
    what_it_is: str,
    best_for: list[str],
    avoid_when: list[str],
    space_support: str,
    data_regime: str,
    uncertainty_quality: str,
    cost: str,
    interpretability: str,
    dependencies: list[str],
    implementation_status: str,
    fallback_to: str | None,
    fallback_trigger: str | None,
    selection_hints: list[str],
    **extra: Any,
) -> dict[str, Any]:
    profile = {
        "what_it_is": what_it_is,
        "best_for": best_for,
        "avoid_when": avoid_when,
        "space_support": space_support,
        "data_regime": data_regime,
        "uncertainty_quality": uncertainty_quality,
        "cost": cost,
        "interpretability": interpretability,
        "dependencies": dependencies,
        "implementation_status": implementation_status,
        "fallback_to": fallback_to,
        "fallback_trigger": fallback_trigger,
        "selection_hints": selection_hints,
    }
    profile.update(extra)
    return profile


SURROGATE_POOL: dict[str, PoolEntry] = {
    "gp_cocabo": PoolEntry(
        key="gp_cocabo",
        display_name="GP with CoCaBO Mixed Kernel",
        description="Gaussian Process with CoCaBO mixed categorical/continuous kernel.",
        tags=_algorithm_profile(
            what_it_is="GP with CoCaBO kernel: indicator categorical kernel plus selected continuous kernel.",
            best_for=["mixed categorical+continuous chemistry", "low-data BO", "DAR/OCM benchmarks"],
            avoid_when=["n < 4", "very large datasets"],
            space_support="mixed categorical + continuous",
            data_regime="4-200 observations",
            uncertainty_quality="high",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_v3",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=["Default GP surrogate for mixed chemistry spaces."],
        ),
        factory=None,
    ),
    "catboost": PoolEntry(
        key="catboost",
        display_name="CatBoost (RMSEWithUncertainty)",
        description="CatBoost with heteroscedastic uncertainty and native categorical features.",
        tags=_algorithm_profile(
            what_it_is="Gradient-boosted trees trained with RMSEWithUncertainty.",
            best_for=["native categorical chemistry spaces", "mid-data nonlinear response surfaces"],
            avoid_when=["n < 12", "catboost is unavailable"],
            space_support="mixed categorical + continuous",
            data_regime="12+ observations",
            uncertainty_quality="moderate-high",
            cost="low-to-medium",
            interpretability="medium",
            dependencies=["catboost"],
            implementation_status="native_v3",
            fallback_to="gp_cocabo",
            fallback_trigger="catboost unavailable or fitting fails",
            selection_hints=["Complements GP when categorical tree splits are useful."],
        ),
        factory=None,
    ),
    "deep_ensemble": PoolEntry(
        key="deep_ensemble",
        display_name="Deep Ensemble (5 x SmallNN)",
        description="Small neural-network ensemble with LLM/RDKit-guided features.",
        tags=_algorithm_profile(
            what_it_is="Five small NNs with bootstrap resampling and ensemble variance.",
            best_for=["later campaign phase", "SMILES-backed categorical variables", "complex nonlinear surfaces"],
            avoid_when=["n < 20", "torch unavailable"],
            space_support="mixed categorical + continuous",
            data_regime="20+ observations",
            uncertainty_quality="moderate",
            cost="moderate",
            interpretability="low",
            dependencies=["torch"],
            implementation_status="native_v3",
            fallback_to="gp_cocabo",
            fallback_trigger="torch unavailable or fitting fails",
            selection_hints=["Use as a diversity complement once enough observations exist."],
        ),
        factory=None,
    ),
}


KERNEL_POOL: dict[str, PoolEntry] = {
    "matern52": PoolEntry(
        key="matern52",
        display_name="Matern-5/2",
        description="General-purpose smooth kernel and default BO baseline.",
        tags=_algorithm_profile(
            what_it_is="Matérn-5/2 kernel with ARD-like behavior in a BoTorch SingleTaskGP.",
            best_for=["general BO", "moderately smooth objectives"],
            avoid_when=["objective is clearly rougher than smooth", "you need explicit mixed-space structure"],
            space_support="continuous or encoded mixed spaces",
            data_regime="strong low-data default",
            uncertainty_quality="high",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_phase1",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=["Use as default when unsure.", "Safe first kernel for most campaigns."],
        ),
        factory=None,
    ),
    "matern32": PoolEntry(
        key="matern32",
        display_name="Matern-3/2",
        description="Rougher kernel than Matern-5/2 and more tolerant of non-smooth response surfaces.",
        tags=_algorithm_profile(
            what_it_is="Matérn-3/2 kernel for rougher objective assumptions.",
            best_for=["rougher surfaces", "less smooth chemistry response"],
            avoid_when=["you believe the objective is highly smooth"],
            space_support="continuous or encoded mixed spaces",
            data_regime="low data",
            uncertainty_quality="high",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_phase1",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=["Try when Matérn-5/2 appears over-smooth or overconfident."],
        ),
        factory=None,
    ),
    "rbf": PoolEntry(
        key="rbf",
        display_name="RBF / Squared Exponential",
        description="Strong smoothness prior for very smooth objectives.",
        tags=_algorithm_profile(
            what_it_is="Squared exponential kernel imposing a strong smoothness prior.",
            best_for=["very smooth objectives", "continuous process tuning"],
            avoid_when=["abrupt categorical effects dominate"],
            space_support="continuous or encoded mixed spaces",
            data_regime="low data with smooth behavior",
            uncertainty_quality="high",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_phase1",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=["Use sparingly; it is often too smooth for categorical-heavy chemistry search."],
        ),
        factory=None,
    ),
    "smkbo": PoolEntry(
        key="smkbo",
        display_name="SMKBO Spectral Mixture",
        description="Additive Gaussian spectral mixture plus heavy-tailed Cauchy spectral mixture kernel.",
        tags=_algorithm_profile(
            what_it_is="A multi-scale spectral kernel that combines standard and heavy-tailed mixture components.",
            best_for=["multi-scale response surfaces", "quasi-periodic structure", "embedding spaces with heterogeneous smoothness"],
            avoid_when=["very low data", "simple monotone trends where Matern is enough"],
            space_support="continuous or encoded mixed spaces",
            data_regime="best once a few informative observations exist",
            uncertainty_quality="high when fit is stable",
            cost="high",
            interpretability="low-to-medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="custom_phase1",
            fallback_to="matern52",
            fallback_trigger="Optimization becomes unstable or the richer spectral prior is unnecessary.",
            selection_hints=[
                "Try when Matérn kernels underfit oscillatory or multi-scale structure.",
                "More expressive, but usually needs more data than the default kernels.",
            ],
        ),
        factory=None,
    ),
    "smk": PoolEntry(
        key="smk",
        display_name="Spectral Mixture Alias",
        description="Compatibility alias that resolves to the repository's existing smkbo kernel implementation.",
        tags=_algorithm_profile(
            what_it_is="Compatibility alias for the existing SMKBO spectral mixture kernel.",
            best_for=["AutoBO GP-SMK compatibility"],
            avoid_when=["you need to distinguish between multiple spectral kernel implementations"],
            space_support="continuous or encoded mixed spaces",
            data_regime="best once a few informative observations exist",
            uncertainty_quality="high when fit is stable",
            cost="high",
            interpretability="low-to-medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="compat_phase2",
            fallback_to="matern52",
            fallback_trigger="Alias resolves to smkbo and follows its fallback behavior.",
            selection_hints=["Use only as a compatibility key; runtime resolves it to smkbo."],
        ),
        factory=None,
    ),
    "sum_kernel": PoolEntry(
        key="sum_kernel",
        display_name="Additive Mixed Kernel",
        description="Approximate additive kernel for partially separable mixed spaces.",
        tags=_algorithm_profile(
            what_it_is="Approximate additive composition for mixed-space effects.",
            best_for=["mixed spaces with semi-independent effects"],
            avoid_when=["interactions dominate the objective"],
            space_support="encoded mixed spaces",
            data_regime="mid data preferred",
            uncertainty_quality="good",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_phase1",
            fallback_to="matern52",
            fallback_trigger="Fit is unstable or main-effects structure is not supported by the data.",
            selection_hints=["Choose when you expect main effects to dominate interactions."],
        ),
        factory=None,
    ),
    "product_kernel": PoolEntry(
        key="product_kernel",
        display_name="Product Mixed Kernel",
        description="Approximate interaction-heavy mixed kernel.",
        tags=_algorithm_profile(
            what_it_is="Approximate multiplicative composition to express stronger interactions.",
            best_for=["mixed spaces with strong interactions"],
            avoid_when=["data is too sparse to identify interactions"],
            space_support="encoded mixed spaces",
            data_regime="mid data preferred",
            uncertainty_quality="good",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_phase1",
            fallback_to="matern52",
            fallback_trigger="Fit is unstable or interaction structure is unsupported by the data.",
            selection_hints=["Choose only when you have a concrete interaction hypothesis."],
        ),
        factory=None,
    ),
    "mixed_sum_product": PoolEntry(
        key="mixed_sum_product",
        display_name="Mixed Sum-Product Kernel",
        description="Approximate balanced mixed kernel for uncertain interaction structure.",
        tags=_algorithm_profile(
            what_it_is="Weighted blend of additive and multiplicative mixed-space behavior.",
            best_for=["mixed categorical-continuous spaces", "uncertain interaction structure"],
            avoid_when=["problem is purely continuous or purely categorical and simple"],
            space_support="encoded mixed spaces",
            data_regime="mid data preferred",
            uncertainty_quality="good",
            cost="moderate",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="native_phase1",
            fallback_to="matern52",
            fallback_trigger="Fit is unstable or the mixed interaction prior is unnecessary.",
            selection_hints=[
                "Recommended mixed-space kernel when the problem has both main effects and interactions.",
                "A good default for encoded chemistry search spaces if GP is chosen.",
            ],
        ),
        factory=None,
    ),
}


AF_POOL: dict[str, PoolEntry] = {
    "log_ei": PoolEntry(
        key="log_ei",
        display_name="Log Expected Improvement",
        description="Numerically stable default improvement-based acquisition.",
        tags=_algorithm_profile(
            what_it_is="Expected Improvement transformed into log space for better numerical behavior.",
            best_for=["default single-objective BO", "stable low-data optimization"],
            avoid_when=["you need explicit exploration control", "true multi-objective BO"],
            space_support="single-objective",
            data_regime="excellent default across low-to-mid data",
            uncertainty_quality="relies on surrogate uncertainty",
            cost="low",
            interpretability="high",
            dependencies=["botorch"],
            implementation_status="native_phase1",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=[
                "Default acquisition unless another strategy is clearly better justified.",
                "Prefer over classic EI in Phase 1.",
            ],
            batch_support=False,
        ),
        factory=lambda params=None: AcquisitionFunction("log_ei", params),
    ),
    "ucb": PoolEntry(
        key="ucb",
        display_name="Upper Confidence Bound",
        description="Explicit exploration-exploitation control via beta.",
        tags=_algorithm_profile(
            what_it_is="Optimistic score mean + beta * std.",
            best_for=["manual exploration control", "stagnating campaigns that need more exploration"],
            avoid_when=["you want a strong improvement-based default without tuning beta"],
            space_support="single-objective",
            data_regime="general",
            uncertainty_quality="relies on surrogate uncertainty",
            cost="low",
            interpretability="high",
            dependencies=["botorch"],
            implementation_status="native_phase1",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=[
                "Increase beta early; reduce it later.",
                "Useful when reflection says the campaign is over-exploiting.",
            ],
            batch_support=False,
        ),
        factory=lambda params=None: AcquisitionFunction("ucb", params),
    ),
    "ts": PoolEntry(
        key="ts",
        display_name="Thompson Sampling",
        description="Posterior-sampling acquisition that naturally promotes diversity.",
        tags=_algorithm_profile(
            what_it_is="Samples from a surrogate posterior proxy and optimizes the sample.",
            best_for=["diverse exploration", "simple batch diversity via repeated sampling"],
            avoid_when=["strict deterministic ranking is preferred"],
            space_support="single-objective",
            data_regime="general",
            uncertainty_quality="relies on surrogate uncertainty",
            cost="low",
            interpretability="medium",
            dependencies=["botorch"],
            implementation_status="native_phase1",
            fallback_to=None,
            fallback_trigger=None,
            selection_hints=["Prefer when shortlist diversity matters more than pure greedy improvement."],
            batch_support=True,
        ),
        factory=lambda params=None: AcquisitionFunction("ts", params),
    ),
    "qlog_ei": PoolEntry(
        key="qlog_ei",
        display_name="q-Log Expected Improvement",
        description="Batch LogEI label with Phase 1 pointwise fallback.",
        tags=_algorithm_profile(
            what_it_is="Batch LogEI-inspired policy backed by BoTorch posterior scoring.",
            best_for=["batch mode with q > 1"],
            avoid_when=["strict joint batch optimization is required"],
            space_support="single-objective batch",
            data_regime="general",
            uncertainty_quality="relies on surrogate uncertainty",
            cost="medium",
            interpretability="medium",
            dependencies=["torch", "gpytorch", "botorch"],
            implementation_status="pointwise_phase1",
            fallback_to="log_ei",
            fallback_trigger="Phase 1 ranks the pool pointwise instead of solving a joint q-batch acquisition problem.",
            selection_hints=["Use when you want a batch-aware intent while keeping the current pool-ranking workflow."],
            batch_support=True,
        ),
        factory=lambda params=None: AcquisitionFunction("qlog_ei", params),
    ),
}


def create_surrogate(
    key: str,
    search_space: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    kernel_key: str = "matern52",
    kernel_params: dict[str, Any] | None = None,
    feature_spec: dict[str, Any] | None = None,
) -> BaseSurrogateModel:
    entry = SURROGATE_POOL.get(key)
    if entry is None:
        raise ValueError("Unknown surrogate key: '%s'. Valid: gp_cocabo, catboost, deep_ensemble" % key)
    normalized_kernel_key = "smkbo" if str(kernel_key).strip().lower() == "smk" else str(kernel_key).strip().lower()
    if entry.key == "gp_cocabo":
        model = CoCaBOGPSurrogate(
            search_space=search_space,
            kernel_name=normalized_kernel_key or "matern52",
            params=params or {},
            kernel_params=kernel_params or {},
            feature_spec=feature_spec or {},
        )
    elif entry.key == "catboost":
        model = CatBoostSurrogate(search_space=search_space, params=params or {}, feature_spec=feature_spec or {})
    elif entry.key == "deep_ensemble":
        model = DeepEnsembleSurrogate(search_space=search_space, params=params or {}, feature_spec=feature_spec or {})
    else:
        raise ValueError(f"Unsupported surrogate key: {entry.key}")
    model.metadata.setdefault("selected_key", key)
    model.metadata.setdefault("resolved_key", entry.key)
    model.metadata.setdefault("resolved_kernel", normalized_kernel_key if entry.key == "gp_cocabo" else None)
    model.metadata.setdefault("runtime_mode", detect_runtime_capabilities()["runtime_mode"])
    return model


def create_acquisition(key: str, params: dict[str, Any] | None = None) -> AcquisitionFunction:
    entry = AF_POOL.get(key) or AF_POOL["log_ei"]
    acquisition = entry.factory(params or {})
    acquisition.metadata.setdefault("selected_key", key)
    acquisition.metadata.setdefault("resolved_key", entry.key)
    acquisition.metadata.setdefault("runtime_mode", detect_runtime_capabilities()["runtime_mode"])
    return acquisition


def candidate_to_key(candidate: dict[str, Any]) -> str:
    items = tuple(sorted((str(key), _stable_value(value)) for key, value in candidate.items()))
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()


def candidate_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    search_space: list[dict[str, Any]],
) -> float:
    """Mixed Hamming + normalized-continuous distance between two candidates."""
    distance = 0.0
    for variable in search_space:
        name = str(variable.get("name") or "")
        if variable.get("type") == "continuous":
            low, high = _continuous_bounds(variable)
            if _safe_float_or_none(left.get(name)) is None or _safe_float_or_none(right.get(name)) is None:
                continue
            scale = _continuous_scale(variable)
            left_value = _normalize_continuous(left.get(name), low, high, scale)
            right_value = _normalize_continuous(right.get(name), low, high, scale)
            distance += abs(left_value - right_value)
            continue
        distance += 0.0 if str(left.get(name, "")) == str(right.get(name, "")) else 1.0
    return distance


def build_doe_pool(
    variables: list[dict[str, Any]],
    *,
    pool_size: int = 60,
    seed: int = 0,
    observed_keys: set[str] | None = None,
    hard_constraints: list[dict[str, Any]] | None = None,
    candidate_pool: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    target_size = max(1, int(pool_size or 1))
    observed = set(observed_keys or set())
    constraints = list(hard_constraints or [])

    if candidate_pool is not None:
        raw_pool = [dict(candidate) for candidate in candidate_pool]
    else:
        source_pool_size = max(target_size * 4, 32)
        discrete_candidates = enumerate_discrete_candidates(variables, max_candidates=source_pool_size)
        if discrete_candidates:
            raw_pool = discrete_candidates
        else:
            raw_pool = hybrid_sample_candidates(variables, source_pool_size, seed=seed)

    feasible: list[dict[str, Any]] = []
    seen = set(observed)
    for candidate in raw_pool:
        key = candidate_to_key(candidate)
        if key in seen or _candidate_violates_constraints(candidate, constraints):
            continue
        seen.add(key)
        feasible.append(dict(candidate))

    if not feasible:
        return []

    limit = min(target_size, len(feasible))
    seeds = _coverage_seed_candidates(feasible, variables, limit=limit, seed=seed)
    selected_keys = {candidate_to_key(candidate) for candidate in seeds}
    remaining = [candidate for candidate in feasible if candidate_to_key(candidate) not in selected_keys]
    selected = _greedy_maximin_candidates(
        remaining,
        variables,
        limit=limit,
        seed=seed,
        initial_selected=seeds,
    )
    return selected[:limit]


def discrete_search_space_size(
    search_space: list[dict[str, Any]],
    max_candidates: int | None = None,
) -> int | None:
    del max_candidates
    total = 1
    for variable in search_space:
        if variable.get("type", "categorical") == "continuous":
            return None
        domain_size = len(_domain_labels(variable))
        if domain_size == 0:
            return 0
        total *= domain_size
    return total


def enumerate_discrete_candidates(
    search_space: list[dict[str, Any]],
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    total = discrete_search_space_size(search_space, max_candidates=max_candidates)
    if total is None:
        return []
    if max_candidates is not None and total > max_candidates:
        return []

    names = []
    domains = []
    for variable in search_space:
        names.append(variable["name"])
        domains.append(_domain_labels(variable))
    candidates = []
    for values in itertools.product(*domains):
        candidates.append(dict(zip(names, values)))
    return candidates


def hybrid_sample_candidates(
    search_space: list[dict[str, Any]],
    num_samples: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    continuous_vars = [var for var in search_space if var.get("type") == "continuous"]
    categorical_vars = [var for var in search_space if var.get("type") != "continuous"]
    lhs_columns: dict[str, np.ndarray] = {}
    if continuous_vars:
        lhs = _latin_hypercube(len(continuous_vars), num_samples, rng)
        for idx, variable in enumerate(continuous_vars):
            lhs_columns[variable["name"]] = lhs[:, idx]

    candidates = []
    for row_idx in range(num_samples):
        candidate = {}
        for variable in categorical_vars:
            labels = _domain_labels(variable) or ["unknown"]
            candidate[variable["name"]] = str(rng.choice(labels))
        for variable in continuous_vars:
            low, high = _continuous_bounds(variable)
            value = _denormalize_continuous(
                float(lhs_columns[variable["name"]][row_idx]),
                low,
                high,
                _continuous_scale(variable),
            )
            candidate[variable["name"]] = round(value if not (float(low).is_integer() and float(high).is_integer()) else round(value), 6)
        candidates.append(candidate)
    return candidates


def _candidate_violates_constraints(
    candidate: dict[str, Any],
    hard_constraints: list[dict[str, Any]],
) -> bool:
    return any(not constraint.get("check", lambda _: True)(candidate) for constraint in hard_constraints)


def _coverage_seed_candidates(
    pool: list[dict[str, Any]],
    variables: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    categorical_variables = [variable for variable in variables if variable.get("type") != "continuous"]

    for variable in categorical_variables:
        if len(selected) >= limit:
            break
        name = str(variable.get("name") or "")
        value_counts = _categorical_value_counts(pool, name)
        prioritized_values = [
            value
            for value, _count in sorted(value_counts.items(), key=lambda item: (-item[1], item[0]))
        ][: min(len(_domain_labels(variable)), 4)]
        for value in prioritized_values:
            if len(selected) >= limit:
                break
            if any(str(candidate.get(name, "")) == value for candidate in selected):
                continue
            matching = [
                candidate
                for candidate in pool
                if str(candidate.get(name, "")) == value and candidate_to_key(candidate) not in selected_keys
            ]
            if not matching:
                continue
            chosen = _select_next_diverse_candidate(matching, variables, selected, seed)
            key = candidate_to_key(chosen)
            if key in selected_keys:
                continue
            selected.append(dict(chosen))
            selected_keys.add(key)
    return selected


def _categorical_value_counts(pool: list[dict[str, Any]], variable_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in pool:
        value = str(candidate.get(variable_name, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _greedy_maximin_candidates(
    pool: list[dict[str, Any]],
    variables: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    initial_selected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = [dict(candidate) for candidate in (initial_selected or [])]
    selected_keys = {candidate_to_key(candidate) for candidate in selected}
    remaining = [dict(candidate) for candidate in pool if candidate_to_key(candidate) not in selected_keys]

    while remaining and len(selected) < limit:
        chosen = _select_next_diverse_candidate(remaining, variables, selected, seed)
        chosen_key = candidate_to_key(chosen)
        selected.append(dict(chosen))
        remaining = [candidate for candidate in remaining if candidate_to_key(candidate) != chosen_key]
    return selected


def _select_next_diverse_candidate(
    candidates: list[dict[str, Any]],
    variables: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    best_candidate = dict(candidates[0])
    best_score = float("-inf")
    best_tie_breaker = _candidate_seed_tie_breaker(best_candidate, seed)

    for candidate in candidates:
        tie_breaker = _candidate_seed_tie_breaker(candidate, seed)
        if not selected:
            if tie_breaker < best_tie_breaker:
                best_candidate = dict(candidate)
                best_tie_breaker = tie_breaker
            continue
        score = min(candidate_distance(candidate, prior, variables) for prior in selected)
        if score > best_score or (math.isclose(score, best_score) and tie_breaker < best_tie_breaker):
            best_candidate = dict(candidate)
            best_score = score
            best_tie_breaker = tie_breaker
    return best_candidate


def _candidate_seed_tie_breaker(candidate: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{candidate_to_key(candidate)}".encode("utf-8")).hexdigest()


def get_pool_summary(pool: dict[str, PoolEntry]) -> list[dict[str, Any]]:
    capabilities = detect_runtime_capabilities()
    summary = []
    for entry in pool.values():
        tags = dict(entry.tags)
        summary.append(
            {
                "key": entry.key,
                "name": entry.display_name,
                "description": entry.description,
                "tags": tags,
                "availability": _entry_availability(tags, capabilities),
            }
        )
    return summary


def get_surrogate_options() -> list[dict[str, Any]]:
    return get_pool_summary(SURROGATE_POOL)


def get_kernel_options() -> list[dict[str, Any]]:
    return get_pool_summary(KERNEL_POOL)


def get_af_options() -> list[dict[str, Any]]:
    return get_pool_summary(AF_POOL)


def _entry_availability(tags: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    dependencies = tags.get("dependencies", [])
    available = True
    missing = []
    for dependency in dependencies:
        if dependency == "rdkit" and not capabilities["rdkit"]:
            available = False
            missing.append("rdkit")
        if dependency in {"torch", "gpytorch", "botorch", "botorch_optional", "advanced_bo_optional"} and not capabilities["torch_stack"]:
            available = False
            missing.append(dependency)
        if dependency in {"network", "embedding_api"}:
            available = False
            missing.append(dependency)
        if dependency == "scikit-learn" and not _has_module("sklearn"):
            available = False
            missing.append("scikit-learn")
        if dependency == "catboost" and not _has_module("catboost"):
            available = False
            missing.append("catboost")
        if dependency == "scipy" and not _has_module("scipy"):
            available = False
            missing.append("scipy")
    return {
        "is_available": available,
        "runtime_mode": capabilities["runtime_mode"],
        "missing_dependencies": missing,
    }


def _gpytorch_kernel(
    kernel_name: str,
    dim: int,
    metadata: dict[str, Any],
    kernel_params: dict[str, Any] | None = None,
):
    if ScaleKernel is None or MaternKernel is None or RBFKernel is None or AdditiveKernel is None or ProductKernel is None:
        raise RuntimeError("BoTorch kernel dependencies are unavailable")
    kernel_params = kernel_params or {}
    kernel_name = "smkbo" if str(kernel_name).strip().lower() == "smk" else str(kernel_name).strip().lower()
    if kernel_name == "rbf":
        return ScaleKernel(RBFKernel(ard_num_dims=dim))
    if kernel_name == "matern32":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=dim))
    if kernel_name == "smkbo":
        if WrappedSMK is None:
            raise RuntimeError("SMKBO kernel dependencies are unavailable")
        num_mixtures1 = max(0, int(kernel_params.get("num_mixtures1", 5)))
        num_mixtures2 = max(0, int(kernel_params.get("num_mixtures2", 4)))
        metadata.setdefault("notes", []).append(
            "smkbo uses an additive SpectralMixture + CauchyMixture kernel without ScaleKernel."
        )
        return WrappedSMK(
            ard_num_dims=dim,
            num_mixtures1=num_mixtures1,
            num_mixtures2=num_mixtures2,
        )
    if kernel_name == "sum_kernel":
        metadata.setdefault("notes", []).append("sum_kernel uses an additive Matern + RBF kernel in GPyTorch.")
        return ScaleKernel(
            AdditiveKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim),
                RBFKernel(ard_num_dims=dim),
            )
        )
    if kernel_name == "product_kernel":
        metadata.setdefault("notes", []).append("product_kernel uses a multiplicative Matern x RBF kernel in GPyTorch.")
        return ScaleKernel(
            ProductKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim),
                RBFKernel(ard_num_dims=dim),
            )
        )
    if kernel_name == "mixed_sum_product":
        metadata.setdefault("notes", []).append("mixed_sum_product blends additive and multiplicative GPyTorch kernels.")
        return ScaleKernel(
            AdditiveKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim),
                RBFKernel(ard_num_dims=dim),
                ProductKernel(
                    MaternKernel(nu=1.5, ard_num_dims=dim),
                    RBFKernel(ard_num_dims=dim),
                ),
            )
        )
    return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim))


def _build_base_cont_kernel_for_cocabo(
    kernel_name: str,
    n_cont: int,
    metadata: dict[str, Any],
    kernel_params: dict[str, Any] | None = None,
):
    if ScaleKernel is None or MaternKernel is None:
        raise RuntimeError("BoTorch kernel dependencies are unavailable")
    kernel_params = kernel_params or {}
    normalized = "smkbo" if str(kernel_name).strip().lower() == "smk" else str(kernel_name).strip().lower()
    ard_dims = max(int(n_cont), 1)
    if normalized == "matern32":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_dims))
    if normalized == "smkbo":
        if WrappedSMK is None:
            raise RuntimeError("SMKBO kernel dependencies are unavailable")
        metadata.setdefault("notes", []).append("SMK applied only to continuous dimensions inside CoCaBO.")
        return WrappedSMK(
            ard_num_dims=ard_dims,
            num_mixtures1=max(0, int(kernel_params.get("num_mixtures1", 4))),
            num_mixtures2=max(0, int(kernel_params.get("num_mixtures2", 3))),
        )
    return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_dims))


def _to_torch_matrix(X: np.ndarray, device: "torch.device | str | None" = None) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch is unavailable")
    resolved_device = device or _resolve_torch_device()
    if hasattr(X, "detach"):
        return X.to(dtype=torch.double, device=resolved_device)
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return torch.as_tensor(array, dtype=torch.double, device=resolved_device)


def _to_torch_column(y: np.ndarray, device: "torch.device | str | None" = None) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch is unavailable")
    resolved_device = device or _resolve_torch_device()
    array = np.asarray(y, dtype=float).reshape(-1, 1)
    return torch.as_tensor(array, dtype=torch.double, device=resolved_device)


def _continuous_bounds(variable: dict[str, Any]) -> tuple[float, float]:
    domain = variable.get("domain", [0.0, 1.0])
    low = float(domain[0])
    high = float(domain[1])
    if math.isclose(low, high):
        high = low + 1.0
    return low, high


def _continuous_scale(variable: dict[str, Any]) -> str | None:
    scale = str(variable.get("scale") or "").strip().lower().replace("-", "_")
    if scale in {"log", "ln", "log_e", "natural_log"}:
        return "log"
    if scale in {"log2", "log_2"}:
        return "log2"
    if scale in {"log10", "log_10"}:
        return "log10"
    return None


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_transform(value: float, scale: str) -> float:
    if scale == "log2":
        return math.log2(value)
    if scale == "log10":
        return math.log10(value)
    return math.log(value)


def _log_inverse(value: float, scale: str) -> float:
    if scale == "log2":
        return 2.0**value
    if scale == "log10":
        return 10.0**value
    return math.exp(value)


def _scaled_continuous_bounds(low: float, high: float, scale: str | None = None) -> tuple[float, float, str | None]:
    if scale is None:
        return low, high, None
    if low <= 0.0 or high <= 0.0:
        return low, high, None
    return _log_transform(low, scale), _log_transform(high, scale), scale


def _normalize_continuous(value: Any, low: float, high: float, scale: str | None = None) -> float:
    if value is None:
        return 0.5
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return 0.5
    scaled_low, scaled_high, active_scale = _scaled_continuous_bounds(float(low), float(high), scale)
    if active_scale is not None:
        if value_f <= 0.0:
            return 0.5
        value_f = _log_transform(value_f, active_scale)
    span = scaled_high - scaled_low
    if math.isclose(span, 0.0):
        return 0.5
    return float(np.clip((value_f - scaled_low) / span, 0.0, 1.0))


def _denormalize_continuous(value: float, low: float, high: float, scale: str | None = None) -> float:
    scaled_low, scaled_high, active_scale = _scaled_continuous_bounds(float(low), float(high), scale)
    raw = scaled_low + float(np.clip(value, 0.0, 1.0)) * (scaled_high - scaled_low)
    if active_scale is not None:
        raw = _log_inverse(raw, active_scale)
    return round(raw, 6)


def _safe_index(labels: list[Any], value: Any) -> int:
    value_str = str(value)
    for idx, label in enumerate(labels):
        if str(label) == value_str:
            return idx
    return 0


def _domain_labels(variable: dict[str, Any]) -> list[str]:
    labels = []
    for entry in variable.get("domain", []):
        if isinstance(entry, dict):
            labels.append(str(entry.get("label") or entry.get("name") or entry.get("value") or entry))
        else:
            labels.append(str(entry))
    return labels


def _variable_smiles_map(variable: dict[str, Any]) -> dict[str, str]:
    smiles_map = {str(key): str(value) for key, value in variable.get("smiles_map", {}).items()}
    for entry in variable.get("domain", []):
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or entry.get("name") or entry.get("value") or "")
        smiles = str(entry.get("smiles") or "")
        if label and smiles:
            smiles_map[label] = smiles
    return smiles_map


def _fingerprint_from_smiles(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    if Chem is None or rdFingerprintGenerator is None or DataStructs is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(molecule)
    array = np.zeros((n_bits,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def _is_missing_label(label: str) -> bool:
    return str(label).strip().lower() in {"", "n.a.", "n/a", "na", "none", "null", "unknown"}


def _physical_feature_vector_from_label(label: str) -> np.ndarray | None:
    cleaned = str(label).strip()
    if not cleaned:
        return None
    if _is_missing_label(cleaned):
        return np.zeros(len(PHYSICAL_FEATURE_NAMES), dtype=float)

    alias = FORMULA_ALIASES.get(cleaned, cleaned)
    if alias in ELEMENT_PHYSICAL_PROPERTIES:
        return _aggregate_element_features({alias: 1.0})

    parsed = _parse_formula_to_counts(alias)
    if parsed:
        return _aggregate_element_features(parsed)
    return None


def _parse_formula_to_counts(formula: str) -> dict[str, float] | None:
    compact = str(formula).strip()
    if not compact or any(token in compact for token in {"/", ".", "[", "]", "(", ")", "+", "-", "@", "="}):
        return None
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", compact))
    if not matches:
        return None
    reconstructed = "".join(match.group(0) for match in matches)
    if reconstructed != compact:
        return None
    counts: dict[str, float] = {}
    for match in matches:
        symbol = match.group(1)
        if symbol not in ELEMENT_PHYSICAL_PROPERTIES:
            return None
        raw_count = match.group(2)
        count = float(raw_count) if raw_count else 1.0
        counts[symbol] = counts.get(symbol, 0.0) + count
    return counts or None


def _aggregate_element_features(counts: dict[str, float]) -> np.ndarray | None:
    total = float(sum(max(value, 0.0) for value in counts.values()))
    if total <= 0.0:
        return None

    mean_atomic_number = 0.0
    mean_group = 0.0
    mean_period = 0.0
    mean_electronegativity = 0.0
    mean_covalent_radius = 0.0
    metal_fraction = 0.0

    for symbol, count in counts.items():
        data = ELEMENT_PHYSICAL_PROPERTIES.get(symbol)
        if data is None:
            return None
        weight = float(count) / total
        mean_atomic_number += weight * data["atomic_number"]
        mean_group += weight * data["group"]
        mean_period += weight * data["period"]
        mean_electronegativity += weight * data["electronegativity"]
        mean_covalent_radius += weight * data["covalent_radius"]
        metal_fraction += weight * data["is_metal"]

    vector = np.asarray(
        [
            mean_atomic_number / 118.0,
            mean_group / 18.0,
            mean_period / 7.0,
            mean_electronegativity / 4.0,
            mean_covalent_radius / 250.0,
            metal_fraction,
            min(float(len(counts)) / 6.0, 1.0),
        ],
        dtype=float,
    )
    return np.clip(vector, 0.0, 1.0)


def _latin_hypercube(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if dim <= 0 or n <= 0:
        return np.zeros((n, dim), dtype=float)
    result = np.zeros((n, dim), dtype=float)
    for column in range(dim):
        perm = rng.permutation(n)
        result[:, column] = (perm + rng.random(n)) / n
    return result


def _stable_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value
