"""
AutoBO Engine: adaptive surrogate selection with optional LLM-guided review.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from core.robust_surrogate import RobustNLLSurrogate, consume_predictions, update_weights, gaussian_log_ei
from scipy.special import logsumexp

import numpy as np
from langchain_core.messages import AIMessage

from core.autobo_prompts import (
    build_af_strategy_prompt,
    build_acquisition_selection_prompt,
    build_ensemble_sur_selection_prompt,
    build_pure_reasoning_selection_prompt,
    build_pure_reasoning_space_selection_prompt,
    build_surrogate_plausibility_prompt,
    build_unseen_category_coverage_prompt,
)
from core.context_builder import ContextBuilder
from core.dataset_oracle import DatasetOracle
from core.domain import is_hpo_domain
from core.ocm_domain import build_domain_prompt as build_ocm_domain_prompt
from core.ocm_domain import decode_candidate as decode_ocm_candidate
from core.ocm_domain import decode_proposal as decode_ocm_proposal
from core.ocm_domain import load_ocm_domain_spec
from core.suzuki_domain import build_domain_prompt as build_suzuki_domain_prompt
from core.suzuki_domain import decode_candidate as decode_suzuki_candidate
from core.suzuki_domain import decode_proposal as decode_suzuki_proposal
from core.suzuki_domain import load_suzuki_domain_spec
from core.zero_llm_ablation import zero_llm_ablation_enabled
from memory.memory_manager import MemoryManager
from pools.component_pools import (
    BaseSurrogateModel,
    CoCaBOGPSurrogate,
    candidate_to_key,
    create_acquisition,
    create_surrogate,
    detect_runtime_capabilities,
)
from tools.chembo_tools import (
    build_candidate_pool as build_bo_candidate_pool,
    build_diverse_fallback_candidates,
    build_shortlist_from_candidates as build_bo_shortlist_from_candidates,
    dataset_candidate_pool_from_spec,
    dedupe_observations,
)


@dataclass
class SurrogateSpec:
    model_id: str
    surrogate_key: str
    kernel_key: str | None
    params: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    kernel_params: dict[str, Any] = field(default_factory=dict)


DEFAULT_SURROGATE_SPECS: list[SurrogateSpec] = [
    SurrogateSpec("gp_matern52", "gp_cocabo", "matern52", {}, "GP-CoCaBO-Matern-5/2"),
    SurrogateSpec("gp_matern32", "gp_cocabo", "matern32", {}, "GP-CoCaBO-Matern-3/2"),
    SurrogateSpec(
        "gp_smk",
        "gp_cocabo",
        "smk",
        {},
        "GP-CoCaBO-SMK",
        kernel_params={"num_mixtures1": 4, "num_mixtures2": 3},
    ),
    SurrogateSpec("gp_indicator_matern52", "gp_cocabo", "matern52", {}, "GP-CoCaBO-Indicator-Matern-5/2"),
    SurrogateSpec("gp_indicator_matern32", "gp_cocabo", "matern32", {}, "GP-CoCaBO-Indicator-Matern-3/2"),
    SurrogateSpec(
        "gp_indicator_smk",
        "gp_cocabo",
        "smk",
        {},
        "GP-CoCaBO-Indicator-SMK",
        kernel_params={"num_mixtures1": 4, "num_mixtures2": 3},
    ),
    SurrogateSpec(
        "gp_weighted_indicator_matern52",
        "gp_cocabo",
        "matern52",
        {},
        "GP-CoCaBO-WeightedIndicator-Matern-5/2",
        kernel_params={"cat_kernel": "weighted_indicator"},
    ),
    SurrogateSpec(
        "gp_weighted_indicator_matern32",
        "gp_cocabo",
        "matern32",
        {},
        "GP-CoCaBO-WeightedIndicator-Matern-3/2",
        kernel_params={"cat_kernel": "weighted_indicator"},
    ),
    SurrogateSpec(
        "gp_weighted_indicator_smk",
        "gp_cocabo",
        "smk",
        {},
        "GP-CoCaBO-WeightedIndicator-SMK",
        kernel_params={"cat_kernel": "weighted_indicator", "num_mixtures1": 4, "num_mixtures2": 3},
    ),
    SurrogateSpec(
        "gp_exp_hamming_matern52",
        "gp_cocabo",
        "matern52",
        {},
        "GP-CoCaBO-ExpHamming-Matern-5/2",
        kernel_params={"cat_kernel": "exp_hamming"},
    ),
    SurrogateSpec(
        "gp_exp_hamming_matern32",
        "gp_cocabo",
        "matern32",
        {},
        "GP-CoCaBO-ExpHamming-Matern-3/2",
        kernel_params={"cat_kernel": "exp_hamming"},
    ),
    SurrogateSpec(
        "gp_exp_hamming_smk",
        "gp_cocabo",
        "smk",
        {},
        "GP-CoCaBO-ExpHamming-SMK",
        kernel_params={"cat_kernel": "exp_hamming", "num_mixtures1": 4, "num_mixtures2": 3},
    ),
    SurrogateSpec(
        "gp_latent_matern52",
        "gp_cocabo",
        "matern52",
        {},
        "GP-CoCaBO-Latent-Matern-5/2",
        kernel_params={"cat_kernel": "latent", "cat_kernel_params": {"latent_dim": 2, "lengthscale": 1.0}},
    ),
    SurrogateSpec(
        "gp_latent_matern32",
        "gp_cocabo",
        "matern32",
        {},
        "GP-CoCaBO-Latent-Matern-3/2",
        kernel_params={"cat_kernel": "latent", "cat_kernel_params": {"latent_dim": 2, "lengthscale": 1.0}},
    ),
    SurrogateSpec(
        "gp_latent_smk",
        "gp_cocabo",
        "smk",
        {},
        "GP-CoCaBO-Latent-SMK",
        kernel_params={
            "cat_kernel": "latent",
            "cat_kernel_params": {"latent_dim": 2, "lengthscale": 1.0},
            "num_mixtures1": 4,
            "num_mixtures2": 3,
        },
    ),
    SurrogateSpec(
        "catboost",
        "catboost",
        None,
        {"iterations": 150, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "bootstrap_type": "Bayesian"},
        "CatBoost-RMSEWithUncertainty",
    ),
    SurrogateSpec(
        "deep_ensemble",
        "deep_ensemble",
        None,
        {"n_models": 5, "hidden1": 64, "hidden2": 32, "n_epochs": 200, "learning_rate": 1e-3, "weight_decay": 1e-3},
        "DeepEnsemble-5NN",
    ),
]

HPO_CONTINUOUS_GP_SURROGATE_POOL = ["gp_matern52", "gp_matern32", "gp_smk"]
HPO_NON_GP_SURROGATE_POOL = ["catboost", "deep_ensemble"]


def surrogate_specs_from_ids(model_ids: list[str] | None = None) -> list[SurrogateSpec]:
    if not model_ids:
        return list(DEFAULT_SURROGATE_SPECS)
    spec_map = {spec.model_id: spec for spec in DEFAULT_SURROGATE_SPECS}
    return [spec_map[model_id] for model_id in model_ids if model_id in spec_map]


def _is_hpo_problem(problem_spec: dict[str, Any] | None) -> bool:
    return is_hpo_domain(problem_spec if isinstance(problem_spec, dict) else {})


def _effective_autobo_surrogate_pool(
    *,
    problem_spec: dict[str, Any] | None,
    settings,
    switching_enabled: bool,
) -> list[str]:
    if not switching_enabled:
        return [_initial_autobo_active_model(problem_spec=problem_spec, settings=settings, switching_enabled=False)]
    if _robust_nll_enabled(settings):
        gps = HPO_CONTINUOUS_GP_SURROGATE_POOL if _is_hpo_problem(problem_spec) else [
            "gp_indicator_matern52", "gp_exp_hamming_matern52", "gp_indicator_matern32",
        ]
        return list(gps) + ["catboost"]
    configured = [
        str(model_id).strip()
        for model_id in (list(getattr(settings, "autobo_surrogate_pool", [])) or [])
        if str(model_id).strip()
    ]
    if not _is_hpo_problem(problem_spec):
        return configured
    non_gp = [model_id for model_id in configured if model_id in HPO_NON_GP_SURROGATE_POOL]
    if not non_gp:
        non_gp = list(HPO_NON_GP_SURROGATE_POOL)
    return list(HPO_CONTINUOUS_GP_SURROGATE_POOL) + non_gp


def _initial_autobo_active_model(
    *,
    problem_spec: dict[str, Any] | None,
    settings,
    switching_enabled: bool,
) -> str:
    if _is_hpo_problem(problem_spec):
        return "gp_matern52"
    if switching_enabled:
        return str(getattr(settings, "autobo_initial_active", "gp_indicator_matern52") or "gp_indicator_matern52")
    return "gp_indicator_matern52"


def _effective_active_model_id(
    *,
    autobo_state: dict[str, Any],
    problem_spec: dict[str, Any] | None,
    settings,
    switching_enabled: bool,
) -> str:
    active = str(autobo_state.get("active_model") or "").strip()
    if active:
        pool = _effective_autobo_surrogate_pool(
            problem_spec=problem_spec,
            settings=settings,
            switching_enabled=switching_enabled,
        )
        if active in pool:
            return active
    return _initial_autobo_active_model(
        problem_spec=problem_spec,
        settings=settings,
        switching_enabled=switching_enabled,
    )


def _surrogate_spec_for_model_id(model_id: str | None) -> SurrogateSpec | None:
    normalized = str(model_id or "").strip()
    if not normalized:
        return None
    for spec in DEFAULT_SURROGATE_SPECS:
        if spec.model_id == normalized:
            return spec
    return None


def _recorded_categorical_kernel(model_id: str | None) -> str | None:
    normalized = str(model_id or "").strip().lower()
    if not normalized.startswith("gp"):
        return None
    if normalized in {"gp_matern52", "gp_matern32", "gp_smk"}:
        return None
    if "weighted_indicator" in normalized:
        return "weighted_indicator"
    if "exp_hamming" in normalized:
        return "exp_hamming"
    if "latent" in normalized:
        return "latent"
    return "indicator"


def _recorded_continuous_kernel(model_id: str | None) -> str | None:
    normalized = str(model_id or "").strip().lower()
    if not normalized.startswith("gp"):
        return None
    if "smk" in normalized:
        return "smk"
    if "matern32" in normalized:
        return "matern32"
    return "matern52"


def canonical_recorded_surrogate_model_id(model_id: str | None) -> str:
    normalized = str(model_id or "").strip()
    if not normalized:
        return ""
    categorical_kernel = _recorded_categorical_kernel(normalized)
    continuous_kernel = _recorded_continuous_kernel(normalized)
    if categorical_kernel and continuous_kernel:
        return f"gp_{categorical_kernel}_{continuous_kernel}"
    return normalized


def resolve_recorded_kernel_config(
    model_id: str | None,
    *,
    kernel_params: dict[str, Any] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    params = dict(kernel_params or {})
    if not params:
        spec = _surrogate_spec_for_model_id(model_id)
        if spec is not None:
            params = dict(spec.kernel_params or {})
    categorical_kernel = _recorded_categorical_kernel(model_id)
    continuous_kernel = _recorded_continuous_kernel(model_id)
    if categorical_kernel and continuous_kernel:
        payload = {
            "key": f"{categorical_kernel}_{continuous_kernel}",
            "params": params,
            "categorical_kernel": categorical_kernel,
            "continuous_kernel": continuous_kernel,
        }
    elif continuous_kernel:
        payload = {
            "key": continuous_kernel,
            "params": params,
            "categorical_kernel": None,
            "continuous_kernel": continuous_kernel,
        }
    else:
        payload = {
            "key": "none",
            "params": params,
            "categorical_kernel": None,
            "continuous_kernel": None,
        }
    if rationale:
        payload["rationale"] = rationale
    return payload


def resolve_recorded_surrogate_components(
    model_id: str | None,
    *,
    acquisition_function: str | None = None,
    kernel_params: dict[str, Any] | None = None,
    kernel_rationale: str | None = None,
) -> dict[str, Any]:
    payload = {
        "surrogate_model": canonical_recorded_surrogate_model_id(model_id),
        "kernel_config": resolve_recorded_kernel_config(
            model_id,
            kernel_params=kernel_params,
            rationale=kernel_rationale,
        ),
    }
    if acquisition_function is not None:
        payload["acquisition_function"] = acquisition_function
    return payload


@dataclass
class LOOCVResult:
    model_id: str
    mu: np.ndarray
    sigma: np.ndarray
    y_true: np.ndarray


def get_eligible_surrogate_specs(
    all_specs: list[SurrogateSpec],
    n_obs: int,
    settings,
) -> list[SurrogateSpec]:
    return [spec for spec in all_specs if _surrogate_min_observations(spec, settings) <= int(n_obs)]


def _surrogate_min_observations(spec: SurrogateSpec, settings) -> int:
    if _robust_nll_enabled(settings):
        return 0
    if spec.surrogate_key == "catboost":
        return max(1, int(getattr(settings, "autobo_catboost_min_obs", 12) or 12))
    if spec.surrogate_key == "deep_ensemble":
        return max(1, int(getattr(settings, "autobo_nn_min_obs", 20) or 20))
    if spec.model_id.startswith("gp_latent_"):
        return max(1, int(getattr(settings, "autobo_latent_gp_min_obs", 20) or 20))
    return 8


def _gated_out_surrogate_reasons(
    all_specs: list[SurrogateSpec],
    n_obs: int,
    settings,
) -> dict[str, str]:
    gated: dict[str, str] = {}
    for spec in all_specs:
        min_obs = _surrogate_min_observations(spec, settings)
        if int(n_obs) < min_obs:
            gated[spec.model_id] = f"requires >= {min_obs} observations"
    return gated


def _create_surrogate_from_spec(
    spec: SurrogateSpec,
    search_space: list[dict[str, Any]],
    feature_spec: dict[str, Any] | None = None,
    torch_device: str | None = None,
) -> BaseSurrogateModel:
    params = dict(spec.params)
    if torch_device and spec.surrogate_key in {"gp_cocabo", "deep_ensemble"}:
        params.setdefault("torch_device", torch_device)
    params.setdefault("prediction_batch_size", 1024)
    return create_surrogate(
        spec.surrogate_key,
        search_space,
        params,
        spec.kernel_key or "matern52",
        dict(spec.kernel_params),
        feature_spec=feature_spec,
    )


def _primary_torch_device(settings) -> str | None:
    device = getattr(settings, "bo_torch_device", None)
    if device:
        return str(device)
    devices = getattr(settings, "bo_torch_devices", None)
    if isinstance(devices, list) and devices:
        return str(devices[0])
    return None


def _loocv_torch_devices(settings) -> list[str]:
    devices = getattr(settings, "bo_torch_devices", None)
    if isinstance(devices, str):
        return [item.strip() for item in devices.split(",") if item.strip()]
    if isinstance(devices, list):
        return [str(item).strip() for item in devices if str(item).strip()]
    primary = _primary_torch_device(settings)
    return [primary] if primary else []


def _loocv_max_workers(settings, n_specs: int) -> int:
    configured = int(getattr(settings, "autobo_loocv_max_workers", 4) or 1)
    return max(1, min(int(n_specs), configured))


def _autobo_acquisition_function_key(settings) -> str:
    if zero_llm_ablation_enabled(settings):
        return "qlog_ei"
    if not _llm_acquisition_strategy_enabled(settings):
        return "qlog_ei"
    if not _surrogate_switching_enabled(settings):
        return "ensemble_af" if bool(getattr(settings, "ensemble_af", True)) else "qlog_ei"
    if bool(getattr(settings, "ensemble_sur", True)) and not _robust_nll_enabled(settings):
        return "ensemble_sur"
    return "ensemble_af" if bool(getattr(settings, "ensemble_af", True)) else "qlog_ei"


def _robust_nll_enabled(settings) -> bool:
    strategy = str(getattr(settings, "autobo_surrogate_strategy", "robust_nll"))
    if strategy not in {"robust_nll", "legacy"}:
        raise ValueError(f"Unknown surrogate strategy: {strategy}")
    return _surrogate_switching_enabled(settings) and strategy == "robust_nll"


def _surrogate_switching_enabled(settings) -> bool:
    return bool(getattr(settings, "switch_surrogate", True))


def _llm_acquisition_strategy_enabled(settings) -> bool:
    return bool(getattr(settings, "llm_acquisition_strategy_ablation", True))


def _memory_enabled(settings) -> bool:
    return bool(getattr(settings, "memory_ablation", True))


def _harness_enabled(settings) -> bool:
    return bool(getattr(settings, "harness_ablation", True))


def _fixed_surrogate_rationale(problem_spec: dict[str, Any] | None) -> str:
    if _is_hpo_problem(problem_spec):
        return "Fixed-surrogate ablation: use gp_matern52 throughout HPO continuous ordered-grid modeling."
    return "switch_surrogate disabled; fixed gp_indicator_matern52 surrogate is used."


def _fixed_surrogate_kernel_rationale(problem_spec: dict[str, Any] | None) -> str:
    if _is_hpo_problem(problem_spec):
        return "Fixed-surrogate ablation uses Matern-5/2 over continuous ordered-grid HPO variables."
    return "Fixed-surrogate ablation uses indicator categorical kernel plus Matern-5/2 continuous kernel."


def bootstrap_autobo_state(
    *,
    state: dict[str, Any],
    problem_spec: dict[str, Any],
    settings,
    proposal_strategy: str,
) -> dict[str, Any]:
    autobo_state = _resolve_autobo_state(state.get("autobo_state", {}), settings)
    switching_enabled = _surrogate_switching_enabled(settings)
    active_model_id = _effective_active_model_id(
        autobo_state=autobo_state,
        problem_spec=problem_spec,
        settings=settings,
        switching_enabled=switching_enabled,
    )
    recorded_active_model = canonical_recorded_surrogate_model_id(active_model_id)
    acquisition_function_key = _autobo_acquisition_function_key(settings)
    if proposal_strategy == "pure_reasoning_ablation":
        bo_config = _pure_reasoning_bo_config(state)
        effective_config = _pure_reasoning_effective_config(state)
        message = AIMessage(
            content="Bootstrapped pure reasoning ablation runtime: next experiments will be selected directly by the LLM."
        )
        return {
            "messages": [message],
            "bo_config": bo_config,
            "config_history": list(state.get("config_history", [])) + [bo_config],
            "effective_config": effective_config,
            "autobo_state": {**autobo_state, "active_model": active_model_id},
            "log_lines": ["[autobo_bootstrap] pure_reasoning_ablation enabled"],
        }
    resolved_components = resolve_recorded_surrogate_components(
        active_model_id,
        acquisition_function=acquisition_function_key,
        kernel_rationale=(
            "HPOBench continuous ordered-grid GP managed by the AutoBO surrogate controller."
            if _is_hpo_problem(problem_spec)
            else "CoCaBO mixed kernel managed by the AutoBO surrogate controller."
            if switching_enabled
            else _fixed_surrogate_kernel_rationale(problem_spec)
        ),
    )
    bootstrap_kernel_config = {
        "key": "cocabo_adaptive",
        "params": {},
        "categorical_kernel": None if _is_hpo_problem(problem_spec) else "adaptive",
        "continuous_kernel": "adaptive",
        "rationale": (
            "AutoBO surrogate controller selects among continuous HPO GP kernels at runtime."
            if _is_hpo_problem(problem_spec)
            else "AutoBO surrogate controller selects the concrete CoCaBO kernel at runtime."
        ),
    }
    bo_config = {
        "surrogate_model": "autobo_pool" if switching_enabled else resolved_components.get("surrogate_model"),
        "surrogate_params": {},
        "kernel_config": bootstrap_kernel_config if switching_enabled else resolved_components.get("kernel_config"),
        "acquisition_function": acquisition_function_key,
        "af_params": {},
        "rationale": (
            "AutoBO adaptive HPO surrogate pool (continuous GP + CatBoost + Deep Ensemble) with configurable acquisition shortlist generation."
            if _is_hpo_problem(problem_spec)
            else "AutoBO adaptive surrogate pool (CoCaBO GP + CatBoost + Deep Ensemble) with configurable acquisition shortlist generation."
            if switching_enabled
            else _fixed_surrogate_rationale(problem_spec)
        ),
        "confidence": 1.0,
        "config_version": len(state.get("config_history", [])) + 1,
        "validated": True,
        "selection_source": "autobo",
        "selection_diagnostics": {},
        "autobo_active_model": active_model_id,
    }
    effective_config = dict(state.get("effective_config", {}))
    effective_config.update(
        {
            "runtime_mode": detect_runtime_capabilities()["runtime_mode"],
            "proposal_strategy": proposal_strategy,
            "resolved_components": resolved_components,
            "surrogate_model": "autobo_pool" if switching_enabled else resolved_components.get("surrogate_model"),
            "kernel_config": bootstrap_kernel_config if switching_enabled else resolved_components.get("kernel_config"),
            "acquisition_function": acquisition_function_key,
            "selection_source": "autobo",
            "autobo_active_model": active_model_id,
        }
    )
    message = AIMessage(
        content=(
            f"Bootstrapped AutoBO v3 runtime: active={recorded_active_model or active_model_id} "
            + (
                "(CoCaBO GP + CatBoost + Deep Ensemble)"
                if switching_enabled
                else "(fixed surrogate; switch_surrogate disabled)"
            )
        )
    )
    return {
        "messages": [message],
        "bo_config": bo_config,
        "config_history": list(state.get("config_history", [])) + [bo_config],
        "effective_config": effective_config,
        "autobo_state": {**autobo_state, "active_model": active_model_id},
        "log_lines": [
            (
                f"[autobo_bootstrap] active={recorded_active_model or active_model_id} "
                + ("(CoCaBO+CatBoost+DeepEnsemble)" if switching_enabled else "(fixed_surrogate)")
            )
        ],
    }


def _get_deep_ensemble_feature_spec(
    *,
    state: dict[str, Any],
    llm,
    invoke_json_node,
    settings,
) -> dict[str, Any]:
    del settings
    from pools.deep_ensemble_features import build_deep_ensemble_feature_spec_prompt

    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    search_space = list(problem_spec.get("variables", []) or [])
    prompt = build_deep_ensemble_feature_spec_prompt(search_space, problem_spec)
    if not prompt:
        return {"variable_features": {}}
    default = {"variable_features": {}}
    parsed, _, _ = invoke_json_node(llm, state, prompt, default, node_name="run_bo_iteration", lightweight=True)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("variable_features"), dict):
        return default
    return parsed


def run_autobo_iteration(
    *,
    state: dict[str, Any],
    settings,
    llm,
    invoke_json_node,
) -> dict[str, Any]:
    autobo_state = _resolve_autobo_state(state.get("autobo_state", {}), settings)
    zero_llm_mode = zero_llm_ablation_enabled(settings)
    switching_enabled = _surrogate_switching_enabled(settings)
    robust_nll = _robust_nll_enabled(settings)
    observations = list(state.get("observations", []))
    problem_spec = state.get("problem_spec", {})
    variables = problem_spec.get("variables", [])
    direction = state.get("optimization_direction", "maximize")
    active_model_id = _effective_active_model_id(
        autobo_state=autobo_state,
        problem_spec=problem_spec,
        settings=settings,
        switching_enabled=switching_enabled,
    )
    acquisition_function_key = _autobo_acquisition_function_key(settings)
    ensemble_sur_enabled = acquisition_function_key == "ensemble_sur"
    ensemble_af_enabled = acquisition_function_key == "ensemble_af"
    shortlist_limit = max(
        int(getattr(settings, "autobo_acq_top_k", 5) or 5),
        int(getattr(settings, "shortlist_top_k", 5) or 5),
        int(getattr(settings, "batch_size", 1) or 1),
    )
    if ensemble_af_enabled:
        shortlist_limit = max(int(getattr(settings, "batch_size", 1) or 1), min(5, shortlist_limit))
    stagnation_length = _autobo_stagnation_length(state.get("performance_log", []))
    deduped = dedupe_observations(observations)
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in deduped
        if item.get("candidate")
    }
    dataset_spec = problem_spec.get("dataset", {})
    dataset_candidate_pool = dataset_candidate_pool_from_spec(dataset_spec)
    candidate_pool = build_bo_candidate_pool(
        variables,
        observed_keys=observed_keys,
        candidate_pool_size=max(256, shortlist_limit * 32),
        seed=_state_seed(state),
        hard_constraints=[],
        candidate_pool=dataset_candidate_pool,
    )
    if not candidate_pool:
        candidate_pool = build_diverse_fallback_candidates(
            variables,
            n_total=shortlist_limit,
            seed=_state_seed(state),
            hard_constraints=[],
            observed_keys=observed_keys,
            candidate_pool=dataset_candidate_pool,
        )

    if not deduped:
        fallback_shortlist = build_bo_shortlist_from_candidates(candidate_pool[:shortlist_limit], [])
        for index, item in enumerate(fallback_shortlist):
            item["autobo_rank"] = index + 1
        resolved_components = resolve_recorded_surrogate_components(
            active_model_id,
            acquisition_function=acquisition_function_key,
        )
        no_observations_switch_info = {
            "switched": False,
            "switch_type": "no_change",
            "switch_subtype": "no_change",
            "reason": "No observations available",
            "trigger_reason": "no_observations",
            "decision_reason": "No observations available",
        }
        payload = {
            "status": "warm_start_fallback",
            "strategy": "autobo_adaptive",
            "shortlist": fallback_shortlist,
            "recommended_index": 0,
            "candidates": [item["candidate"] for item in fallback_shortlist],
            "predictions": [item["predicted_value"] for item in fallback_shortlist],
            "uncertainties": [item["uncertainty"] for item in fallback_shortlist],
            "acquisition_values": [item["acquisition_value"] for item in fallback_shortlist],
            "resolved_components": resolved_components,
            "metadata": {
                "proposal_strategy": "autobo_adaptive",
                "active_model": resolved_components.get("surrogate_model") or active_model_id,
                "active_model_internal": active_model_id,
                "fit_results": {},
                "trigger_reason": "no_observations",
                "switch_info": no_observations_switch_info,
                "switch_decision": {},
                "candidate_pool_source": "dataset" if dataset_candidate_pool is not None else "search_space",
            },
        }
        return {
            "messages": [
                AIMessage(
                    content=(
                        "AutoBO fallback: no observations available, using a deterministic shortlist "
                        "from the candidate pool."
                    )
                )
            ],
            "proposal_shortlist": fallback_shortlist,
            "payload": payload,
            "effective_config": _effective_config_with_components(
                state,
                active_model_id=active_model_id,
                resolved_components=resolved_components,
                switch_info=no_observations_switch_info,
                trigger_reason="no_observations",
                acquisition_function=acquisition_function_key,
            ),
            "bo_config": _bo_config_with_active_model(state.get("bo_config", {}), active_model_id, acquisition_function_key),
            "autobo_state": {**autobo_state, "active_model": active_model_id},
            "llm_usage": _empty_usage_delta(),
            "log_lines": [
                f"[run_bo_iteration] autobo active={resolved_components.get('surrogate_model') or active_model_id} "
                f"switched=False shortlist={len(fallback_shortlist)}"
            ],
        }

    llm_usage = _empty_usage_delta()
    all_specs = surrogate_specs_from_ids(
        _effective_autobo_surrogate_pool(
            problem_spec=problem_spec,
            settings=settings,
            switching_enabled=switching_enabled,
        )
    )
    spec_lookup = {spec.model_id: spec for spec in all_specs}
    feature_spec = autobo_state.get("deep_ensemble_feature_spec")
    if feature_spec is None:
        de_specs = [spec for spec in all_specs if spec.surrogate_key == "deep_ensemble"]
        de_min = _surrogate_min_observations(de_specs[0], settings) if de_specs else 20
        if zero_llm_mode:
            feature_spec = {}
            autobo_state = {**autobo_state, "deep_ensemble_feature_spec": feature_spec}
        elif de_specs and len(deduped) >= de_min:
            feature_spec = _get_deep_ensemble_feature_spec(
                state=state,
                llm=llm,
                invoke_json_node=invoke_json_node,
                settings=settings,
            )
            autobo_state = {**autobo_state, "deep_ensemble_feature_spec": feature_spec}
        else:
            feature_spec = {}
    elif not isinstance(feature_spec, dict):
        feature_spec = {}
    try:
        tracker = FitnessTracker(
            weights=dict(getattr(settings, "autobo_fitness_weights", {})),
            seq_start_n=0,
            ci_level=float(getattr(settings, "autobo_cal_ci_level", 0.95)),
            coverage_history=autobo_state.get("coverage_history", {}),
            last_loocv_fold_hits=autobo_state.get("last_loocv_fold_hits", {}),
        )
    except TypeError:
        tracker = FitnessTracker(
            weights=dict(getattr(settings, "autobo_fitness_weights", {})),
            seq_start_n=0,
            ci_level=float(getattr(settings, "autobo_cal_ci_level", 0.95)),
        )
    fitted_ids: list[str] = []
    fit_results: dict[str, dict[str, Any]] = {}
    gated_out_models = _gated_out_surrogate_reasons(all_specs, len(deduped), settings)
    for model_id, reason in gated_out_models.items():
        fit_results[model_id] = {"success": False, "error": reason, "stage": "eligibility_gate"}

    warm_start_target = int(state.get("warm_start_target", 0) or 0)
    n_total_obs = len(deduped)
    n_bo_obs = max(0, n_total_obs - warm_start_target)
    last_eval_n = int(autobo_state.get("last_eval_n", -1))
    eval_interval = max(1, int(getattr(settings, "autobo_eval_interval", 5) or 5))
    should_trigger, trigger_reason = _should_trigger_model_schema_evaluation(
        n_total_obs=n_total_obs,
        warm_start_target=warm_start_target,
        last_eval_n=last_eval_n,
        eval_interval=eval_interval,
    )
    if robust_nll:
        should_trigger = False
        trigger_reason = "robust_nll"
    if not switching_enabled:
        should_trigger = False
        trigger_reason = "switch_surrogate_disabled"
    composite: dict[str, FitnessScores] = {}
    switched = False
    switch_info = {
        "switched": False,
        "switch_type": "no_change",
        "switch_subtype": "no_change",
        "from": autobo_state.get("active_model"),
        "to": active_model_id,
        "reason": (
            "Surrogate evaluation not triggered this iteration."
            if switching_enabled
            else _fixed_surrogate_rationale(problem_spec)
        ),
        "trigger_reason": trigger_reason,
        "decision_reason": (
            "Surrogate evaluation not triggered this iteration."
            if switching_enabled
            else _fixed_surrogate_rationale(problem_spec)
        ),
    }
    switch_decision_payload: dict[str, Any] = {}
    if should_trigger:
        eligible_specs = get_eligible_surrogate_specs(all_specs, len(deduped), settings)
        if eligible_specs:
            loocv_scores, eval_fit_results, tracker = _parallel_loocv_evaluate(
                eligible_specs=eligible_specs,
                search_space=variables,
                deduped_observations=deduped,
                feature_spec=feature_spec,
                direction=direction,
                settings=settings,
            )
            fit_results.update(eval_fit_results)
            fitted_ids = list(loocv_scores.keys())
            if fitted_ids:
                composite = tracker.compute_composite(
                    fitted_ids=fitted_ids,
                    f_llm_scores={},
                    effective_llm_weight=0.0,
                )
                ranked = sorted(composite.values(), key=lambda item: item.composite, reverse=True)
                top_score = ranked[0]
                active_score = composite.get(active_model_id)
                min_gap = float(getattr(settings, "autobo_switch_min_gap", 0.10))
                if active_score is None:
                    old_active = active_model_id
                    active_model_id = top_score.model_id
                    switched = True
                    switch_info = {
                        "switched": True,
                        "switch_type": "active_failed",
                        "switch_subtype": "active_failed",
                        "from": old_active,
                        "to": active_model_id,
                        "reason": f"Active model {old_active} failed LOOCV; switched to {active_model_id}.",
                        "trigger_reason": trigger_reason,
                        "decision_reason": f"Active model {old_active} failed LOOCV; switched to {active_model_id}.",
                    }
                elif top_score.model_id != active_model_id and top_score.composite - active_score.composite > min_gap:
                    old_active = active_model_id
                    gap = float(top_score.composite - active_score.composite)
                    active_model_id = top_score.model_id
                    switched = True
                    switch_info = {
                        "switched": True,
                        "switch_type": "deliberate",
                        "switch_subtype": "normal_gap",
                        "from": old_active,
                        "to": active_model_id,
                        "reason": f"Switched from {old_active} to {active_model_id} (composite gap={gap:.3f} > min_gap={min_gap:.2f}).",
                        "trigger_reason": trigger_reason,
                        "decision_reason": f"Composite gap {gap:.3f} exceeded min_gap {min_gap:.2f}.",
                    }
                else:
                    gap = float(top_score.composite - active_score.composite) if active_score is not None else None
                    switch_info = {
                        "switched": False,
                        "switch_type": "no_change",
                        "switch_subtype": "no_change",
                        "from": autobo_state.get("active_model"),
                        "to": active_model_id,
                        "reason": "Active model retained after surrogate evaluation.",
                        "trigger_reason": trigger_reason,
                        "decision_reason": (
                            "Active model remains top-ranked."
                            if top_score.model_id == active_model_id
                            else f"Top challenger gap {gap:.3f} did not exceed min_gap {min_gap:.2f}."
                        ),
                    }
                active_score_after = composite.get(active_model_id)
                switch_decision_payload = {
                    "active_model": autobo_state.get("active_model"),
                    "top_challenger": top_score.model_id,
                    "active_composite": active_score.composite if active_score is not None else None,
                    "selected_active_composite": active_score_after.composite if active_score_after is not None else None,
                    "top_composite": top_score.composite,
                    "gap": (top_score.composite - active_score.composite) if active_score is not None else None,
                    "effective_threshold": min_gap,
                    "switch_subtype": switch_info.get("switch_subtype"),
                    "decision_reason": switch_info.get("decision_reason"),
                }
            else:
                switch_info = {
                    "switched": False,
                    "switch_type": "no_change",
                    "switch_subtype": "no_fit",
                    "from": autobo_state.get("active_model"),
                    "to": active_model_id,
                    "reason": "No surrogate completed LOOCV evaluation.",
                    "trigger_reason": trigger_reason,
                    "decision_reason": "No surrogate completed LOOCV evaluation.",
                }
        else:
            trigger_reason = "no_eligible_surrogate_for_switch"
            switch_info = {
                "switched": False,
                "switch_type": "no_change",
                "switch_subtype": "no_eligible",
                "from": autobo_state.get("active_model"),
                "to": active_model_id,
                "reason": "No eligible surrogate available for switching.",
                "trigger_reason": trigger_reason,
                "decision_reason": "No eligible surrogate available for switching.",
            }
        autobo_state["last_eval_n"] = n_bo_obs

    y_obs = np.asarray([float(item["result"]) for item in deduped], dtype=float)
    y_model = y_obs if direction != "minimize" else -1.0 * y_obs
    scored_candidates = [item.get("candidate", {}) for item in deduped]

    if ensemble_sur_enabled:
        eligible_specs = get_eligible_surrogate_specs(all_specs, len(deduped), settings)
        return _run_ensemble_sur_iteration(
            state=state,
            settings=settings,
            variables=variables,
            direction=direction,
            observations=deduped,
            scored_candidates=scored_candidates,
            y_model=y_model,
            candidate_pool=candidate_pool,
            eligible_specs=eligible_specs,
            feature_spec=feature_spec,
            autobo_state=autobo_state,
            fit_results=fit_results,
            gated_out_models=gated_out_models,
            composite=composite,
            tracker=tracker,
            trigger_reason=trigger_reason if (robust_nll or should_trigger or not fitted_ids) else "no_trigger",
            should_trigger=should_trigger,
            fitted_ids=fitted_ids,
            shortlist_limit=shortlist_limit,
            acquisition_function_key=acquisition_function_key,
        )

    shortlist_only_model_id: str | None = None
    active_model = None
    active_spec = spec_lookup.get(active_model_id) or _surrogate_spec_for_model_id(active_model_id)
    primary_torch_device = _primary_torch_device(settings)
    if active_spec is not None and not robust_nll:
        try:
            active_model = _create_surrogate_from_spec(active_spec, variables, feature_spec, torch_device=primary_torch_device)
            active_model.fit(scored_candidates, y_model)
            fit_results[active_model_id] = {
                "success": True,
                "error": "",
                "stage": "shortlist",
                "torch_device": primary_torch_device,
            }
        except Exception as exc:
            fit_results[active_model_id] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "stage": "shortlist",
                "torch_device": primary_torch_device,
            }

    if robust_nll:
        model_ids = [spec.model_id for spec in all_specs]
        cache = autobo_state.get("robust_nll", {})
        if (cache.get("model_ids") != model_ids
                or cache.get("direction", direction) != direction
                or int(cache.get("observed_count", 0)) > len(observations)):
            cache = {}
        history = consume_predictions(cache, observations, candidate_to_key, direction)
        previous = cache.get("weights", [1.0 / len(model_ids)] * len(model_ids))
        weights = np.asarray(previous, dtype=float)
        if len(observations) > int(cache.get("observed_count", -1)):
            weights = update_weights(history, previous, float(getattr(settings, "robust_nll_kl_penalty", 1.0)))

        def mixture_factory(device=primary_torch_device):
            factories = [
                lambda spec=spec: _create_surrogate_from_spec(spec, variables, None, torch_device=device)
                for spec in all_specs
            ]
            return RobustNLLSurrogate(factories, weights)

        active_model = mixture_factory()
        active_model.fit(scored_candidates, y_model)
        active_model_id = "robust_nll"
        fitted_ids = model_ids
        fit_results.update({mid: {"success": True, "error": "", "stage": "robust_nll"} for mid in model_ids})
        autobo_state["robust_nll"] = {
            "model_ids": model_ids, "weights": weights.tolist(), "history": history,
            "observed_count": len(observations), "direction": direction, "pending": {},
        }
        switch_info.update({
            "to": "robust_nll", "switch_type": "robust_nll", "switch_subtype": "mixture",
            "weights": dict(zip(model_ids, weights.tolist())),
            "reason": "Rolling prequential robust NLL mixture update.",
            "decision_reason": "Rolling prequential robust NLL mixture update.",
        })

    if active_model is None and fitted_ids and switching_enabled:
        if composite:
            active_model_id = max(composite.values(), key=lambda item: item.composite).model_id
        else:
            active_model_id = fitted_ids[0]
        switch_decision_payload = {
            "active_model": autobo_state.get("active_model"),
            "top_challenger": active_model_id,
            "active_composite": None,
            "top_composite": None,
            "gap": None,
            "effective_threshold": None,
            "streak": 0,
            "required_streak": 1,
            "hysteresis_blocked": False,
            "active_distressed": False,
            "switch_subtype": "fallback_no_fit",
            "decision_reason": "Fell back to the first successfully fitted surrogate.",
        }
        switch_info = {
            "switched": True,
            "switch_type": "fallback_no_fit",
            "switch_subtype": "fallback_no_fit",
            "from": autobo_state.get("active_model"),
            "to": active_model_id,
            "reason": "Fell back to the first successfully fitted surrogate.",
            "trigger_reason": trigger_reason,
            "decision_reason": "Fell back to the first successfully fitted surrogate.",
        }
        switched = True
        active_spec = spec_lookup.get(active_model_id) or _surrogate_spec_for_model_id(active_model_id)
        if active_spec is not None:
            try:
                active_model = _create_surrogate_from_spec(active_spec, variables, feature_spec, torch_device=primary_torch_device)
                active_model.fit(scored_candidates, y_model)
                fit_results[active_model_id] = {
                    "success": True,
                    "error": "",
                    "stage": "shortlist_fallback",
                    "torch_device": primary_torch_device,
                }
            except Exception as exc:
                fit_results[active_model_id] = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stage": "shortlist_fallback",
                    "torch_device": primary_torch_device,
                }
    elif not should_trigger and not robust_nll:
        shortlist_only_model_id = active_model_id

    shortlist_raw = []
    coverage_audit: dict[str, Any] = _unseen_category_coverage_skip_audit(
        settings=settings,
        observations=deduped,
        warm_start_target=warm_start_target,
        ensemble_sur_enabled=ensemble_sur_enabled,
        zero_llm_mode=zero_llm_mode,
    )
    prefilter_multiplier = int(getattr(settings, "autobo_shortlist_prefilter_multiplier", 10) or 10)
    hallucination_mode = str(getattr(settings, "autobo_shortlist_hallucination_mode", "kriging_believer"))
    acquisition_flow: AcquisitionFlow | EnsembleAcquisitionFlow
    af_strategy = _default_af_strategy(settings)
    if ensemble_af_enabled:
        af_strategy, af_usage = _resolve_af_strategy(
            state=state,
            settings=settings,
            llm=llm,
            invoke_json_node=invoke_json_node,
            autobo_state={**autobo_state, "active_model": active_model_id},
            stagnation_length=stagnation_length,
            switch_info=switch_info,
            zero_llm_mode=zero_llm_mode,
        )
        llm_usage = _accumulate_usage_delta(llm_usage, af_usage)
        acquisition_flow = EnsembleAcquisitionFlow(
            top_k=shortlist_limit,
            prefilter_multiplier=prefilter_multiplier,
            hallucination_mode=hallucination_mode,
            ucb_beta=af_strategy.get("qucb_beta"),
            af_weights=af_strategy.get("weights"),
            af_strategy_source=str(af_strategy.get("source") or "mechanical_default"),
        )
    else:
        acquisition_flow = AcquisitionFlow(
            top_k=shortlist_limit,
            prefilter_multiplier=prefilter_multiplier,
            hallucination_mode=hallucination_mode,
        )
    if active_model is not None:
        active_spec = spec_lookup.get(active_model_id) or _surrogate_spec_for_model_id(active_model_id)
        refit_model_factory = None
        if robust_nll:
            refit_model_factory = mixture_factory
            cpu_refit_model_factory = lambda: mixture_factory("cpu")
        elif active_spec is not None:
            refit_model_factory = lambda spec=active_spec, ss=variables, fs=feature_spec, td=primary_torch_device: _create_surrogate_from_spec(spec, ss, fs, torch_device=td)
            cpu_refit_model_factory = lambda spec=active_spec, ss=variables, fs=feature_spec: _create_surrogate_from_spec(spec, ss, fs, torch_device="cpu")
        else:
            cpu_refit_model_factory = None
        shortlist_kwargs = {
            "active_model": active_model,
            "refit_model_factory": refit_model_factory,
            "cpu_refit_model_factory": cpu_refit_model_factory,
            "candidate_pool": candidate_pool,
            "observations": deduped,
            "direction": direction,
            "seed": _state_seed(state),
        }
        if ensemble_af_enabled and isinstance(acquisition_flow, EnsembleAcquisitionFlow):
            shortlist_kwargs.update(
                {
                    "iteration": int(state.get("iteration", 0)),
                    "stagnation_length": stagnation_length,
                }
            )
        shortlist_raw = acquisition_flow.propose_candidates(**shortlist_kwargs)
        if _unseen_category_coverage_should_run(
            settings=settings,
            observations=deduped,
            warm_start_target=warm_start_target,
            ensemble_sur_enabled=ensemble_sur_enabled,
            zero_llm_mode=zero_llm_mode,
        ):
            coverage_records, coverage_audit, coverage_usage = _build_llm_guided_unseen_category_coverage_records(
                state=state,
                settings=settings,
                llm=llm,
                invoke_json_node=invoke_json_node,
                active_model=active_model,
                candidate_pool=candidate_pool,
                observations=deduped,
                search_space=variables,
                direction=direction,
                normal_shortlist=shortlist_raw,
                top_k=shortlist_limit,
            )
            llm_usage = _accumulate_usage_delta(llm_usage, coverage_usage)
            shortlist_raw = _merge_shortlist_with_coverage(
                shortlist_raw,
                coverage_records,
                top_k=shortlist_limit,
                coverage_slots=int(getattr(settings, "autobo_unseen_category_slots", 1) or 1),
            )

    if shortlist_raw:
        shortlist = [
            {
                "candidate": item["candidate"],
                "predicted_value": item["predicted_value"],
                "uncertainty": item["uncertainty"],
                "acquisition_value": item["acquisition_value"],
                "acquisition_value_raw": item.get("acquisition_value_raw"),
                "selection_step": item.get("selection_step"),
                "selection_mode": item.get("selection_mode"),
                "af_sources": list(item.get("af_sources", [])) if isinstance(item.get("af_sources"), list) else [],
                "af_ranks": dict(item.get("af_ranks", {})) if isinstance(item.get("af_ranks"), dict) else {},
                "af_consensus_count": int(item.get("af_consensus_count", 0) or 0),
                "ensemble_reference_score": item.get("ensemble_reference_score"),
                "ensemble_weighted_rank_score": item.get("ensemble_weighted_rank_score"),
                "ensemble_diversity_bonus": item.get("ensemble_diversity_bonus"),
                "coverage_targets": list(item.get("coverage_targets", [])) if isinstance(item.get("coverage_targets"), list) else [],
                "coverage_domain_size": item.get("coverage_domain_size"),
                "constraint_violations": [],
                "constraint_satisfied": True,
                "autobo_rank": item["rank"],
            }
            for item in shortlist_raw
        ]
        status = "shortlist_only_fallback" if shortlist_only_model_id else "success"
    else:
        shortlist = build_bo_shortlist_from_candidates(candidate_pool[:shortlist_limit], [])
        for index, item in enumerate(shortlist):
            item["autobo_rank"] = index + 1
        status = "fallback"
    if robust_nll:
        # Forecast the actual selectable points using the unfantasized models.
        candidates = [item["candidate"] for item in shortlist]
        means, stds = active_model.component_predictions(candidates, observation_noise=True)
        autobo_state["robust_nll"]["pending"] = {
            candidate_to_key(candidate): {"mean": means[:, i].tolist(), "std": stds[:, i].tolist()}
            for i, candidate in enumerate(candidates)
        }

    calibration_entry = {
        "iteration": int(state.get("iteration", 0)),
        "active_model": active_model_id,
        "coverage": {
            model_id: _recent_calibration_coverage(getattr(tracker, "coverage_history", {}).get(model_id, []))
            for model_id in fitted_ids
        },
        "trigger_reason": trigger_reason if (robust_nll or should_trigger or not fitted_ids) else "no_trigger",
    }
    fitness_entry = {
        model_id: {
            "f_seq": score.f_seq,
            "f_cal": score.f_cal,
            "f_rank": score.f_rank,
            "f_llm": score.f_llm,
            "composite": score.composite,
        }
        for model_id, score in composite.items()
    }
    fitness_log = dict(autobo_state.get("fitness_log", {}))
    if switching_enabled:
        fitness_log[str(int(state.get("iteration", 0)))] = fitness_entry
    switch_history = list(autobo_state.get("switch_history", []))
    if switched:
        switch_history.append(
            {
                "iteration": int(state.get("iteration", 0)),
                **switch_info,
                "scores": {model_id: score.composite for model_id, score in composite.items()},
            }
        )

    resolved_components = resolve_recorded_surrogate_components(
        active_model_id,
        acquisition_function=acquisition_function_key,
    )
    payload = {
        "status": status,
        "strategy": "autobo_adaptive",
        "shortlist": shortlist,
        "recommended_index": 0,
        "candidates": [item["candidate"] for item in shortlist],
        "predictions": [item["predicted_value"] for item in shortlist],
        "uncertainties": [item["uncertainty"] for item in shortlist],
        "acquisition_values": [item["acquisition_value"] for item in shortlist],
        "resolved_components": resolved_components,
        "metadata": {
            "proposal_strategy": "autobo_adaptive",
            "active_model": resolved_components.get("surrogate_model") or active_model_id,
            "active_model_internal": active_model_id,
            "fit_results": fit_results,
            "trigger_reason": trigger_reason if (robust_nll or should_trigger or not fitted_ids) else "no_trigger",
            "switch_info": switch_info,
            "switch_decision": switch_decision_payload,
            "surrogate_weights": switch_info.get("weights", {}),
            "candidate_pool_source": "dataset" if dataset_candidate_pool is not None else "search_space",
            "shortlist_prefilter_size": acquisition_flow.last_prefilter_size,
            "shortlist_hallucination_mode": acquisition_flow.hallucination_mode,
            "ensemble_af_enabled": ensemble_af_enabled,
            "af_slot_targets": getattr(acquisition_flow, "last_af_slot_targets", {}),
            "af_slot_filled": getattr(acquisition_flow, "last_af_slot_filled", {}),
            "af_failures": getattr(acquisition_flow, "last_af_failures", []),
            "af_strategy": af_strategy if ensemble_af_enabled else {},
            "af_strategy_source": getattr(acquisition_flow, "af_strategy_source", "none"),
            "ucb_beta": getattr(acquisition_flow, "last_ucb_beta", None),
            "ucb_sigma_multiplier": getattr(acquisition_flow, "last_ucb_sigma_multiplier", None),
            "gated_out_models": gated_out_models,
            "shortlist_only_model": shortlist_only_model_id,
            "stagnation_length": stagnation_length,
            "unseen_category_coverage": coverage_audit,
            "unseen_grid_coverage": coverage_audit if _is_hpo_problem(problem_spec) else {},
        },
    }
    next_autobo_state = {
        **autobo_state,
        "active_model": active_model_id,
        "deep_ensemble_feature_spec": autobo_state.get("deep_ensemble_feature_spec"),
        "fitness_log": (
            _trim_autobo_mapping(fitness_log, limit=50)
            if switching_enabled
            else _trim_autobo_mapping(dict(autobo_state.get("fitness_log", {})), limit=50)
        ),
        "calibration_log": (
            _trim_autobo_list(list(autobo_state.get("calibration_log", [])) + [calibration_entry], limit=50)
            if switching_enabled
            else _trim_autobo_list(list(autobo_state.get("calibration_log", [])), limit=50)
        ),
        "switch_history": _trim_autobo_list(switch_history, limit=50),
        "last_layer2_iteration": int(state.get("iteration", 0)) if should_trigger else int(autobo_state.get("last_layer2_iteration", 0)),
        "hysteresis_until": 0,
        "last_eval_n": int(autobo_state.get("last_eval_n", -1)),
        "effective_llm_weight": 0.0,
        "coverage_history": {
            key: list(value)[-20:]
            for key, value in getattr(tracker, "coverage_history", {}).items()
        } if switching_enabled else dict(autobo_state.get("coverage_history", {})),
        "last_loocv_fold_hits": {
            key: list(value)
            for key, value in getattr(tracker, "last_loocv_fold_hits", getattr(tracker, "cal_log", {})).items()
        } if switching_enabled else dict(autobo_state.get("last_loocv_fold_hits", {})),
        "challenger_lead_streak": {},
        "af_strategy": af_strategy if ensemble_af_enabled else dict(autobo_state.get("af_strategy", {})),
        "llm_plaus_audit": list(autobo_state.get("llm_plaus_audit", [])),
    }
    message = AIMessage(
        content=(
            f"AutoBO iter={state.get('iteration', 0)} active={resolved_components.get('surrogate_model') or active_model_id} "
            f"fitted={len(fitted_ids)} shortlist={len(shortlist)} "
            f"stagnation={stagnation_length} {switch_info['reason']}"
        )
    )
    return {
        "messages": [message],
        "proposal_shortlist": shortlist,
        "payload": payload,
        "effective_config": _effective_config_with_components(
            state,
            active_model_id=active_model_id,
            resolved_components=resolved_components,
            switch_info=switch_info,
            trigger_reason=trigger_reason,
            switch_decision=switch_decision_payload,
            acquisition_function=acquisition_function_key,
        ),
        "bo_config": _bo_config_with_active_model(state.get("bo_config", {}), active_model_id, acquisition_function_key),
        "autobo_state": next_autobo_state,
        "llm_usage": llm_usage,
        "log_lines": [
            f"[run_bo_iteration] autobo active={resolved_components.get('surrogate_model') or active_model_id} switched={switched} "
            f"shortlist={len(shortlist)} stagnation={stagnation_length}"
        ],
    }


def _run_ensemble_sur_iteration(
    *,
    state: dict[str, Any],
    settings,
    variables: list[dict[str, Any]],
    direction: str,
    observations: list[dict[str, Any]],
    scored_candidates: list[dict[str, Any]],
    y_model: np.ndarray,
    candidate_pool: list[dict[str, Any]],
    eligible_specs: list[SurrogateSpec],
    feature_spec: dict[str, Any] | None,
    autobo_state: dict[str, Any],
    fit_results: dict[str, dict[str, Any]],
    gated_out_models: dict[str, str],
    composite: dict[str, FitnessScores],
    tracker: FitnessTracker,
    trigger_reason: str,
    should_trigger: bool,
    fitted_ids: list[str],
    shortlist_limit: int,
    acquisition_function_key: str,
) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    primary_torch_device = _primary_torch_device(settings)
    devices = _loocv_torch_devices(settings)
    max_workers = _loocv_max_workers(settings, max(len(eligible_specs), 1))
    fitted_models: dict[str, BaseSurrogateModel] = {}

    def _fit_one(index: int, spec: SurrogateSpec) -> tuple[str, BaseSurrogateModel | None, dict[str, Any]]:
        torch_device = devices[index % len(devices)] if devices else primary_torch_device
        try:
            model = _create_surrogate_from_spec(spec, variables, feature_spec, torch_device=torch_device)
            model.fit(scored_candidates, y_model)
            return spec.model_id, model, {"success": True, "error": "", "stage": "ensemble_sur_fit", "torch_device": torch_device}
        except Exception as exc:
            return spec.model_id, None, {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "stage": "ensemble_sur_fit",
                "torch_device": torch_device,
            }

    if len(eligible_specs) <= 1 or max_workers <= 1:
        fit_outputs = [_fit_one(index, spec) for index, spec in enumerate(eligible_specs)]
    else:
        fit_outputs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fit_one, index, spec) for index, spec in enumerate(eligible_specs)]
            for future in as_completed(futures):
                fit_outputs.append(future.result())

    for model_id, model, result in fit_outputs:
        fit_results[model_id] = result
        if model is not None:
            fitted_models[model_id] = model

    scale_context = _build_observation_scale_context(observations, direction=direction)
    per_model_scores: dict[str, dict[str, Any]] = {}
    proposed: dict[str, dict[str, Any]] = {}
    for model_id, model in fitted_models.items():
        try:
            scores = _score_candidate_pool(
                surrogate=model,
                candidate_pool=candidate_pool,
                best_f_scaled=float(scale_context.get("best_f_scaled", 0.0) or 0.0),
                y_mean=0.0,
                y_std=1.0,
                direction=direction,
                seed=_state_seed(state) + len(per_model_scores) * 997,
            )
        except Exception as exc:
            fit_results[model_id] = {
                **dict(fit_results.get(model_id, {})),
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "stage": "ensemble_sur_score",
            }
            continue
        per_model_scores[model_id] = scores
        acquisition = np.asarray(scores.get("acquisition", []), dtype=float)
        if len(acquisition) == 0:
            continue
        order = np.argsort(acquisition)[::-1]
        top_index = int(order[0])
        candidate = dict(scores["candidate_pool"][top_index])
        key = candidate_to_key(candidate)
        record = proposed.setdefault(
            key,
            {
                "candidate": candidate,
                "proposed_by": [],
                "_candidate_key": key,
                "_best_proposer_rank": 10**9,
                "_best_proposer_composite": float("-inf"),
            },
        )
        record["proposed_by"].append(model_id)
        record["_best_proposer_rank"] = min(int(record["_best_proposer_rank"]), 1)
        composite_value = _composite_value_for_model(model_id, composite, autobo_state)
        if composite_value is not None:
            record["_best_proposer_composite"] = max(float(record["_best_proposer_composite"]), float(composite_value))

    if not proposed:
        fallback_shortlist = build_bo_shortlist_from_candidates(candidate_pool[:shortlist_limit], [])
        for index, item in enumerate(fallback_shortlist):
            item["autobo_rank"] = index + 1
            item["selection_mode"] = "ensemble_sur_fallback"
        status = "fallback"
    else:
        candidate_records = _attach_ensemble_sur_cross_scores(
            proposed_records=list(proposed.values()),
            per_model_scores=per_model_scores,
            direction=direction,
            top_k=shortlist_limit,
        )
        candidate_records.sort(
            key=lambda item: (
                -int(item.get("surrogate_consensus_count", 0) or 0),
                -float(item.get("_best_proposer_composite", float("-inf"))),
                int(item.get("_best_cross_rank", 10**9) or 10**9),
                str(item.get("_candidate_key", "")),
            )
        )
        fallback_shortlist = []
        for index, item in enumerate(candidate_records[:shortlist_limit]):
            item["selection_step"] = index + 1
            item["selection_mode"] = "ensemble_sur_candidate" if index else "ensemble_sur_reference"
            item["rank"] = index + 1
            item["autobo_rank"] = index + 1
            item["af_sources"] = []
            item["af_ranks"] = {}
            item["af_consensus_count"] = int(item.get("surrogate_consensus_count", 0) or 0)
            item.pop("_candidate_key", None)
            item.pop("_best_cross_rank", None)
            item.pop("_best_proposer_rank", None)
            item.pop("_best_proposer_composite", None)
            fallback_shortlist.append(item)
        status = "success"

    composite_summary = _ensemble_sur_composite_summary(
        composite=composite,
        autobo_state=autobo_state,
        model_ids=list(fitted_models.keys()),
        current=bool(should_trigger and composite),
    )
    calibration_entry = {
        "iteration": int(state.get("iteration", 0)),
        "active_model": "ensemble_sur",
        "coverage": {
            model_id: _recent_calibration_coverage(getattr(tracker, "coverage_history", {}).get(model_id, []))
            for model_id in fitted_ids
        },
        "trigger_reason": trigger_reason,
    }
    fitness_entry = {
        model_id: {
            "f_seq": score.f_seq,
            "f_cal": score.f_cal,
            "f_rank": score.f_rank,
            "f_llm": score.f_llm,
            "composite": score.composite,
        }
        for model_id, score in composite.items()
    }
    fitness_log = dict(autobo_state.get("fitness_log", {}))
    if fitness_entry:
        fitness_log[str(int(state.get("iteration", 0)))] = fitness_entry

    resolved_components = {
        "surrogate_model": "ensemble_sur",
        "kernel_config": {
            "key": "multi_surrogate",
            "params": {},
            "categorical_kernel": "per_surrogate",
            "continuous_kernel": "per_surrogate",
            "rationale": "Each eligible surrogate proposes one fixed-LogEI candidate; the LLM chooses among the merged candidates.",
        },
        "acquisition_function": acquisition_function_key,
    }
    no_switch_info = {
        "switched": False,
        "switch_type": "ensemble_sur",
        "switch_subtype": "multi_surrogate",
        "from": autobo_state.get("active_model"),
        "to": "ensemble_sur",
        "reason": "ensemble_sur fits multiple surrogates each round instead of switching one active surrogate.",
        "trigger_reason": trigger_reason,
        "decision_reason": "Surrogate switching is bypassed in ensemble_sur mode.",
    }
    payload = {
        "status": status,
        "strategy": "ensemble_sur",
        "shortlist": fallback_shortlist,
        "recommended_index": 0,
        "candidates": [item.get("candidate", {}) for item in fallback_shortlist],
        "predictions": [item.get("predicted_value") for item in fallback_shortlist],
        "uncertainties": [item.get("uncertainty") for item in fallback_shortlist],
        "acquisition_values": [item.get("acquisition_value") for item in fallback_shortlist],
        "resolved_components": resolved_components,
        "metadata": {
            "proposal_strategy": "ensemble_sur",
            "active_model": "ensemble_sur",
            "active_model_internal": "ensemble_sur",
            "fit_results": fit_results,
            "trigger_reason": trigger_reason,
            "switch_info": no_switch_info,
            "switch_decision": {},
            "candidate_pool_source": "dataset" if dataset_candidate_pool_from_spec(state.get("problem_spec", {}).get("dataset", {})) is not None else "search_space",
            "ensemble_sur_enabled": True,
            "ensemble_af_enabled": False,
            "surrogate_composite_summary": composite_summary,
            "surrogate_composite_explanation": "composite is a recent LOOCV confidence score; larger means the surrogate has been more reliable recently.",
            "gated_out_models": gated_out_models,
            "stagnation_length": _autobo_stagnation_length(state.get("performance_log", [])),
        },
    }
    next_autobo_state = {
        **autobo_state,
        "active_model": "ensemble_sur",
        "fitness_log": _trim_autobo_mapping(fitness_log, limit=50),
        "calibration_log": _trim_autobo_list(list(autobo_state.get("calibration_log", [])) + [calibration_entry], limit=50),
        "last_layer2_iteration": int(state.get("iteration", 0)) if should_trigger else int(autobo_state.get("last_layer2_iteration", 0)),
        "last_eval_n": int(autobo_state.get("last_eval_n", -1)),
        "coverage_history": {
            key: list(value)[-20:]
            for key, value in getattr(tracker, "coverage_history", {}).items()
        },
        "last_loocv_fold_hits": {
            key: list(value)
            for key, value in getattr(tracker, "last_loocv_fold_hits", getattr(tracker, "cal_log", {})).items()
        },
        "effective_llm_weight": 0.0,
        "af_strategy": {},
    }
    message = AIMessage(
        content=(
            f"AutoBO iter={state.get('iteration', 0)} ensemble_sur fitted={len(fitted_models)} "
            f"shortlist={len(fallback_shortlist)} trigger={trigger_reason}"
        )
    )
    return {
        "messages": [message],
        "proposal_shortlist": fallback_shortlist,
        "payload": payload,
        "effective_config": _effective_config_with_components(
            state,
            active_model_id="ensemble_sur",
            resolved_components=resolved_components,
            switch_info=no_switch_info,
            trigger_reason=trigger_reason,
            acquisition_function=acquisition_function_key,
        ),
        "bo_config": _bo_config_with_active_model(state.get("bo_config", {}), "ensemble_sur", acquisition_function_key),
        "autobo_state": next_autobo_state,
        "llm_usage": _empty_usage_delta(),
        "log_lines": [
            f"[run_bo_iteration] ensemble_sur fitted={len(fitted_models)} shortlist={len(fallback_shortlist)} trigger={trigger_reason}"
        ],
    }


def _attach_ensemble_sur_cross_scores(
    *,
    proposed_records: list[dict[str, Any]],
    per_model_scores: dict[str, dict[str, Any]],
    direction: str,
    top_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in proposed_records:
        candidate = dict(record.get("candidate", {}))
        key = str(record.get("_candidate_key") or candidate_to_key(candidate))
        cross_scores: dict[str, dict[str, Any]] = {}
        display_mus: list[float] = []
        display_sigmas: list[float] = []
        display_logei: list[float] = []
        best_rank = 10**9
        for model_id, scores in per_model_scores.items():
            candidate_pool = list(scores.get("candidate_pool", []))
            index = next((idx for idx, item in enumerate(candidate_pool) if candidate_to_key(item) == key), None)
            if index is None:
                continue
            acquisition = np.asarray(scores.get("acquisition", []), dtype=float)
            order = np.argsort(acquisition)[::-1]
            rank_lookup = {int(candidate_index): rank + 1 for rank, candidate_index in enumerate(order)}
            rank = int(rank_lookup.get(index, len(order) + 1))
            mu = float(np.asarray(scores["pred_mean"], dtype=float)[index])
            sigma = float(np.asarray(scores["pred_std"], dtype=float)[index])
            logei = float(acquisition[index])
            best_rank = min(best_rank, rank)
            display_mus.append(mu)
            display_sigmas.append(sigma)
            display_logei.append(logei)
            cross_scores[model_id] = {
                "mu": mu,
                "sigma": sigma,
                "logei": logei,
                "rank": rank,
                "proposed": model_id in set(record.get("proposed_by", [])),
            }
        predicted_value = float(np.mean(display_mus)) if display_mus else None
        uncertainty = float(np.mean(display_sigmas)) if display_sigmas else None
        acquisition_value = float(np.mean(display_logei)) if display_logei else None
        records.append(
            {
                **record,
                "candidate": candidate,
                "proposed_by": list(record.get("proposed_by", [])),
                "surrogate_consensus_count": len(record.get("proposed_by", [])),
                "surrogate_cross_scores": cross_scores,
                "predicted_value": predicted_value,
                "uncertainty": uncertainty,
                "acquisition_value": acquisition_value,
                "acquisition_value_raw": acquisition_value,
                "_best_cross_rank": best_rank,
            }
        )
    return records


def _composite_value_for_model(
    model_id: str,
    composite: dict[str, FitnessScores],
    autobo_state: dict[str, Any],
) -> float | None:
    score = composite.get(model_id)
    if score is not None:
        return float(score.composite)
    fitness_log = autobo_state.get("fitness_log", {}) if isinstance(autobo_state.get("fitness_log"), dict) else {}
    numeric_keys = [key for key in fitness_log if str(key).isdigit()]
    if not numeric_keys:
        return None
    latest = fitness_log[max(numeric_keys, key=lambda key: int(str(key)))]
    if not isinstance(latest, dict) or not isinstance(latest.get(model_id), dict):
        return None
    value = latest[model_id].get("composite")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensemble_sur_composite_summary(
    *,
    composite: dict[str, FitnessScores],
    autobo_state: dict[str, Any],
    model_ids: list[str],
    current: bool,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model_id in model_ids:
        value = _composite_value_for_model(model_id, composite, autobo_state)
        summary.append(
            {
                "model_id": model_id,
                "composite": round(float(value), 4) if value is not None else None,
                "status": "current" if current and model_id in composite else "stale" if value is not None else "unavailable",
            }
        )
    summary.sort(
        key=lambda item: (
            item.get("composite") is None,
            -float(item.get("composite") if item.get("composite") is not None else -10**9),
            str(item.get("model_id") or ""),
        )
    )
    return summary


def select_autobo_candidate(
    *,
    state: dict[str, Any],
    settings,
    llm,
    invoke_json_node,
) -> dict[str, Any]:
    shortlist = list(state.get("proposal_shortlist", []))
    zero_llm_mode = zero_llm_ablation_enabled(settings)
    state_payload = {}
    if isinstance(state.get("last_tool_payload"), dict):
        state_payload = state.get("last_tool_payload", {})
    elif isinstance(state.get("payload"), dict):
        state_payload = state.get("payload", {})
    runtime_metadata = state_payload.get("metadata", {}) if isinstance(state_payload.get("metadata"), dict) else {}
    state_effective = state.get("effective_config", {}) if isinstance(state.get("effective_config"), dict) else {}
    ensemble_sur_mode = bool(runtime_metadata.get("ensemble_sur_enabled"))
    if not ensemble_sur_mode:
        ensemble_sur_mode = bool(
            state_effective.get("acquisition_function") == "ensemble_sur"
            or any(isinstance(item.get("surrogate_cross_scores"), dict) and item.get("surrogate_cross_scores") for item in shortlist)
        )
    ensemble_mode = bool(runtime_metadata.get("ensemble_af_enabled"))
    if not ensemble_mode:
        ensemble_mode = bool(
            state_effective.get("acquisition_function") == "ensemble_af"
            or any(isinstance(item.get("af_sources"), list) and item.get("af_sources") for item in shortlist)
        )
    if not shortlist:
        return {
            "messages": [AIMessage(content="AutoBO shortlist is empty; no candidate could be selected.")],
            "proposal_selected": {
                "selected_index": 0,
                "override": False,
                "candidate": {},
                "rationale": {
                    "chemical_reasoning": "AutoBO shortlist was empty.",
                    "comparison_to_top1": "",
                    "selection_mode": "top1_follow",
                    "hypothesis_alignment": "",
                    "information_value": "",
                    "concerns": "",
                    "override_evidence": {
                        "evidence_type": "none",
                        "evidence_ids": [],
                        "trajectory_references": [],
                        "chemistry_argument": "",
                        "validated": False,
                    },
                },
                "confidence": 0.0,
                "selection_source": "autobo_empty_shortlist",
                "selected_rank": 0,
                "top1_candidate": {},
            },
            "current_proposal": {"candidates": [{}], "selected_index": 0},
            "llm_usage": _empty_usage_delta(),
            "log_lines": ["[select_candidate] autobo shortlist empty"],
        }

    deterministic_logei_mode = (
        zero_llm_mode
        or not _llm_acquisition_strategy_enabled(settings)
        or not bool(getattr(settings, "autobo_llm_acq_enabled", True))
    )
    if deterministic_logei_mode:
        selected_record = shortlist[0]
        candidate = selected_record.get("candidate", {})
        af_sources = list(selected_record.get("af_sources", [])) if isinstance(selected_record.get("af_sources"), list) else []
        af_ranks = dict(selected_record.get("af_ranks", {})) if isinstance(selected_record.get("af_ranks"), dict) else {}
        coverage_targets = (
            list(selected_record.get("coverage_targets", []))
            if isinstance(selected_record.get("coverage_targets"), list)
            else []
        )
        qlogei_rank = None
        if ensemble_sur_mode:
            qlogei_rank = None
        elif not ensemble_mode:
            qlogei_rank = 1
        else:
            qlogei_rank = af_ranks.get("qlogei")
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Zero-LLM AutoBO mode: using shortlist rank-1 qLogEI candidate."
                        if zero_llm_mode
                        else "LLM acquisition strategy ablation disabled; using rank-1 LogEI candidate."
                        if not _llm_acquisition_strategy_enabled(settings)
                        else "AutoBO LLM acquisition disabled; using ensemble-sur reference candidate."
                        if ensemble_sur_mode
                        else "AutoBO LLM acquisition disabled; using shortlist rank-1 ensemble reference candidate."
                        if ensemble_mode
                        else "AutoBO LLM acquisition disabled; using shortlist rank-1 raw acquisition candidate."
                    )
                )
            ],
            "proposal_selected": {
                "selected_index": 0,
                "override": False,
                "candidate": candidate,
                "rationale": {
                    "chemical_reasoning": "Selected the highest-ranked AutoBO shortlist candidate.",
                    "comparison_to_top1": (
                        "Candidate #1 is accepted as the deterministic qLogEI top-1 choice."
                        if zero_llm_mode or not _llm_acquisition_strategy_enabled(settings)
                        else (
                        "Candidate #1 is accepted as the current ensemble-sur reference choice."
                        if ensemble_sur_mode
                        else (
                        "Candidate #1 is accepted as the best current ensemble reference choice."
                        if ensemble_mode
                        else "Candidate #1 is accepted as the best current choice."
                        )
                        )
                    ),
                    "selection_mode": (
                        "qlogei_top1_follow"
                        if zero_llm_mode or not _llm_acquisition_strategy_enabled(settings)
                        else "ensemble_sur_reference_follow"
                        if ensemble_sur_mode
                        else "top1_follow"
                    ),
                    "hypothesis_alignment": "",
                    "information_value": "",
                    "concerns": "",
                },
                "confidence": 1.0,
                "selection_source": (
                    "autobo_qlogei_top1"
                    if zero_llm_mode or not _llm_acquisition_strategy_enabled(settings)
                    else "autobo_ensemble_sur_top1"
                    if ensemble_sur_mode
                    else "autobo_top1"
                ),
                "autobo_qlogei_rank": qlogei_rank,
                "autobo_shortlist_rank": 1,
                "selected_rank": 1,
                "top1_candidate": dict(shortlist[0].get("candidate", {})),
                "af_sources": af_sources,
                "af_ranks": af_ranks,
                "af_consensus_count": int(selected_record.get("af_consensus_count", len(af_sources)) or len(af_sources)),
                "proposed_by": list(selected_record.get("proposed_by", [])) if isinstance(selected_record.get("proposed_by"), list) else [],
                "coverage_targets": coverage_targets,
                "coverage_domain_size": selected_record.get("coverage_domain_size"),
            },
            "current_proposal": {
                "candidates": [candidate],
                "selected_index": 0,
            },
            "llm_usage": _empty_usage_delta(),
            "log_lines": [
                "[select_candidate] autobo qlogei top1 deterministic"
                if zero_llm_mode or not _llm_acquisition_strategy_enabled(settings)
                else "[select_candidate] autobo top1 fallback"
            ],
        }

    memory_manager = MemoryManager.from_dict(state.get("memory", {}))
    context = (
        _memoryless_autobo_acquisition_select_context(state)
        if not _memory_enabled(settings)
        else ContextBuilder.for_autobo_acquisition_select(state, memory_manager)
    )
    context_shortlist = list(context.get("shortlist", [])) or shortlist
    early_exploration_info = _early_post_warm_start_prompt_info(
        settings=settings,
        observations=list(state.get("observations", [])),
        warm_start_target=int(state.get("warm_start_target", 0) or 0),
    )
    if not _memory_enabled(settings):
        early_exploration_info = {"enabled": False, "bo_round_index": 0, "window": 0}
    prompt_limit = int(getattr(settings, "autobo_acq_top_k", 5) or 5)
    if bool(early_exploration_info.get("enabled")) and any(
        isinstance(item.get("coverage_targets"), list) and item.get("coverage_targets")
        for item in context_shortlist
    ):
        prompt_limit = max(prompt_limit, len(context_shortlist))
    prompt_shortlist = context_shortlist[:prompt_limit]
    stagnation_info = {
        "is_stagnant": False if not _memory_enabled(settings) else bool((state.get("convergence_state", {}) or {}).get("is_stagnant")),
        "stagnation_length": 0 if not _memory_enabled(settings) else int((state.get("convergence_state", {}) or {}).get("stagnation_length", 0) or 0),
        "last_improvement_iteration": None if not _memory_enabled(settings) else (state.get("convergence_state", {}) or {}).get("last_improvement_iteration"),
        "best_result": None if not _memory_enabled(settings) else state.get("best_result"),
    }
    if ensemble_sur_mode:
        prompt = build_ensemble_sur_selection_prompt(
            reaction_context=context.get("reaction_context", {}),
            top_observations=context.get("top_observations", []),
            bottom_observations=context.get("bottom_observations", []),
            candidates=[
                {
                    "id": index + 1,
                    "candidate": item.get("candidate", {}),
                    "proposed_by": item.get("proposed_by", []),
                    "surrogate_consensus_count": item.get("surrogate_consensus_count"),
                    "surrogate_cross_scores": item.get("surrogate_cross_scores"),
                }
                for index, item in enumerate(prompt_shortlist)
            ],
            total_observations=int(context.get("total_observations", 0)),
            surrogate_composite_summary=runtime_metadata.get("surrogate_composite_summary", []),
            composite_explanation=str(runtime_metadata.get("surrogate_composite_explanation") or ""),
            knowledge_cards_text=context.get("knowledge_cards_text", ""),
            memory_rules=context.get("memory_rules", []),
            active_hypotheses=context.get("active_hypotheses", []),
            stagnation_info=stagnation_info,
        )
        default = {
            "selected_id": 1,
            "reasoning": "Default to the current ensemble-sur reference candidate.",
            "comparison_to_top1": "Candidate #1 is accepted as the current ensemble-sur reference choice.",
            "selection_mode": "ensemble_sur_choice",
            "model_confidence_assessment": "",
            "exploration_rationale": "",
            "knowledge_memory_check": "",
            "confidence": 0.7,
        }
    else:
        prompt = build_acquisition_selection_prompt(
            reaction_context=context.get("reaction_context", {}),
            top_observations=context.get("top_observations", []),
            bottom_observations=context.get("bottom_observations", []),
            candidates=[
                {
                    "id": index + 1,
                    "candidate": item.get("candidate", {}),
                    "predicted_value": item.get("predicted_value"),
                    "uncertainty": item.get("uncertainty"),
                    "acquisition_value": item.get("acquisition_value"),
                    "acquisition_value_raw": item.get("acquisition_value_raw"),
                    "selection_step": item.get("selection_step"),
                    "selection_mode": item.get("selection_mode"),
                    "af_sources": item.get("af_sources"),
                    "af_ranks": item.get("af_ranks"),
                    "af_consensus_count": item.get("af_consensus_count"),
                    "ensemble_reference_score": item.get("ensemble_reference_score"),
                    "ensemble_weighted_rank_score": item.get("ensemble_weighted_rank_score"),
                    "ensemble_diversity_bonus": item.get("ensemble_diversity_bonus"),
                    "sigma_rank": item.get("sigma_rank"),
                    "value_attempt_counts": item.get("value_attempt_counts"),
                    "changed_vs_best": item.get("changed_vs_best"),
                    "coverage_targets": item.get("coverage_targets"),
                    "unseen_categorical_values": item.get("unseen_categorical_values"),
                    "coverage_domain_size": item.get("coverage_domain_size"),
                }
                for index, item in enumerate(prompt_shortlist)
            ],
            total_observations=int(context.get("total_observations", 0)),
            knowledge_cards_text=context.get("knowledge_cards_text", ""),
            memory_rules=context.get("memory_rules", []),
            active_hypotheses=context.get("active_hypotheses", []),
            recent_override_outcomes=context.get("recent_override_outcomes", []),
            stagnation_info=stagnation_info,
            ensemble_mode=ensemble_mode,
            early_exploration_info=early_exploration_info,
        )
        default = {
            "selected_id": 1,
            "reasoning": "Default to the current shortlist reference candidate.",
            "comparison_to_top1": (
                "Candidate #1 is accepted as the best current ensemble reference choice."
                if ensemble_mode
                else "Candidate #1 is accepted as the best current choice."
            ),
            "selection_mode": "top1_follow",
        }
    parsed, messages, llm_usage = invoke_json_node(
        llm,
        state,
        prompt,
        default,
        node_name="select_candidate",
        lightweight=not _memory_enabled(settings),
    )
    outbound_messages = list(messages)
    raw_selected_id = parsed.get("selected_id")
    selected_id = _coerce_selected_id(raw_selected_id, default=1)
    parsed_selected_id = selected_id
    max_allowed_id = len(prompt_shortlist)
    selection_audit: dict[str, Any] = {
        "raw_selected_id": raw_selected_id,
        "parsed_selected_id": parsed_selected_id,
        "max_allowed_id": max_allowed_id,
        "selection_fallback_reason": "",
        "dataset_fallback_applied": False,
    }
    if not 1 <= selected_id <= max_allowed_id:
        if raw_selected_id not in (None, "", 1, "1"):
            outbound_messages.append(
                AIMessage(
                    content=(
                        f"AutoBO LLM selection {raw_selected_id!r} parsed as #{selected_id} outside the allowed "
                        f"top-{max_allowed_id} range; defaulted to #1."
                    )
                )
            )
            selection_audit["selection_fallback_reason"] = "selected_id_out_of_prompt_range"
        selected_id = 1
    if ensemble_sur_mode:
        override_evidence = {
            "evidence_type": "none",
            "evidence_ids": [],
            "trajectory_references": [],
            "chemistry_argument": "",
            "validated": False,
        }
        evidence_error = ""
    else:
        override_evidence, evidence_error = _validate_override_evidence(
            parsed.get("override_evidence"),
            state=state,
            memory_manager=memory_manager,
            reasoning=str(parsed.get("reasoning") or ""),
        )
    if selected_id == 1 and evidence_error == "missing override_evidence":
        evidence_error = ""
    if evidence_error:
        outbound_messages.append(
            AIMessage(content=f"AutoBO LLM override evidence warning: {evidence_error}; preserving selected candidate.")
        )
    chosen_prompt_index = selected_id - 1
    chosen_index = chosen_prompt_index
    chosen_index = min(max(chosen_index, 0), len(shortlist) - 1)
    selected_record = shortlist[chosen_index]
    candidate = selected_record.get("candidate", {})
    raw_comparison_to_top1 = str(parsed.get("comparison_to_top1") or "")
    comparison_to_top1 = raw_comparison_to_top1 or default["comparison_to_top1"]
    selection_mode = str(parsed.get("selection_mode") or default["selection_mode"])
    if selected_id != 1 and len(raw_comparison_to_top1.strip()) < 20:
        comparison_to_top1 = (
            f"The LLM selected shortlist candidate #{selected_id} instead of the ensemble reference. "
            if ensemble_mode
            else f"The LLM overrode shortlist top-1 and chose candidate #{selected_id}. "
        ) + (
            "Provide a more explicit comparison in future runs."
        )
    if ensemble_sur_mode:
        selection_mode = "ensemble_sur_choice"
    elif selected_id != 1 and selection_mode == "top1_follow":
        selection_mode = "ensemble_non_reference_choice" if ensemble_mode else "non_top1_override"

    oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
    if oracle is not None:
        if oracle.candidate_exists(candidate):
            candidate = oracle.lookup(candidate)["candidate"]
            selected_record = dict(selected_record)
            selected_record["candidate"] = candidate
        else:
            fallback_selection = _first_dataset_backed_shortlist_record(shortlist, oracle, preferred_index=chosen_index)
            if fallback_selection is not None:
                original_chosen_index = chosen_index
                chosen_index, selected_record = fallback_selection
                candidate = selected_record.get("candidate", {})
                selection_audit["dataset_fallback_applied"] = True
                selection_audit["dataset_fallback_from_index"] = original_chosen_index
                selection_audit["dataset_fallback_to_index"] = chosen_index
                selection_audit["selection_fallback_reason"] = (
                    selection_audit["selection_fallback_reason"] or "selected_candidate_not_dataset_backed"
                )
                outbound_messages.append(
                    AIMessage(
                        content=(
                            f"Replaced invalid AutoBO selection rank {selected_id} with dataset-backed shortlist index {chosen_index}."
                        )
                )
                )

    af_sources = list(selected_record.get("af_sources", [])) if isinstance(selected_record.get("af_sources"), list) else []
    af_ranks = dict(selected_record.get("af_ranks", {})) if isinstance(selected_record.get("af_ranks"), dict) else {}
    coverage_targets = (
        list(selected_record.get("coverage_targets", []))
        if isinstance(selected_record.get("coverage_targets"), list)
        else []
    )
    final_selected_rank = _coerce_int(selected_record.get("autobo_rank"), default=chosen_index + 1)
    intended_rank = selected_id
    selected_qlogei_rank = None
    if ensemble_sur_mode:
        selected_qlogei_rank = None
    elif ensemble_mode:
        qlogei_rank_value = af_ranks.get("qlogei")
        if qlogei_rank_value is not None:
            selected_qlogei_rank = _coerce_int(qlogei_rank_value, default=0) or None
    else:
        selected_qlogei_rank = final_selected_rank
    evidence_validation_status = (
        "validated" if override_evidence.get("validated") else "warning" if evidence_error else "not_required"
    )

    proposal_selected = {
        "selected_index": chosen_index,
        "override": bool(final_selected_rank != 1),
        "candidate": candidate,
        "rationale": {
            "chemical_reasoning": str(parsed.get("reasoning") or default["reasoning"]),
            "comparison_to_top1": comparison_to_top1,
            "selection_mode": selection_mode,
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "override_evidence": override_evidence,
            "raw_selected_id": raw_selected_id,
            "parsed_selected_id": parsed_selected_id,
            "intended_selected_rank": intended_rank,
            "actual_selected_rank": final_selected_rank,
            "selection_fallback_reason": selection_audit.get("selection_fallback_reason", ""),
            "dataset_fallback_applied": bool(selection_audit.get("dataset_fallback_applied")),
            "evidence_validation_status": evidence_validation_status,
            "evidence_warning": evidence_error,
            "model_confidence_assessment": str(parsed.get("model_confidence_assessment") or ""),
            "exploration_rationale": str(parsed.get("exploration_rationale") or ""),
            "knowledge_memory_check": str(parsed.get("knowledge_memory_check") or ""),
        },
        "confidence": _coerce_float(parsed.get("confidence"), default=0.8) if ensemble_sur_mode else 0.8,
        "selection_source": "autobo_ensemble_sur_llm" if ensemble_sur_mode else "autobo_llm_acquisition",
        "autobo_qlogei_rank": selected_qlogei_rank,
        "autobo_shortlist_rank": final_selected_rank,
        "selected_rank": final_selected_rank,
        "intended_selected_rank": intended_rank,
        "actual_selected_rank": final_selected_rank,
        "raw_selected_id": raw_selected_id,
        "parsed_selected_id": parsed_selected_id,
        "selection_fallback_reason": selection_audit.get("selection_fallback_reason", ""),
        "dataset_fallback_applied": bool(selection_audit.get("dataset_fallback_applied")),
        "evidence_validation_status": evidence_validation_status,
        "evidence_warning": evidence_error,
        "top1_candidate": dict(shortlist[0].get("candidate", {})),
        "af_sources": af_sources,
        "af_ranks": af_ranks,
        "af_consensus_count": int(selected_record.get("af_consensus_count", len(af_sources)) or len(af_sources)),
        "proposed_by": list(selected_record.get("proposed_by", [])) if isinstance(selected_record.get("proposed_by"), list) else [],
        "coverage_targets": coverage_targets,
        "coverage_domain_size": selected_record.get("coverage_domain_size"),
    }
    updated_shortlist = list(shortlist)
    updated_shortlist[chosen_index] = dict(selected_record)
    return {
        "messages": outbound_messages,
        "proposal_shortlist": updated_shortlist,
        "proposal_selected": proposal_selected,
        "current_proposal": {
            "candidates": [candidate],
            "selected_index": chosen_index,
        },
        "llm_usage": llm_usage,
        "log_lines": [f"[select_candidate] autobo rank={final_selected_rank} shortlist_index={chosen_index}"],
    }


def select_pure_reasoning_candidate(
    *,
    state: dict[str, Any],
    settings,
    llm,
    invoke_json_node,
) -> dict[str, Any]:
    structured_spec = _build_pure_reasoning_space_spec(state)
    if structured_spec is None:
        return _select_pure_reasoning_from_candidate_pool(
            state=state,
            settings=settings,
            llm=llm,
            invoke_json_node=invoke_json_node,
        )
    return _select_pure_reasoning_from_structured_space(
        state=state,
        llm=llm,
        invoke_json_node=invoke_json_node,
        structured_spec=structured_spec,
    )


def _select_pure_reasoning_from_candidate_pool(
    *,
    state: dict[str, Any],
    settings,
    llm,
    invoke_json_node,
) -> dict[str, Any]:
    display_limit = 32
    iteration_seed = _state_seed(state)
    observations = list(state.get("observations", []))
    variables = state.get("problem_spec", {}).get("variables", [])
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in observations
        if item.get("candidate")
    }
    dataset_spec = state.get("problem_spec", {}).get("dataset", {})
    dataset_candidate_pool = dataset_candidate_pool_from_spec(dataset_spec)
    candidate_pool = build_bo_candidate_pool(
        variables,
        observed_keys=observed_keys,
        candidate_pool_size=max(256, display_limit * 8),
        seed=iteration_seed,
        hard_constraints=[],
        candidate_pool=dataset_candidate_pool,
    )
    if not candidate_pool:
        candidate_pool = build_diverse_fallback_candidates(
            variables,
            n_total=display_limit,
            seed=iteration_seed,
            hard_constraints=[],
            observed_keys=observed_keys,
            candidate_pool=dataset_candidate_pool,
        )
    elif len(candidate_pool) > display_limit:
        candidate_pool = build_diverse_fallback_candidates(
            variables,
            n_total=display_limit,
            seed=iteration_seed,
            hard_constraints=[],
            observed_keys=set(),
            candidate_pool=candidate_pool,
        )

    if candidate_pool:
        rng = np.random.default_rng(iteration_seed)
        shuffled_indices = list(rng.permutation(len(candidate_pool)))
        prompt_candidates = [dict(candidate_pool[index]) for index in shuffled_indices[:display_limit]]
    else:
        prompt_candidates = []

    shortlist = build_bo_shortlist_from_candidates(prompt_candidates, [])
    for index, item in enumerate(shortlist):
        item["selection_step"] = index + 1
        item["selection_mode"] = "llm_reasoning_pool"

    resolved_components = _pure_reasoning_resolved_components()
    payload = {
        "status": "success" if shortlist else "empty_pool",
        "strategy": "pure_reasoning_ablation",
        "shortlist": shortlist,
        "recommended_index": None,
        "candidates": [item.get("candidate", {}) for item in shortlist],
        "resolved_components": resolved_components,
        "metadata": {
            "proposal_strategy": "pure_reasoning_ablation",
            "candidate_pool_source": "dataset" if dataset_candidate_pool is not None else "search_space",
            "candidate_pool_size": len(candidate_pool),
            "prompt_candidate_count": len(shortlist),
            "representation_mode": "candidate_pool_fallback",
        },
    }
    if not shortlist:
        return {
            "messages": [AIMessage(content="Pure reasoning candidate pool is empty; no candidate could be selected.")],
            "proposal_shortlist": shortlist,
            "proposal_selected": {
                "selected_index": 0,
                "override": False,
                "candidate": {},
                "rationale": {
                    "chemical_reasoning": "No legal candidates remained in the pure reasoning pool.",
                    "comparison_to_top1": "",
                    "selection_mode": "llm_direct_select",
                    "hypothesis_alignment": "",
                    "information_value": "",
                    "concerns": "",
                },
                "confidence": 0.0,
                "selection_source": "pure_reasoning_empty_pool",
            },
            "current_proposal": {"candidates": [{}], "selected_index": 0},
            "payload": payload,
            "effective_config": _pure_reasoning_effective_config(state),
            "llm_usage": _empty_usage_delta(),
            "log_lines": ["[select_candidate] pure_reasoning empty_pool"],
        }

    memory_manager = MemoryManager.from_dict(state.get("memory", {}))
    context = ContextBuilder.for_autobo_acquisition_select(state, memory_manager)
    prompt = build_pure_reasoning_selection_prompt(
        reaction_context=context.get("reaction_context", {}),
        top_observations=context.get("top_observations", []),
        bottom_observations=context.get("bottom_observations", []),
        candidates=[
            {
                "id": index + 1,
                "candidate": item.get("candidate", {}),
            }
            for index, item in enumerate(shortlist)
        ],
        total_observations=int(context.get("total_observations", 0)),
        knowledge_cards_text=context.get("knowledge_cards_text", ""),
        memory_rules=context.get("memory_rules", []),
        active_hypotheses=context.get("active_hypotheses", []),
        stagnation_info={
            "is_stagnant": bool((state.get("convergence_state", {}) or {}).get("is_stagnant")),
            "stagnation_length": int((state.get("convergence_state", {}) or {}).get("stagnation_length", 0) or 0),
            "last_improvement_iteration": (state.get("convergence_state", {}) or {}).get("last_improvement_iteration"),
            "best_result": state.get("best_result"),
        },
    )
    default = {
        "selected_id": 1,
        "reasoning": "Select the first legal candidate in the pure reasoning pool.",
        "hypothesis_alignment": "",
        "information_value": "",
        "concerns": "",
        "confidence": 0.6,
    }
    parsed, messages, llm_usage = invoke_json_node(
        llm,
        state,
        prompt,
        default,
        node_name="select_candidate",
    )
    selected_id = _coerce_int(parsed.get("selected_id"), default=1)
    chosen_index = min(max(selected_id - 1, 0), len(shortlist) - 1)
    selected_record = shortlist[chosen_index]
    candidate = selected_record.get("candidate", {})
    proposal_selected = {
        "selected_index": chosen_index,
        "override": False,
        "candidate": candidate,
        "rationale": {
            "chemical_reasoning": str(parsed.get("reasoning") or default["reasoning"]),
            "comparison_to_top1": "",
            "selection_mode": "llm_direct_select",
            "hypothesis_alignment": str(parsed.get("hypothesis_alignment") or ""),
            "information_value": str(parsed.get("information_value") or ""),
            "concerns": str(parsed.get("concerns") or ""),
        },
        "confidence": _coerce_float(parsed.get("confidence"), default=0.6),
        "selection_source": "pure_reasoning_llm",
        "selected_rank": chosen_index + 1,
    }
    return {
        "messages": messages,
        "proposal_shortlist": shortlist,
        "proposal_selected": proposal_selected,
        "current_proposal": {
            "candidates": [candidate],
            "selected_index": chosen_index,
        },
        "payload": payload,
        "effective_config": _pure_reasoning_effective_config(state),
        "llm_usage": llm_usage,
        "log_lines": [f"[select_candidate] pure_reasoning selected={chosen_index + 1} pool={len(shortlist)}"],
    }


def _select_pure_reasoning_from_structured_space(
    *,
    state: dict[str, Any],
    llm,
    invoke_json_node,
    structured_spec: dict[str, Any],
) -> dict[str, Any]:
    memory_manager = MemoryManager.from_dict(state.get("memory", {}))
    context = ContextBuilder.for_autobo_acquisition_select(state, memory_manager)
    all_messages: list[Any] = []
    total_usage = _empty_usage_delta()
    validation_feedback = ""
    resolved_candidate: dict[str, Any] | None = None
    parsed_response: dict[str, Any] = dict(structured_spec.get("default_response", {}))
    failure_reason = ""

    for attempt in range(2):
        prompt = build_pure_reasoning_space_selection_prompt(
            reaction_context=context.get("reaction_context", {}),
            top_observations=context.get("top_observations", []),
            bottom_observations=context.get("bottom_observations", []),
            total_observations=int(context.get("total_observations", 0)),
            space_description=str(structured_spec.get("space_description") or ""),
            output_schema=str(structured_spec.get("output_schema") or "{}"),
            knowledge_cards_text=context.get("knowledge_cards_text", ""),
            memory_rules=context.get("memory_rules", []),
            active_hypotheses=context.get("active_hypotheses", []),
            stagnation_info={
                "is_stagnant": bool((state.get("convergence_state", {}) or {}).get("is_stagnant")),
                "stagnation_length": int((state.get("convergence_state", {}) or {}).get("stagnation_length", 0) or 0),
                "last_improvement_iteration": (state.get("convergence_state", {}) or {}).get("last_improvement_iteration"),
                "best_result": state.get("best_result"),
            },
            validation_feedback=validation_feedback,
        )
        parsed, messages, llm_usage = invoke_json_node(
            llm,
            state,
            prompt,
            dict(structured_spec.get("default_response", {})),
            node_name="select_candidate",
        )
        all_messages.extend(messages)
        total_usage = _accumulate_usage_delta(total_usage, llm_usage)
        parsed_response = dict(parsed or {})
        candidate, failure_reason = _resolve_structured_pure_reasoning_candidate(
            parsed_response,
            structured_spec=structured_spec,
            state=state,
        )
        if candidate is not None:
            resolved_candidate = candidate
            break
        validation_feedback = failure_reason

    if resolved_candidate is None:
        fallback_candidate = _first_valid_unseen_candidate_from_structured_space(structured_spec, state)
        if fallback_candidate is None:
            return {
                "messages": [AIMessage(content="Pure reasoning could not produce a valid structured recommendation.")],
                "proposal_shortlist": [],
                "proposal_selected": {
                    "selected_index": 0,
                    "override": False,
                    "candidate": {},
                    "rationale": {
                        "chemical_reasoning": failure_reason or "No valid structured recommendation was produced.",
                        "comparison_to_top1": "",
                        "selection_mode": "llm_direct_select",
                        "hypothesis_alignment": "",
                        "information_value": "",
                        "concerns": failure_reason or "",
                    },
                    "confidence": 0.0,
                    "selection_source": "pure_reasoning_empty_pool",
                },
                "current_proposal": {"candidates": [{}], "selected_index": 0},
                "payload": {
                    "status": "invalid_selection",
                    "strategy": "pure_reasoning_ablation",
                    "resolved_components": _pure_reasoning_resolved_components(),
                    "metadata": {
                        "proposal_strategy": "pure_reasoning_ablation",
                        "representation_mode": structured_spec.get("mode"),
                        "selection_error": failure_reason,
                    },
                },
                "effective_config": _pure_reasoning_effective_config(state),
                "llm_usage": total_usage,
                "log_lines": [f"[select_candidate] pure_reasoning invalid mode={structured_spec.get('mode')}"],
            }
        resolved_candidate = fallback_candidate

    shortlist = _pure_reasoning_selected_shortlist(resolved_candidate)
    payload = {
        "status": "success",
        "strategy": "pure_reasoning_ablation",
        "shortlist": shortlist,
        "recommended_index": 0,
        "candidates": [resolved_candidate],
        "resolved_components": _pure_reasoning_resolved_components(),
        "metadata": {
            "proposal_strategy": "pure_reasoning_ablation",
            "representation_mode": structured_spec.get("mode"),
            **dict(structured_spec.get("metadata", {})),
        },
    }
    proposal_selected = {
        "selected_index": 0,
        "override": False,
        "candidate": resolved_candidate,
        "rationale": {
            "chemical_reasoning": str(parsed_response.get("reasoning") or "Selected directly from the structured search space."),
            "comparison_to_top1": "",
            "selection_mode": "llm_direct_select",
            "hypothesis_alignment": str(parsed_response.get("hypothesis_alignment") or ""),
            "information_value": str(parsed_response.get("information_value") or ""),
            "concerns": str(parsed_response.get("concerns") or ""),
        },
        "confidence": _coerce_float(parsed_response.get("confidence"), default=0.6),
        "selection_source": "pure_reasoning_llm",
        "selected_rank": 1,
    }
    return {
        "messages": all_messages,
        "proposal_shortlist": shortlist,
        "proposal_selected": proposal_selected,
        "current_proposal": {"candidates": [resolved_candidate], "selected_index": 0},
        "payload": payload,
        "effective_config": _pure_reasoning_effective_config(state),
        "llm_usage": total_usage,
        "log_lines": [f"[select_candidate] pure_reasoning structured mode={structured_spec.get('mode')}"],
    }


def _build_pure_reasoning_space_spec(state: dict[str, Any]) -> dict[str, Any] | None:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    reaction_type = str(problem_spec.get("reaction_type") or "").strip().upper()
    if reaction_type == "OCM":
        ocm_spec = _build_ocm_encoded_spec(state)
        if ocm_spec is not None:
            return ocm_spec
    if reaction_type == "SUZUKI":
        suzuki_spec = _build_suzuki_encoded_spec(state)
        if suzuki_spec is not None:
            return suzuki_spec
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    if oracle is not None:
        if reaction_type == "OER":
            oer_spec = _build_oer_discrete_simplex_spec(state, oracle)
            if oer_spec is not None:
                return oer_spec
        if _should_expose_declared_dataset_space(problem_spec):
            return _build_declared_dataset_variable_space_spec(state, oracle)
        cartesian_spec = _build_cartesian_dataset_spec(state, oracle)
        if cartesian_spec is not None:
            return cartesian_spec
        return None
    return _build_generic_variable_space_spec(state)


def _should_expose_declared_dataset_space(problem_spec: dict[str, Any]) -> bool:
    warm_start_spec = problem_spec.get("warm_start_spec")
    if not isinstance(warm_start_spec, dict):
        return False
    return bool(warm_start_spec.get("expose_declared_variable_space"))


def _build_declared_dataset_variable_space_spec(
    state: dict[str, Any],
    oracle: DatasetOracle,
) -> dict[str, Any]:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    warm_start_spec = (
        problem_spec.get("warm_start_spec")
        if isinstance(problem_spec.get("warm_start_spec"), dict)
        else {}
    )
    spec = _build_generic_variable_space_spec(state)
    representation_mode = str(
        warm_start_spec.get("representation_mode") or "declared_dataset_variable_space"
    ).strip()
    spec["mode"] = representation_mode or "declared_dataset_variable_space"
    spec["dataset_backed"] = bool(warm_start_spec.get("validate_against_dataset", True))
    spec["feature_columns"] = list(oracle.feature_columns)
    spec["space_description"] = "\n".join(
        [
            str(spec.get("space_description") or ""),
            "- Dataset validation:",
            "  - The LLM may reason over the declared variable ranges and constraints above.",
            "  - A proposed point is accepted only if it matches an unseen row in the dataset oracle.",
            "  - If validation rejects a point, propose another unseen composition.",
        ]
    )
    if str(warm_start_spec.get("invalid_candidate_instruction") or "").strip():
        spec["space_description"] = (
            f"{spec['space_description']}\n"
            f"  - {str(warm_start_spec.get('invalid_candidate_instruction')).strip()}"
        )
    metadata = dict(spec.get("metadata", {}))
    metadata.update(
        {
            "representation_mode": spec["mode"],
            "dataset_backed": spec["dataset_backed"],
            "legal_unseen_count": len(oracle.candidates) - _observed_candidate_count(state),
        }
    )
    spec["metadata"] = metadata
    return spec


def _build_oer_discrete_simplex_spec(
    state: dict[str, Any],
    oracle: DatasetOracle,
) -> dict[str, Any] | None:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    variables = list(problem_spec.get("variables", []) or [])
    variables_by_name = {
        str(variable.get("name") or ""): dict(variable)
        for variable in variables
        if isinstance(variable, dict) and str(variable.get("name") or "").strip()
    }
    feature_columns = [str(column) for column in oracle.feature_columns]
    if not feature_columns or any(column not in variables_by_name for column in feature_columns):
        return None

    allowed_values: dict[str, list[str]] = {}
    for column in feature_columns:
        variable_values = _continuous_allowed_values(variables_by_name[column])
        dataset_values = _sorted_choice_values({candidate.get(column, "") for candidate in oracle.candidates})
        values = variable_values or dataset_values
        if not values:
            return None
        allowed_values[column] = values

    warm_start_spec = (
        problem_spec.get("warm_start_spec")
        if isinstance(problem_spec.get("warm_start_spec"), dict)
        else {}
    )
    requested_mode = str(warm_start_spec.get("representation_mode") or "").strip()
    representation_mode = (
        requested_mode
        if requested_mode and requested_mode != "declared_continuous_simplex"
        else "declared_discrete_simplex"
    )
    variable_constraints = _structured_variable_constraints(problem_spec)
    if not variable_constraints:
        variable_constraints = [
            {
                "type": "sum_equals",
                "variables": feature_columns,
                "value": 1.0,
                "tolerance": 1e-6,
            }
        ]

    lines = [
        "This OER benchmark is a discrete catalyst-composition simplex.",
        "Choose exact grid levels only; do not propose arbitrary continuous fractions.",
    ]
    for column in feature_columns:
        lines.append(f"- {column}: exact allowed levels = [{', '.join(allowed_values[column])}]")
    lines.append("- Structured variable constraints:")
    lines.extend([f"  - {_describe_structured_variable_constraint(item)}" for item in variable_constraints])
    lines.extend(
        [
            "- Dataset validation:",
            "  - A proposed grid point is accepted only if it matches an unseen row in the OER dataset oracle.",
            "  - If validation rejects a point, choose another unseen grid composition.",
            f"Unseen legal experiments remaining: {len(oracle.candidates) - _observed_candidate_count(state)}",
        ]
    )
    if str(warm_start_spec.get("invalid_candidate_instruction") or "").strip():
        lines.append(f"  - {str(warm_start_spec.get('invalid_candidate_instruction')).strip()}")

    default_candidate = _first_unseen_oracle_candidate(oracle, state) or dict(oracle.candidates[0])
    return {
        "mode": representation_mode,
        "space_description": "\n".join(lines),
        "output_schema": _variable_map_output_schema({column: default_candidate.get(column, "0.0") for column in feature_columns}),
        "default_response": {
            "variables": {column: default_candidate.get(column, "0.0") for column in feature_columns},
            "reasoning": "Choose one legal unseen OER composition from the discrete simplex grid.",
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "confidence": 0.6,
        },
        "metadata": {
            "representation_mode": representation_mode,
            "dataset_backed": bool(warm_start_spec.get("validate_against_dataset", True)),
            "grid_value_count": {column: len(values) for column, values in allowed_values.items()},
            "legal_unseen_count": len(oracle.candidates) - _observed_candidate_count(state),
        },
        "feature_columns": feature_columns,
        "allowed_values": allowed_values,
        "variable_constraints": variable_constraints,
        "dataset_backed": bool(warm_start_spec.get("validate_against_dataset", True)),
    }


def _build_cartesian_dataset_spec(state: dict[str, Any], oracle: DatasetOracle) -> dict[str, Any] | None:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    warm_start_spec = (
        problem_spec.get("warm_start_spec")
        if isinstance(problem_spec.get("warm_start_spec"), dict)
        else {}
    )
    variables = list(problem_spec.get("variables", []) or [])
    variables_by_name = {
        str(variable.get("name") or ""): variable
        for variable in variables
        if isinstance(variable, dict) and str(variable.get("name") or "").strip()
    }
    feature_columns = [str(column) for column in oracle.feature_columns]
    if any(column not in variables_by_name for column in feature_columns):
        return None
    unique_values = {
        column: _sorted_choice_values({candidate.get(column, "") for candidate in oracle.candidates})
        for column in feature_columns
    }
    total = 1
    for column in feature_columns:
        total *= max(len(unique_values[column]), 1)
    if total != oracle.size:
        return None

    lines = [
        "This benchmark is an exact cartesian grid over the following per-variable choices.",
        "Any unseen combination formed from these exact levels is a legal experiment.",
    ]
    choice_maps: dict[str, dict[str, str]] = {}
    for column in feature_columns:
        variable = variables_by_name[column]
        values = unique_values[column]
        if variable.get("type") == "continuous":
            lines.append(f"- {column}: exact allowed levels = [{', '.join(values)}]")
            continue
        prefix = _choice_prefix(column)
        mapping = {f"{prefix}{index + 1}": value for index, value in enumerate(values)}
        choice_maps[column] = mapping
        lines.append(f"- {column}:")
        lines.extend([f"  {choice_id} = {value}" for choice_id, value in mapping.items()])
    constraints = [str(item).strip() for item in problem_spec.get("constraints", []) if str(item).strip()]
    if constraints:
        lines.append("- Constraints:")
        lines.extend([f"  - {item}" for item in constraints[:8]])
    lines.extend(
        [
            "- Dataset validation:",
            "  - A proposed cartesian grid point is accepted only if it matches an unseen row in the dataset oracle.",
            "  - If validation rejects a point, choose another unseen exact grid configuration.",
        ]
    )
    if str(warm_start_spec.get("invalid_candidate_instruction") or "").strip():
        lines.append(f"  - {str(warm_start_spec.get('invalid_candidate_instruction')).strip()}")
    lines.append(f"Unseen legal experiments remaining: {len(oracle.candidates) - _observed_candidate_count(state)}")

    output_schema = _variable_map_output_schema(
        {
            column: next(iter(choice_maps.get(column, {}).keys()), unique_values[column][0])
            for column in feature_columns
        }
    )
    return {
        "mode": "dataset_cartesian",
        "space_description": "\n".join(lines),
        "output_schema": output_schema,
        "default_response": {
            "variables": {column: next(iter(choice_maps.get(column, {}).keys()), unique_values[column][0]) for column in feature_columns},
            "reasoning": "Select one valid unseen combination from the exact dataset grid.",
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "confidence": 0.6,
        },
        "metadata": {
            "representation_mode": "dataset_cartesian",
            "feature_count": len(feature_columns),
            "dataset_backed": bool(warm_start_spec.get("validate_against_dataset", True)),
            "legal_unseen_count": len(oracle.candidates) - _observed_candidate_count(state),
        },
        "feature_columns": feature_columns,
        "choice_maps": choice_maps,
        "exact_values": unique_values,
        "dataset_backed": bool(warm_start_spec.get("validate_against_dataset", True)),
    }


def _build_ocm_encoded_spec(state: dict[str, Any]) -> dict[str, Any] | None:
    dataset_path = _dataset_path_from_problem_spec(state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {})
    if dataset_path is None:
        return None
    try:
        domain = load_ocm_domain_spec(dataset_path)
    except Exception:
        return None

    lines = [build_ocm_domain_prompt(dataset_path)]
    lines.append(f"Unseen legal experiments remaining: {len(domain.dataframe) - _observed_candidate_count(state)}")
    output_schema = """{
  "cat": "0",
  "Temp": 700,
  "CT": 0.38,
  "ar_level": "low",
  "ch4_o2_ratio": 0,
  "reasoning": "...",
  "hypothesis_alignment": "...",
  "information_value": "...",
  "concerns": "...",
  "confidence": 0.75
}"""
    default_ct = domain.ct_values[0]
    default_level = domain.ar_level_values_by_ct[default_ct][0]
    default_ratio = domain.ratio_slots_by_ct_level[(default_ct, default_level)][0]
    return {
        "mode": "ocm_encoded_domain",
        "space_description": "\n".join(lines),
        "output_schema": output_schema,
        "default_response": {
            "cat": "0",
            "Temp": _coerce_float(domain.temperature_values[0], default=0.0),
            "CT": _coerce_float(default_ct, default=0.0),
            "ar_level": default_level,
            "ch4_o2_ratio": default_ratio,
            "reasoning": "Choose one legal OCM catalyst/condition combination from the encoded domain.",
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "confidence": 0.6,
        },
        "metadata": {
            "representation_mode": "ocm_encoded_domain",
            "catalyst_count": len(domain.catalyst_list),
            "temperature_count": len(domain.temperature_values),
            "ct_count": len(domain.ct_values),
            "legal_unseen_count": len(domain.dataframe) - _observed_candidate_count(state),
        },
        "ocm_dataset_path": str(dataset_path),
        "dataset_backed": DatasetOracle.from_problem_spec(state.get("problem_spec", {})) is not None,
    }


def _build_suzuki_encoded_spec(state: dict[str, Any]) -> dict[str, Any] | None:
    dataset_path = _dataset_path_from_problem_spec(state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {})
    if dataset_path is None:
        return None
    try:
        domain = load_suzuki_domain_spec(dataset_path)
    except Exception:
        return None

    lines = [build_suzuki_domain_prompt(dataset_path)]
    lines.append(f"Unseen legal experiments remaining: {len(domain.dataframe) - _observed_candidate_count(state)}")
    output_schema = """{
  "pair_id": "1",
  "Ligand_Short_Hand": "AmPhos",
  "Reagent_1_Short_Hand": "LiOtBu",
  "Solvent_1_Short_Hand": "MeCN",
  "reasoning": "...",
  "hypothesis_alignment": "...",
  "information_value": "...",
  "concerns": "...",
  "confidence": 0.75
}"""
    default_pair_id = next(iter(domain.pair_index_to_pair))
    return {
        "mode": "suzuki_encoded_domain",
        "space_description": "\n".join(lines),
        "output_schema": output_schema,
        "default_response": {
            "pair_id": default_pair_id,
            "Ligand_Short_Hand": domain.ligand_values[0],
            "Reagent_1_Short_Hand": domain.reagent_values[0],
            "Solvent_1_Short_Hand": domain.solvent_values[0],
            "reasoning": "Choose one legal Suzuki substrate pair and exact ligand/base/solvent combination from the encoded domain.",
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "confidence": 0.6,
        },
        "metadata": {
            "representation_mode": "suzuki_encoded_domain",
            "pair_count": len(domain.pair_list),
            "ligand_count": len(domain.ligand_values),
            "reagent_count": len(domain.reagent_values),
            "solvent_count": len(domain.solvent_values),
            "legal_unseen_count": len(domain.dataframe) - _observed_candidate_count(state),
        },
        "suzuki_dataset_path": str(dataset_path),
        "dataset_backed": DatasetOracle.from_problem_spec(state.get("problem_spec", {})) is not None,
    }


def _build_ocm_factorized_dataset_spec(state: dict[str, Any], oracle: DatasetOracle) -> dict[str, Any] | None:
    rows = [dict(candidate) for candidate in oracle.candidates]
    required_columns = {"M1", "M2", "M3", "Support", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "CT"}
    if not required_columns.issubset(set(oracle.feature_columns)):
        return None

    temp_values = _sorted_choice_values({row["Temp"] for row in rows})
    flow_recipes = _sorted_tuple_records(
        {(row["Ar_flow"], row["CH4_flow"], row["O2_flow"], row["CT"]) for row in rows}
    )
    combo_tuples = _sorted_tuple_records(
        {(row["M1"], row["M2"], row["M3"], row["Support"]) for row in rows}
    )
    if len({(row["Temp"], row["Ar_flow"], row["CH4_flow"], row["O2_flow"], row["CT"]) for row in rows}) != len(temp_values) * len(flow_recipes):
        return None

    flow_set = set(flow_recipes)
    allowed_temps_by_combo: dict[str, list[str]] = {}
    combo_map: dict[str, tuple[str, str, str, str]] = {}
    for index, combo_tuple in enumerate(combo_tuples):
        combo_id = f"C{index + 1}"
        combo_map[combo_id] = combo_tuple
        temp_to_flows: dict[str, set[tuple[str, str, str, str]]] = {}
        for row in rows:
            if (row["M1"], row["M2"], row["M3"], row["Support"]) != combo_tuple:
                continue
            temp_to_flows.setdefault(row["Temp"], set()).add((row["Ar_flow"], row["CH4_flow"], row["O2_flow"], row["CT"]))
        allowed = [temp for temp in temp_values if temp_to_flows.get(temp) == flow_set]
        if not allowed:
            return None
        if any(flows and flows != flow_set for flows in temp_to_flows.values()):
            return None
        allowed_temps_by_combo[combo_id] = allowed

    flow_map = {f"F{index + 1}": recipe for index, recipe in enumerate(flow_recipes)}
    lines = [
        "This OCM benchmark factorizes into CatalystCombo x Temperature x FlowRecipe.",
        "Catalyst identity is constrained by experimentally observed tuples; FlowRecipe already includes CT.",
        "- CatalystCombo options:",
    ]
    lines.extend([f"  {combo_id} = {'|'.join(values)}" for combo_id, values in combo_map.items()])
    lines.append(f"- Temperature options: [{', '.join(temp_values)}]")
    lines.append("- FlowRecipe options:")
    lines.extend(
        [
            f"  {flow_id} = Ar_flow={recipe[0]}, CH4_flow={recipe[1]}, O2_flow={recipe[2]}, CT={recipe[3]}"
            for flow_id, recipe in flow_map.items()
        ]
    )
    restricted = [
        f"  {combo_id} only allows temperatures [{', '.join(allowed)}]"
        for combo_id, allowed in allowed_temps_by_combo.items()
        if len(allowed) != len(temp_values)
    ]
    if restricted:
        lines.append("- Temperature restrictions:")
        lines.extend(restricted)
    lines.append(f"Unseen legal experiments remaining: {len(rows) - _observed_candidate_count(state)}")

    output_schema = """{
  "catalyst_combo_id": "C1",
  "temperature": "850",
  "flow_recipe_id": "F1",
  "reasoning": "...",
  "hypothesis_alignment": "...",
  "information_value": "...",
  "concerns": "...",
  "confidence": 0.75
}"""
    default_combo_id = next(iter(combo_map))
    default_temp = allowed_temps_by_combo[default_combo_id][0]
    default_flow_id = next(iter(flow_map))
    return {
        "mode": "ocm_factorized_dataset",
        "space_description": "\n".join(lines),
        "output_schema": output_schema,
        "default_response": {
            "catalyst_combo_id": default_combo_id,
            "temperature": default_temp,
            "flow_recipe_id": default_flow_id,
            "reasoning": "Choose one legal OCM catalyst/process combination from the factorized search space.",
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "confidence": 0.6,
        },
        "metadata": {
            "representation_mode": "ocm_factorized_dataset",
            "catalyst_combo_count": len(combo_map),
            "temperature_count": len(temp_values),
            "flow_recipe_count": len(flow_map),
            "legal_unseen_count": len(rows) - _observed_candidate_count(state),
        },
        "combo_map": combo_map,
        "allowed_temps_by_combo": allowed_temps_by_combo,
        "temp_values": temp_values,
        "flow_map": flow_map,
    }


def _build_generic_variable_space_spec(state: dict[str, Any]) -> dict[str, Any]:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    variables = [dict(variable) for variable in (problem_spec.get("variables", []) or []) if isinstance(variable, dict)]
    lines = ["Choose the next experiment by assigning values directly to the variables below."]
    choice_maps: dict[str, dict[str, str]] = {}
    for variable in variables:
        name = str(variable.get("name") or "")
        if not name:
            continue
        if variable.get("type") == "continuous":
            allowed = _continuous_allowed_values(variable)
            if allowed:
                lines.append(f"- {name}: exact allowed levels = [{', '.join(allowed)}]")
            else:
                low, high = _continuous_domain_bounds(variable)
                lines.append(f"- {name}: continuous in [{low}, {high}]")
            continue
        labels = _variable_domain_labels(variable)
        prefix = _choice_prefix(name)
        mapping = {f"{prefix}{index + 1}": label for index, label in enumerate(labels)}
        choice_maps[name] = mapping
        lines.append(f"- {name}:")
        lines.extend([f"  {choice_id} = {label}" for choice_id, label in mapping.items()])
    constraints = [str(item).strip() for item in problem_spec.get("constraints", []) if str(item).strip()]
    if constraints:
        lines.append("- Constraints:")
        lines.extend([f"  - {item}" for item in constraints[:8]])
    variable_constraints = _structured_variable_constraints(problem_spec)
    if variable_constraints:
        lines.append("- Structured variable constraints:")
        lines.extend([f"  - {_describe_structured_variable_constraint(item)}" for item in variable_constraints])

    output_variables = {}
    for variable in variables:
        name = str(variable.get("name") or "")
        if not name:
            continue
        if variable.get("type") == "continuous":
            allowed = _continuous_allowed_values(variable)
            if allowed:
                output_variables[name] = allowed[0]
            else:
                low, high = _continuous_domain_bounds(variable)
                output_variables[name] = str((low + high) / 2.0)
        else:
            output_variables[name] = next(iter(choice_maps.get(name, {}).keys()), "")
    output_variables = _apply_default_variable_constraints(output_variables, variables, variable_constraints)

    return {
        "mode": "generic_variable_space",
        "space_description": "\n".join(lines),
        "output_schema": _variable_map_output_schema({"variable_name": "choice id or numeric value"}),
        "default_response": {
            "variables": output_variables,
            "reasoning": "Choose one legal assignment directly from the declared variable domains.",
            "hypothesis_alignment": "",
            "information_value": "",
            "concerns": "",
            "confidence": 0.6,
        },
        "metadata": {
            "representation_mode": "generic_variable_space",
            "variable_count": len(variables),
        },
        "variables": variables,
        "choice_maps": choice_maps,
        "variable_constraints": variable_constraints,
    }


def _structured_variable_constraints(problem_spec: dict[str, Any]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for item in problem_spec.get("variable_constraints", []) or []:
        if not isinstance(item, dict):
            continue
        constraint_type = str(item.get("type") or "").strip().lower()
        if constraint_type != "sum_equals":
            continue
        variables = [str(name).strip() for name in item.get("variables", []) if str(name).strip()]
        if not variables:
            continue
        constraints.append(
            {
                "type": "sum_equals",
                "variables": variables,
                "value": _coerce_float(item.get("value"), default=1.0),
                "tolerance": abs(_coerce_float(item.get("tolerance"), default=1e-6)),
            }
        )
    return constraints


def _describe_structured_variable_constraint(constraint: dict[str, Any]) -> str:
    if constraint.get("type") == "sum_equals":
        variables = [str(name) for name in constraint.get("variables", [])]
        value = _coerce_float(constraint.get("value"), default=1.0)
        tolerance = _coerce_float(constraint.get("tolerance"), default=1e-6)
        return f"{' + '.join(variables)} must equal {value} within tolerance {tolerance}."
    return str(constraint)


def _apply_default_variable_constraints(
    output_variables: dict[str, Any],
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    adjusted = dict(output_variables)
    variables_by_name = {str(variable.get("name") or ""): variable for variable in variables}
    for constraint in constraints:
        if constraint.get("type") != "sum_equals":
            continue
        names = [str(name) for name in constraint.get("variables", [])]
        if not names or any(name not in variables_by_name for name in names):
            continue
        if any(variables_by_name[name].get("type") != "continuous" for name in names):
            continue
        target = _coerce_float(constraint.get("value"), default=1.0)
        share = target / len(names)
        if all(
            _continuous_domain_bounds(variables_by_name[name])[0]
            <= share
            <= _continuous_domain_bounds(variables_by_name[name])[1]
            for name in names
        ):
            for name in names:
                low, high = _continuous_domain_bounds(variables_by_name[name])
                adjusted[name] = str(_format_continuous_choice(share, low=low, high=high))
    return adjusted


def _resolve_structured_pure_reasoning_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    mode = str(structured_spec.get("mode") or "")
    if mode == "dataset_cartesian":
        return _resolve_cartesian_dataset_candidate(parsed, structured_spec=structured_spec, state=state)
    if mode == "ocm_encoded_domain":
        return _resolve_ocm_encoded_candidate(parsed, structured_spec=structured_spec, state=state)
    if mode == "suzuki_encoded_domain":
        return _resolve_suzuki_encoded_candidate(parsed, structured_spec=structured_spec, state=state)
    if mode == "ocm_factorized_dataset":
        return _resolve_ocm_factorized_candidate(parsed, structured_spec=structured_spec, state=state)
    if mode == "declared_discrete_simplex":
        return _resolve_discrete_simplex_candidate(parsed, structured_spec=structured_spec, state=state)
    return _resolve_generic_variable_candidate(parsed, structured_spec=structured_spec, state=state)


def _resolve_cartesian_dataset_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    if oracle is None:
        return None, "Dataset oracle is unavailable for cartesian selection."
    raw_variables = parsed.get("variables", {}) if isinstance(parsed.get("variables"), dict) else {}
    candidate: dict[str, Any] = {}
    for column in structured_spec.get("feature_columns", []):
        exact_values = list(structured_spec.get("exact_values", {}).get(column, []))
        choice_map = dict(structured_spec.get("choice_maps", {}).get(column, {}))
        raw_value = raw_variables.get(column)
        matched = _match_structured_choice(raw_value, exact_values=exact_values, choice_map=choice_map)
        if matched is None:
            return None, f"Invalid choice for `{column}`. Use one of the declared exact levels or option IDs."
        candidate[column] = matched
    return _normalize_and_validate_dataset_candidate(candidate, oracle=oracle, state=state)


def _resolve_ocm_encoded_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    dataset_path = structured_spec.get("ocm_dataset_path")
    if not dataset_path:
        return None, "OCM domain path is unavailable."
    try:
        candidate = decode_ocm_candidate(parsed, dataset_path=dataset_path)
    except ValueError as exc:
        return None, str(exc)

    if bool(structured_spec.get("dataset_backed")):
        try:
            row = decode_ocm_proposal(parsed, dataset_path=dataset_path)
        except ValueError as exc:
            return None, str(exc)
        candidate = {
            "M1": str(row["M1"]).strip(),
            "M2": str(row["M2"]).strip(),
            "M3": str(row["M3"]).strip(),
            "Support": str(row["Support"]).strip(),
            "Temp": str(row["Temp"]).strip(),
            "Ar_flow": str(row["Ar_flow"]).strip(),
            "CH4_flow": str(row["CH4_flow"]).strip(),
            "O2_flow": str(row["O2_flow"]).strip(),
            "CT": str(row["CT"]).strip(),
        }
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is not None:
            return _normalize_and_validate_dataset_candidate(candidate, oracle=oracle, state=state)

    normalized_candidate = {key: value for key, value in candidate.items() if key != "Name"}
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    if candidate_to_key(normalized_candidate) in observed_keys:
        return None, "That recommendation repeats an already observed experiment. Choose an unseen point."
    return normalized_candidate, ""


def _resolve_discrete_simplex_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    raw_variables = parsed.get("variables", {}) if isinstance(parsed.get("variables"), dict) else {}
    feature_columns = [str(column) for column in structured_spec.get("feature_columns", [])]
    allowed_values = {
        str(column): [str(value) for value in values]
        for column, values in (structured_spec.get("allowed_values", {}) or {}).items()
        if isinstance(values, list)
    }
    candidate: dict[str, Any] = {}
    for column in feature_columns:
        matched = _match_exact_value(raw_variables.get(column), allowed_values.get(column, []))
        if matched is None:
            return None, f"`{column}` must be one of the declared discrete simplex levels."
        candidate[column] = matched

    constraint_failure = _validate_structured_variable_constraints(
        candidate,
        structured_spec.get("variable_constraints", []),
    )
    if constraint_failure:
        return None, constraint_failure

    if bool(structured_spec.get("dataset_backed")):
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is None:
            return None, "Dataset validation was requested, but no dataset oracle is available."
        return _normalize_and_validate_dataset_candidate(candidate, oracle=oracle, state=state)

    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    if candidate_to_key(candidate) in observed_keys:
        return None, "That recommendation repeats an already observed experiment. Choose an unseen point."
    return candidate, ""


def _resolve_ocm_factorized_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    if oracle is None:
        return None, "Dataset oracle is unavailable for OCM factorized selection."
    combo_map = dict(structured_spec.get("combo_map", {}))
    flow_map = dict(structured_spec.get("flow_map", {}))
    combo_id = _match_mapping_key(parsed.get("catalyst_combo_id"), combo_map)
    if combo_id is None:
        combo_id = _match_combo_value(parsed.get("catalyst_combo_id"), combo_map)
    if combo_id is None:
        return None, "Invalid `catalyst_combo_id`. Choose one of the declared CatalystCombo IDs."
    flow_id = _match_mapping_key(parsed.get("flow_recipe_id"), flow_map)
    if flow_id is None:
        return None, "Invalid `flow_recipe_id`. Choose one of the declared FlowRecipe IDs."
    temperature = _match_exact_value(parsed.get("temperature"), structured_spec.get("temp_values", []))
    if temperature is None:
        return None, "Invalid `temperature`. Use one of the declared temperature values."
    if temperature not in set(structured_spec.get("allowed_temps_by_combo", {}).get(combo_id, [])):
        return None, f"{combo_id} cannot be combined with temperature {temperature} in the OCM dataset."
    combo_values = combo_map[combo_id]
    flow_values = flow_map[flow_id]
    candidate = {
        "M1": combo_values[0],
        "M2": combo_values[1],
        "M3": combo_values[2],
        "Support": combo_values[3],
        "Temp": temperature,
        "Ar_flow": flow_values[0],
        "CH4_flow": flow_values[1],
        "O2_flow": flow_values[2],
        "CT": flow_values[3],
    }
    return _normalize_and_validate_dataset_candidate(candidate, oracle=oracle, state=state)


def _resolve_suzuki_encoded_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    dataset_path = structured_spec.get("suzuki_dataset_path")
    if not dataset_path:
        return None, "Suzuki domain path is unavailable."
    try:
        candidate = decode_suzuki_candidate(parsed, dataset_path=dataset_path)
    except ValueError as exc:
        return None, str(exc)

    if bool(structured_spec.get("dataset_backed")):
        try:
            row = decode_suzuki_proposal(parsed, dataset_path=dataset_path)
        except ValueError as exc:
            return None, str(exc)
        candidate = {
            "Reactant_1_Name": str(row["Reactant_1_Name"]).strip(),
            "Reactant_2_Name": str(row["Reactant_2_Name"]).strip(),
            "Ligand_Short_Hand": str(row["Ligand_Short_Hand"]).strip(),
            "Reagent_1_Short_Hand": str(row["Reagent_1_Short_Hand"]).strip(),
            "Solvent_1_Short_Hand": str(row["Solvent_1_Short_Hand"]).strip(),
        }
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is not None:
            return _normalize_and_validate_dataset_candidate(candidate, oracle=oracle, state=state)

    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    if candidate_to_key(candidate) in observed_keys:
        return None, "That recommendation repeats an already observed experiment. Choose an unseen point."
    return candidate, ""


def _resolve_generic_variable_candidate(
    parsed: dict[str, Any],
    *,
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    raw_variables = parsed.get("variables", {}) if isinstance(parsed.get("variables"), dict) else {}
    candidate: dict[str, Any] = {}
    for variable in structured_spec.get("variables", []):
        name = str(variable.get("name") or "")
        if not name:
            continue
        raw_value = raw_variables.get(name)
        if variable.get("type") == "continuous":
            numeric = _coerce_finite_float(raw_value)
            if numeric is None:
                return None, f"Invalid numeric value for `{name}`."
            low, high = _continuous_domain_bounds(variable)
            if numeric < low or numeric > high:
                return None, f"`{name}` must stay within [{low}, {high}]."
            allowed = _continuous_allowed_values(variable)
            if allowed:
                matched = _match_exact_value(raw_value, allowed)
                if matched is None:
                    return None, f"`{name}` must be one of the declared exact levels."
                candidate[name] = matched
            else:
                candidate[name] = _format_continuous_choice(numeric, low=low, high=high)
            continue
        matched = _match_structured_choice(
            raw_value,
            exact_values=_variable_domain_labels(variable),
            choice_map=structured_spec.get("choice_maps", {}).get(name, {}),
        )
        if matched is None:
            return None, f"Invalid categorical choice for `{name}`."
        candidate[name] = matched

    constraint_failure = _validate_structured_variable_constraints(
        candidate,
        structured_spec.get("variable_constraints", []),
    )
    if constraint_failure:
        return None, constraint_failure

    if bool(structured_spec.get("dataset_backed")):
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is None:
            return None, "Dataset validation was requested, but no dataset oracle is available."
        return _normalize_and_validate_dataset_candidate(candidate, oracle=oracle, state=state)

    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    if candidate_to_key(candidate) in observed_keys:
        return None, "That recommendation repeats an already observed experiment. Choose an unseen point."
    return candidate, ""


def _validate_structured_variable_constraints(
    candidate: dict[str, Any],
    constraints: Any,
) -> str:
    if not isinstance(constraints, list):
        return ""
    for constraint in constraints:
        if not isinstance(constraint, dict) or constraint.get("type") != "sum_equals":
            continue
        names = [str(name) for name in constraint.get("variables", [])]
        values: list[float] = []
        for name in names:
            value = _coerce_finite_float(candidate.get(name))
            if value is None:
                return f"`{name}` must be numeric for the sum-equals constraint."
            values.append(value)
        target = _coerce_float(constraint.get("value"), default=1.0)
        tolerance = abs(_coerce_float(constraint.get("tolerance"), default=1e-6))
        total = sum(values)
        if abs(total - target) > tolerance:
            return f"`{' + '.join(names)}` must equal {target}; got {round(total, 9)}."
    return ""


def _normalize_and_validate_dataset_candidate(
    candidate: dict[str, Any],
    *,
    oracle: DatasetOracle,
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    try:
        matched = oracle.lookup(candidate)
    except KeyError:
        matched = _lookup_numeric_tolerant_dataset_candidate(candidate, oracle)
        if matched is None:
            return (
                None,
                "That variable combination does not correspond to a legal dataset row. Choose another unseen legal option.",
            )
    normalized = dict(matched.get("candidate", {}))
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    if candidate_to_key(normalized) in observed_keys:
        return None, "That recommendation repeats an already observed experiment. Choose an unseen point."
    return normalized, ""


def _lookup_numeric_tolerant_dataset_candidate(
    candidate: dict[str, Any],
    oracle: DatasetOracle,
) -> dict[str, Any] | None:
    for dataset_candidate in oracle.candidates:
        if all(
            _dataset_values_equivalent(candidate.get(column), dataset_candidate.get(column))
            for column in oracle.feature_columns
        ):
            return oracle.lookup(dataset_candidate)
    return None


def _dataset_values_equivalent(left: Any, right: Any) -> bool:
    left_text = str(left).strip()
    right_text = str(right).strip()
    if left_text == right_text:
        return True
    left_numeric = _coerce_finite_float(left)
    right_numeric = _coerce_finite_float(right)
    if left_numeric is None or right_numeric is None:
        return False
    return abs(left_numeric - right_numeric) < 1e-9


def _first_valid_unseen_candidate_from_structured_space(
    structured_spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any] | None:
    mode = str(structured_spec.get("mode") or "")
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    if mode == "dataset_cartesian":
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is None:
            return None
        for candidate in oracle.candidates:
            if candidate_to_key(candidate) not in observed_keys:
                return dict(candidate)
        return None
    if mode == "ocm_factorized_dataset":
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is None:
            return None
        for candidate in oracle.candidates:
            if candidate_to_key(candidate) not in observed_keys:
                return dict(candidate)
        return None
    if mode == "ocm_encoded_domain":
        dataset_path = structured_spec.get("ocm_dataset_path")
        if not dataset_path:
            return None
        try:
            domain = load_ocm_domain_spec(dataset_path)
        except Exception:
            return None
        for row in domain.dataframe.itertuples(index=False):
            candidate = {
                "M1": str(row.M1),
                "M2": str(row.M2),
                "M3": str(row.M3),
                "Support": str(row.Support),
                "Temp": str(row.Temp),
                "Ar_flow": str(row.Ar_flow),
                "CH4_flow": str(row.CH4_flow),
                "O2_flow": str(row.O2_flow),
                "CT": str(row.CT),
            }
            if candidate_to_key(candidate) not in observed_keys:
                return candidate
        return None
    if mode == "suzuki_encoded_domain":
        dataset_path = structured_spec.get("suzuki_dataset_path")
        if not dataset_path:
            return None
        try:
            domain = load_suzuki_domain_spec(dataset_path)
        except Exception:
            return None
        for row in domain.dataframe.itertuples(index=False):
            candidate = {
                "Reactant_1_Name": str(row.Reactant_1_Name),
                "Reactant_2_Name": str(row.Reactant_2_Name),
                "Ligand_Short_Hand": str(row.Ligand_Short_Hand),
                "Reagent_1_Short_Hand": str(row.Reagent_1_Short_Hand),
                "Solvent_1_Short_Hand": str(row.Solvent_1_Short_Hand),
            }
            if candidate_to_key(candidate) not in observed_keys:
                return candidate
        return None
    if bool(structured_spec.get("dataset_backed")):
        oracle = DatasetOracle.from_problem_spec(state.get("problem_spec", {}))
        if oracle is None:
            return None
        for candidate in oracle.candidates:
            if candidate_to_key(candidate) not in observed_keys:
                return dict(candidate)
        return None
    variables = structured_spec.get("variables", [])
    candidate_pool = build_bo_candidate_pool(
        variables,
        observed_keys=observed_keys,
        candidate_pool_size=128,
        seed=_state_seed(state),
        hard_constraints=[],
        candidate_pool=None,
    )
    return dict(candidate_pool[0]) if candidate_pool else None


def _pure_reasoning_selected_shortlist(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    shortlist = build_bo_shortlist_from_candidates([candidate], [])
    if shortlist:
        shortlist[0]["selection_step"] = 1
        shortlist[0]["selection_mode"] = "llm_direct_select"
    return shortlist


def _dataset_path_from_problem_spec(problem_spec: dict[str, Any]) -> str | None:
    dataset = problem_spec.get("dataset")
    if isinstance(dataset, dict) and dataset.get("csv_path"):
        return str(dataset.get("csv_path"))
    virtual_oracle = problem_spec.get("virtual_oracle")
    if isinstance(virtual_oracle, dict) and virtual_oracle.get("train_csv_path"):
        return str(virtual_oracle.get("train_csv_path"))
    return None


def _observed_candidate_count(state: dict[str, Any]) -> int:
    return sum(1 for item in state.get("observations", []) if item.get("candidate"))


def _first_unseen_oracle_candidate(oracle: DatasetOracle, state: dict[str, Any]) -> dict[str, Any] | None:
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in state.get("observations", [])
        if item.get("candidate")
    }
    for candidate in oracle.candidates:
        if candidate_to_key(candidate) not in observed_keys:
            return dict(candidate)
    return None


def _variable_map_output_schema(example_variables: dict[str, Any]) -> str:
    rendered = ",\n".join(
        [f'    "{key}": "{value}"' for key, value in example_variables.items()]
    )
    return """{
  "variables": {
%s
  },
  "reasoning": "...",
  "hypothesis_alignment": "...",
  "information_value": "...",
  "concerns": "...",
  "confidence": 0.75
}""" % rendered


def _sorted_choice_values(values: set[str]) -> list[str]:
    return sorted((str(value) for value in values), key=_choice_sort_key)


def _sorted_tuple_records(values: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return sorted(values, key=lambda record: tuple(_choice_sort_key(item) for item in record))


def _choice_sort_key(value: Any) -> tuple[int, float | str]:
    numeric = _coerce_finite_float(value)
    if numeric is not None:
        return (0, float(numeric))
    return (1, str(value))


def _choice_prefix(name: str) -> str:
    letters = [char for char in str(name) if char.isalpha()]
    if not letters:
        return "V"
    prefix = "".join(letters[:2]).upper()
    return prefix[:2] if prefix else "V"


def _match_structured_choice(
    raw_value: Any,
    *,
    exact_values: list[str],
    choice_map: dict[str, str],
) -> str | None:
    choice_id = _match_mapping_key(raw_value, choice_map)
    if choice_id is not None:
        return choice_map[choice_id]
    return _match_exact_value(raw_value, exact_values)


def _match_mapping_key(raw_value: Any, mapping: dict[str, Any]) -> str | None:
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        return None
    for key in mapping:
        if text == str(key).strip():
            return key
    return None


def _match_combo_value(raw_value: Any, combo_map: dict[str, tuple[str, str, str, str]]) -> str | None:
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        return None
    for combo_id, combo in combo_map.items():
        if text == "|".join(combo):
            return combo_id
    return None


def _match_exact_value(raw_value: Any, values: list[str]) -> str | None:
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        return None
    numeric = _coerce_finite_float(text)
    for value in values:
        if text == str(value).strip():
            return value
        value_numeric = _coerce_finite_float(value)
        if numeric is not None and value_numeric is not None and abs(numeric - value_numeric) < 1e-9:
            return value
    return None


def _variable_domain_labels(variable: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for item in variable.get("domain", []):
        if isinstance(item, dict):
            label = item.get("label") or item.get("name") or item.get("value")
            if label is not None:
                labels.append(str(label))
            continue
        labels.append(str(item))
    return labels


def _continuous_allowed_values(variable: dict[str, Any]) -> list[str]:
    for key in ("allowed_values", "discrete_values", "grid_values", "levels"):
        raw_values = variable.get(key)
        if isinstance(raw_values, list) and raw_values:
            return _sorted_choice_values({str(value) for value in raw_values})

    step = _coerce_finite_float(variable.get("step"))
    if step is None or step <= 0:
        return []
    low, high = _continuous_domain_bounds(variable)
    values: list[str] = []
    index = 0
    current = low
    while current <= high + 1e-9 and index < 10000:
        values.append(str(_format_continuous_choice(current, low=low, high=high)))
        index += 1
        current = low + index * step
    return _sorted_choice_values(set(values))


def _continuous_domain_bounds(variable: dict[str, Any]) -> tuple[float, float]:
    domain = list(variable.get("domain", [0.0, 1.0]))
    if len(domain) < 2:
        return 0.0, 1.0
    low = _coerce_float(domain[0], default=0.0)
    high = _coerce_float(domain[1], default=1.0)
    return (min(low, high), max(low, high))


def _format_continuous_choice(value: float, *, low: float, high: float) -> float | int:
    bounded = min(max(float(value), low), high)
    return round(bounded, 6)


def record_autobo_result(
    *,
    state: dict[str, Any],
    settings,
    selected: dict[str, Any],
    shortlist: list[dict[str, Any]],
    candidate: dict[str, Any],
    result_value: float,
) -> dict[str, Any]:
    autobo_state = _resolve_autobo_state(state.get("autobo_state", {}), settings)
    calibrator = ReverseCalibrator.from_dict({"plaus_records": autobo_state.get("llm_plaus_audit", [])})
    log_lines: list[str] = []
    calibrator.plaus_records = _resolve_pending_plausibility_records(
        calibrator.plaus_records,
        candidate,
        result_value,
    )
    effective_llm_weight = float(autobo_state.get("effective_llm_weight", 0.10))
    should_degrade, recommended_weight, degrade_reason = calibrator.should_degrade_llm_weight()
    if should_degrade:
        effective_llm_weight = min(effective_llm_weight, float(recommended_weight))
        log_lines.append(f"[autobo_llm_weight] degraded_to={effective_llm_weight:.2f} reason={degrade_reason}")

    return {
        "autobo_state": {
            **autobo_state,
            "llm_plaus_audit": _trim_autobo_list(calibrator.plaus_records, limit=50),
            "effective_llm_weight": effective_llm_weight,
        },
        "log_lines": log_lines,
    }


class SurrogatePool:
    """Fit and query a pool of surrogate models while isolating failures."""

    def __init__(
        self,
        specs: list[SurrogateSpec] | None = None,
        search_space: list[dict[str, Any]] | None = None,
        feature_spec: dict[str, Any] | None = None,
    ):
        resolved_specs = DEFAULT_SURROGATE_SPECS if specs is None else specs
        self.specs = {spec.model_id: spec for spec in resolved_specs}
        self.search_space = list(search_space or [])
        self.feature_spec = dict(feature_spec or {})
        self.models: dict[str, BaseSurrogateModel] = {}
        self.fit_status: dict[str, bool] = {}
        self.fit_errors: dict[str, str] = {}

    def fit_all(self, candidates: list[dict[str, Any]], y: np.ndarray) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for model_id, spec in self.specs.items():
            try:
                model = _create_surrogate_from_spec(spec, self.search_space, self.feature_spec)
                model.fit(candidates, y)
                self.models[model_id] = model
                self.fit_status[model_id] = True
                self.fit_errors[model_id] = ""
                results[model_id] = {"success": True, "error": ""}
            except Exception as exc:  # pragma: no cover - best effort isolation
                self.fit_status[model_id] = False
                self.fit_errors[model_id] = f"{type(exc).__name__}: {exc}"
                results[model_id] = {"success": False, "error": self.fit_errors[model_id]}
        return results

    def predict(self, model_id: str, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        model = self.models.get(model_id)
        if model is None:
            raise RuntimeError(f"Model '{model_id}' is not fitted.")
        return model.predict(candidates)

    def predict_all(self, candidates: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model_id, ok in self.fit_status.items():
            if not ok:
                continue
            model = self.models.get(model_id)
            if model is None:
                continue
            try:
                outputs[model_id] = model.predict(candidates)
            except Exception:  # pragma: no cover
                continue
        return outputs

    def get_fitted_ids(self) -> list[str]:
        return [model_id for model_id, ok in self.fit_status.items() if ok]

    def get_active_model(self, active_id: str) -> BaseSurrogateModel | None:
        if self.fit_status.get(active_id):
            return self.models.get(active_id)
        return None


@dataclass
class FitnessScores:
    model_id: str
    f_seq: float = 0.0
    f_cal: float = 0.0
    f_rank: float = 0.0
    f_llm: float = 0.0
    composite: float = 0.0


class FitnessTracker:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        seq_start_n: int = 8,
        ci_level: float = 0.95,
        coverage_history: dict[str, list[float]] | None = None,
        last_loocv_fold_hits: dict[str, list[bool]] | None = None,
    ):
        self.weights = dict(weights or {"seq": 0.45, "cal": 0.25, "rank": 0.20, "llm": 0.10})
        self.seq_start_n = max(0, int(seq_start_n))  # deprecated under full LOOCV mode
        self.ci_level = float(ci_level)
        self.z_score = _z_score_for_ci(self.ci_level)
        self.seq_log: dict[str, list[float]] = {}
        self.last_loocv_fold_hits: dict[str, list[bool]] = {
            str(key): [bool(item) for item in value]
            for key, value in (last_loocv_fold_hits or {}).items()
            if isinstance(value, list)
        }
        self.coverage_history: dict[str, list[float]] = {
            str(key): [
                float(item)
                for item in value
                if isinstance(item, (int, float)) and np.isfinite(float(item))
            ][-20:]
            for key, value in (coverage_history or {}).items()
            if isinstance(value, list)
        }
        self.cal_log = self.last_loocv_fold_hits
        self.latest_scores: dict[str, FitnessScores] = {}

    def compute_loocv_predictions(
        self,
        model_id: str,
        spec: SurrogateSpec,
        search_space: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        feature_spec: dict[str, Any] | None = None,
        torch_device: str | None = None,
    ) -> LOOCVResult:
        candidates, y_raw = _observations_to_candidates(observations)
        n_obs = len(candidates)
        if n_obs < 2:
            return LOOCVResult(
                model_id=model_id,
                mu=np.zeros(n_obs, dtype=float),
                sigma=np.ones(n_obs, dtype=float),
                y_true=np.asarray(y_raw, dtype=float),
            )

        mu_raw = np.zeros(n_obs, dtype=float)
        sigma_raw = np.zeros(n_obs, dtype=float)
        failed_folds = 0
        fallback_mu = float(np.mean(y_raw)) if len(y_raw) else 0.0
        fallback_sigma = max(float(np.std(y_raw)) if len(y_raw) else 0.0, 1.0)
        for index in range(n_obs):
            train_candidates = [candidate for idx, candidate in enumerate(candidates) if idx != index]
            train_y_raw = np.asarray([value for idx, value in enumerate(y_raw) if idx != index], dtype=float)
            if not train_candidates:
                raise RuntimeError(f"LOOCV for {model_id} requires at least one training point per fold.")
            try:
                model = _create_surrogate_from_spec(spec, search_space, feature_spec, torch_device=torch_device)
                model.fit(train_candidates, train_y_raw)
                fold_mu, fold_sigma = model.predict([candidates[index]])
                mu_raw[index] = float(np.asarray(fold_mu, dtype=float)[0])
                sigma_raw[index] = float(max(np.asarray(fold_sigma, dtype=float)[0], 1e-6))
            except Exception:
                failed_folds += 1
                mu_raw[index] = fallback_mu
                sigma_raw[index] = fallback_sigma

        if n_obs and failed_folds > max(1, int(np.floor(0.30 * n_obs))):
            raise RuntimeError(f"LOOCV failure rate {failed_folds}/{n_obs} too high for {model_id}.")

        return LOOCVResult(
            model_id=model_id,
            mu=np.asarray(mu_raw, dtype=float),
            sigma=np.maximum(np.asarray(sigma_raw, dtype=float), 1e-6),
            y_true=np.asarray(y_raw, dtype=float),
        )

    def compute_loocv_metrics(
        self,
        model_id: str,
        spec: SurrogateSpec,
        search_space: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        feature_spec: dict[str, Any] | None = None,
        torch_device: str | None = None,
        direction: str = "maximize",
    ) -> FitnessScores:
        loocv = self.compute_loocv_predictions(
            model_id,
            spec,
            search_space,
            observations,
            feature_spec=feature_spec,
            torch_device=torch_device,
        )
        sigma_safe = np.maximum(np.asarray(loocv.sigma, dtype=float), 1e-6)
        y_true = np.asarray(loocv.y_true, dtype=float)
        mu = np.asarray(loocv.mu, dtype=float)

        log_likelihood = -0.5 * np.log(2.0 * np.pi * sigma_safe**2) - 0.5 * ((y_true - mu) / sigma_safe) ** 2
        f_seq = float(np.mean(log_likelihood)) if len(log_likelihood) else 0.0
        self.seq_log.setdefault(model_id, []).append(f_seq)

        lower = mu - self.z_score * sigma_safe
        upper = mu + self.z_score * sigma_safe
        in_ci = (y_true >= lower) & (y_true <= upper)
        self.last_loocv_fold_hits[model_id] = [bool(item) for item in in_ci.tolist()]
        self.cal_log = self.last_loocv_fold_hits
        coverage = float(np.mean(in_ci)) if len(in_ci) else 0.0
        history = self.coverage_history.setdefault(model_id, [])
        history.append(coverage)
        self.coverage_history[model_id] = history[-20:]
        f_cal = -abs(coverage - self.ci_level)

        if len(y_true) < 3:
            f_rank = 0.0
        else:
            if len(y_true) < 5:
                rank_indices = np.arange(len(y_true))
            elif direction == "minimize":
                rank_indices = np.argsort(y_true)[:5]
            else:
                rank_indices = np.argsort(y_true)[-5:][::-1]
            f_rank = _safe_spearman(mu[rank_indices], y_true[rank_indices]) if len(rank_indices) >= 3 else 0.0

        return FitnessScores(
            model_id=model_id,
            f_seq=f_seq,
            f_cal=float(f_cal),
            f_rank=float(f_rank),
        )

    def compute_composite(
        self,
        fitted_ids: list[str],
        f_llm_scores: dict[str, float] | None = None,
        effective_llm_weight: float = 0.10,
    ) -> dict[str, FitnessScores]:
        if not fitted_ids:
            return {}
        f_llm_scores = f_llm_scores or {}
        raw: dict[str, dict[str, float]] = {}
        for model_id in fitted_ids:
            latest = self.latest_scores.get(model_id, FitnessScores(model_id))
            raw[model_id] = {
                "seq": latest.f_seq,
                "cal": latest.f_cal,
                "rank": latest.f_rank,
                "llm": float(f_llm_scores.get(model_id, 0.0)),
            }

        normalized: dict[str, dict[str, float]] = {model_id: {} for model_id in fitted_ids}
        for signal in ("seq", "cal", "rank", "llm"):
            values = np.asarray([raw[model_id][signal] for model_id in fitted_ids], dtype=float)
            mean_val = float(np.mean(values))
            std_val = float(np.std(values)) or 1.0
            for model_id in fitted_ids:
                normalized[model_id][signal] = (raw[model_id][signal] - mean_val) / std_val

        weights = dict(self.weights)
        if not f_llm_scores:
            llm_weight = float(weights.get("llm", 0.10))
            residual = max(1.0 - llm_weight, 1e-6)
            for signal in ("seq", "cal", "rank"):
                weights[signal] = float(weights.get(signal, 0.0)) / residual
            weights["llm"] = 0.0
        else:
            weights["llm"] = float(effective_llm_weight)
            total = sum(float(value) for value in weights.values()) or 1.0
            weights = {key: float(value) / total for key, value in weights.items()}

        result: dict[str, FitnessScores] = {}
        for model_id in fitted_ids:
            z_values = normalized[model_id]
            composite = (
                weights.get("seq", 0.0) * z_values["seq"]
                + weights.get("cal", 0.0) * z_values["cal"]
                + weights.get("rank", 0.0) * z_values["rank"]
                + weights.get("llm", 0.0) * z_values["llm"]
            )
            result[model_id] = FitnessScores(
                model_id=model_id,
                f_seq=raw[model_id]["seq"],
                f_cal=raw[model_id]["cal"],
                f_rank=raw[model_id]["rank"],
                f_llm=raw[model_id]["llm"],
                composite=float(composite),
            )
        self.latest_scores = result
        return result


def _parallel_loocv_evaluate(
    *,
    eligible_specs: list[SurrogateSpec],
    search_space: list[dict[str, Any]],
    deduped_observations: list[dict[str, Any]],
    feature_spec: dict[str, Any] | None,
    direction: str,
    settings,
) -> tuple[dict[str, FitnessScores], dict[str, dict[str, Any]], FitnessTracker]:
    """Evaluate eligible surrogates with independent LOOCV trackers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    loocv_scores: dict[str, FitnessScores] = {}
    fit_results: dict[str, dict[str, Any]] = {}
    tracker = FitnessTracker(
        weights=dict(getattr(settings, "autobo_fitness_weights", {})),
        seq_start_n=0,
        ci_level=float(getattr(settings, "autobo_cal_ci_level", 0.95)),
    )

    devices = _loocv_torch_devices(settings)
    tasks = [
        (spec, devices[index % len(devices)] if devices else None)
        for index, spec in enumerate(eligible_specs)
    ]
    max_workers = _loocv_max_workers(settings, len(tasks))

    def _evaluate_one(spec: SurrogateSpec, torch_device: str | None) -> tuple[str, FitnessScores | None, dict[str, Any], dict[str, list[float]], dict[str, list[bool]]]:
        try:
            local_tracker = FitnessTracker(
                weights=dict(getattr(settings, "autobo_fitness_weights", {})),
                seq_start_n=0,
                ci_level=float(getattr(settings, "autobo_cal_ci_level", 0.95)),
            )
            score = local_tracker.compute_loocv_metrics(
                spec.model_id,
                spec,
                search_space,
                deduped_observations,
                feature_spec=feature_spec,
                torch_device=torch_device,
                direction=direction,
            )
            return (
                spec.model_id,
                score,
                {"success": True, "error": "", "stage": "loocv", "torch_device": torch_device},
                dict(local_tracker.coverage_history),
                dict(local_tracker.last_loocv_fold_hits),
            )
        except Exception as exc:
            return (
                spec.model_id,
                None,
                {"success": False, "error": f"{type(exc).__name__}: {exc}", "stage": "loocv", "torch_device": torch_device},
                {},
                {},
            )

    if len(tasks) <= 1 or max_workers <= 1:
        results = [_evaluate_one(spec, torch_device) for spec, torch_device in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_evaluate_one, spec, torch_device) for spec, torch_device in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    for model_id, score, result, coverage_history, fold_hits in results:
        fit_results[model_id] = result
        if score is None:
            continue
        loocv_scores[model_id] = score
        tracker.latest_scores[model_id] = score
        tracker.coverage_history.update(coverage_history)
        tracker.last_loocv_fold_hits.update(fold_hits)
    tracker.cal_log = tracker.last_loocv_fold_hits
    return loocv_scores, fit_results, tracker


class TriggerMonitor:
    def __init__(self, settings_dict: dict[str, Any]):
        self.layer2_min_interval = int(settings_dict.get("autobo_layer2_min_interval", 8))
        self.hysteresis_cooldown = int(settings_dict.get("autobo_hysteresis_cooldown", 3))
        self.switch_threshold = float(settings_dict.get("autobo_switch_threshold", 0.50))
        self.consecutive_lead_required = max(1, int(settings_dict.get("autobo_consecutive_lead", 1)))
        self.seq_lead_threshold = float(settings_dict.get("autobo_seq_lead_threshold", 1.5))
        self.active_distress_seq_gap = float(settings_dict.get("autobo_active_distress_seq_gap", 2.0))
        self.active_distress_cal_floor = float(settings_dict.get("autobo_active_distress_cal_floor", -0.45))
        self.distress_switch_threshold = float(settings_dict.get("autobo_distress_switch_threshold", 0.25))
        self.distress_bypass_hysteresis = bool(settings_dict.get("autobo_distress_bypass_hysteresis", True))
        self.cal_lower = float(settings_dict.get("autobo_cal_lower_bound", 0.70))
        self.cal_upper = float(settings_dict.get("autobo_cal_upper_bound", 0.99))
        self.stagnation_window = int(settings_dict.get("autobo_stagnation_window", 3))
        self._challenger_lead_streak: dict[str, int] = {}
        self.last_switch_decision: dict[str, Any] = {}

    def _record_switch_decision(
        self,
        *,
        active_model_id: str,
        top_candidate: FitnessScores | None = None,
        active_score: FitnessScores | None = None,
        gap: float | None = None,
        effective_threshold: float | None = None,
        streak: int = 0,
        hysteresis_blocked: bool = False,
        active_distressed: bool = False,
        switch_subtype: str = "no_change",
        decision_reason: str = "",
    ) -> None:
        self.last_switch_decision = {
            "active_model": active_model_id,
            "top_challenger": top_candidate.model_id if top_candidate is not None else None,
            "active_composite": active_score.composite if active_score is not None else None,
            "top_composite": top_candidate.composite if top_candidate is not None else None,
            "gap": gap,
            "effective_threshold": effective_threshold,
            "streak": streak,
            "required_streak": self.consecutive_lead_required,
            "hysteresis_blocked": bool(hysteresis_blocked),
            "active_distressed": bool(active_distressed),
            "switch_subtype": switch_subtype,
            "decision_reason": decision_reason,
        }

    def check_layer1(
        self,
        active_model_id: str,
        fitness_tracker: FitnessTracker,
        iteration: int,
        last_layer2_iter: int,
        performance_log: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        active_scores = fitness_tracker.latest_scores.get(active_model_id)
        if active_scores is None:
            return False, "Active model has no scores"

        for model_id, challenger in fitness_tracker.latest_scores.items():
            if model_id == active_model_id:
                continue
            if challenger.f_seq - active_scores.f_seq > self.seq_lead_threshold:
                return True, f"Challenger {model_id} leads in F_seq"

        coverage_series = fitness_tracker.coverage_history.get(active_model_id, [])
        if len(coverage_series) >= 5:
            recent_5 = [float(item) for item in coverage_series[-5:]]
            mean_recent = float(np.mean(recent_5))
            if mean_recent < self.cal_lower or mean_recent > self.cal_upper:
                if len(coverage_series) >= 10:
                    prev_5 = [float(item) for item in coverage_series[-10:-5]]
                    mean_prev = float(np.mean(prev_5))
                    drift = abs(mean_recent - mean_prev)
                    severe_low = mean_recent < self.cal_lower * 0.7
                    severe_high = mean_recent > min(1.0, self.cal_upper + 0.005)
                    if drift >= 0.10 or severe_low or severe_high:
                        return True, f"Active coverage drift: {mean_prev:.2f} -> {mean_recent:.2f}"
                else:
                    return True, f"Active coverage early-stage out of range: {mean_recent:.2f}"

        if len(performance_log) >= self.stagnation_window:
            recent = performance_log[-self.stagnation_window :]
            if all(not bool(item.get("improved", False)) for item in recent):
                return True, f"No improvement in last {self.stagnation_window} iterations"

        if int(iteration) - int(last_layer2_iter) >= self.layer2_min_interval:
            return True, f"Periodic refresh interval={self.layer2_min_interval}"

        return False, ""

    def decide_switch(
        self,
        active_model_id: str,
        composite_scores: dict[str, FitnessScores],
        iteration: int,
        hysteresis_until: int,
    ) -> tuple[str, bool, str, str]:
        if not composite_scores:
            self._record_switch_decision(
                active_model_id=active_model_id,
                decision_reason="No composite scores available",
            )
            return active_model_id, False, "no_change", "No composite scores available"

        ranked = sorted(composite_scores.values(), key=lambda item: item.composite, reverse=True)
        top_candidate = ranked[0]
        active_score = composite_scores.get(active_model_id)
        if active_score is None:
            self._challenger_lead_streak.clear()
            self._record_switch_decision(
                active_model_id=active_model_id,
                top_candidate=top_candidate,
                switch_subtype="active_failed",
                decision_reason=f"Active model {active_model_id} has no score",
            )
            return top_candidate.model_id, True, "active_failed", f"Active model {active_model_id} has no score"
        if top_candidate.model_id == active_model_id:
            self._challenger_lead_streak.clear()
            self._record_switch_decision(
                active_model_id=active_model_id,
                top_candidate=top_candidate,
                active_score=active_score,
                gap=0.0,
                effective_threshold=self.switch_threshold,
                decision_reason="Active model remains top-ranked",
            )
            return active_model_id, False, "no_change", "Active model remains top-ranked"
        gap = float(top_candidate.composite - active_score.composite)

        seq_gap = float(top_candidate.f_seq - active_score.f_seq)
        active_distressed = seq_gap >= self.active_distress_seq_gap or float(active_score.f_cal) <= self.active_distress_cal_floor
        effective_threshold = self.distress_switch_threshold if active_distressed else self.switch_threshold
        hysteresis_blocked = int(iteration) < int(hysteresis_until)
        if hysteresis_blocked and (not active_distressed or not self.distress_bypass_hysteresis):
            self._record_switch_decision(
                active_model_id=active_model_id,
                top_candidate=top_candidate,
                active_score=active_score,
                gap=gap,
                effective_threshold=effective_threshold,
                hysteresis_blocked=True,
                active_distressed=active_distressed,
                decision_reason="In hysteresis cooldown",
            )
            return active_model_id, False, "no_change", "In hysteresis cooldown"

        if gap >= effective_threshold:
            if active_distressed:
                self._challenger_lead_streak.clear()
                reason = (
                    f"Switching from {active_model_id} to {top_candidate.model_id} "
                    f"with active_distress gap={gap:.3f}, seq_gap={seq_gap:.3f}, "
                    f"active_f_cal={active_score.f_cal:.3f}"
                )
                self._record_switch_decision(
                    active_model_id=active_model_id,
                    top_candidate=top_candidate,
                    active_score=active_score,
                    gap=gap,
                    effective_threshold=effective_threshold,
                    hysteresis_blocked=hysteresis_blocked,
                    active_distressed=True,
                    switch_subtype="active_distress",
                    decision_reason=reason,
                )
                return top_candidate.model_id, True, "deliberate", reason

            next_streak = self._challenger_lead_streak.get(top_candidate.model_id, 0) + 1
            self._challenger_lead_streak = {top_candidate.model_id: next_streak}
            if next_streak >= self.consecutive_lead_required:
                self._challenger_lead_streak.clear()
                reason = (
                    f"Switching from {active_model_id} to {top_candidate.model_id} "
                    f"with gap={gap:.3f}, lead_streak={self.consecutive_lead_required}"
                )
                self._record_switch_decision(
                    active_model_id=active_model_id,
                    top_candidate=top_candidate,
                    active_score=active_score,
                    gap=gap,
                    effective_threshold=effective_threshold,
                    streak=next_streak,
                    switch_subtype="normal_gap",
                    decision_reason=reason,
                )
                return top_candidate.model_id, True, "deliberate", reason
            reason = (
                f"Challenger {top_candidate.model_id} leads by {gap:.3f}; "
                f"waiting for confirmation {next_streak}/{self.consecutive_lead_required}"
            )
            self._record_switch_decision(
                active_model_id=active_model_id,
                top_candidate=top_candidate,
                active_score=active_score,
                gap=gap,
                effective_threshold=effective_threshold,
                streak=next_streak,
                switch_subtype="normal_gap",
                decision_reason=reason,
            )
            return active_model_id, False, "no_change", reason
        self._challenger_lead_streak.clear()
        reason = f"Top gap {gap:.3f} below threshold {effective_threshold:.3f}"
        self._record_switch_decision(
            active_model_id=active_model_id,
            top_candidate=top_candidate,
            active_score=active_score,
            gap=gap,
            effective_threshold=effective_threshold,
            active_distressed=active_distressed,
            switch_subtype="active_distress" if active_distressed else "normal_gap",
            decision_reason=reason,
        )
        return active_model_id, False, "no_change", reason


class AcquisitionFlow:
    def __init__(
        self,
        top_k: int = 8,
        prefilter_multiplier: int = 10,
        hallucination_mode: str = "kriging_believer",
    ):
        self.top_k = max(1, int(top_k))
        self.prefilter_multiplier = max(1, int(prefilter_multiplier))
        self.hallucination_mode = str(hallucination_mode or "kriging_believer").strip().lower()
        self.last_prefilter_size = 0

    def propose_candidates(
        self,
        active_model: BaseSurrogateModel,
        refit_model_factory: Callable[[], BaseSurrogateModel] | None,
        candidate_pool: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        direction: str = "maximize",
        seed: int = 0,
        cpu_refit_model_factory: Callable[[], BaseSurrogateModel] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidate_pool:
            self.last_prefilter_size = 0
            return []

        try:
            scale_context = _build_observation_scale_context(observations, direction=direction)
            shortlist = _build_sequential_fantasized_shortlist(
                active_model=active_model,
                refit_model_factory=refit_model_factory,
                candidate_pool=candidate_pool,
                scale_context=scale_context,
                top_k=self.top_k,
                prefilter_multiplier=self.prefilter_multiplier,
                hallucination_mode=self.hallucination_mode,
                seed=seed,
            )
            self.last_prefilter_size = int(
                min(
                    len(candidate_pool),
                    max(self.top_k, self.prefilter_multiplier * self.top_k),
                )
            )
            for item in shortlist:
                item.pop("_predicted_value_scaled", None)
            return shortlist
        except Exception:
            rng = np.random.default_rng(seed)
            indices = list(rng.choice(len(candidate_pool), size=min(self.top_k, len(candidate_pool)), replace=False))
            self.last_prefilter_size = min(len(candidate_pool), max(self.top_k, self.prefilter_multiplier * self.top_k))
            return [
                {
                    "candidate": dict(candidate_pool[index]),
                    "predicted_value": None,
                    "uncertainty": None,
                    "acquisition_value": None,
                    "acquisition_value_raw": None,
                    "selection_step": rank + 1,
                    "selection_mode": "fallback_random",
                    "rank": rank + 1,
                }
                for rank, index in enumerate(indices)
            ]


def _build_observation_scale_context(
    observations: list[dict[str, Any]],
    *,
    direction: str,
) -> dict[str, Any]:
    valid = [dict(item) for item in observations if item.get("result") is not None]
    results = np.asarray([float(item["result"]) for item in valid], dtype=float)
    if direction == "minimize":
        y_model = -1.0 * results
    else:
        y_model = results
    scaled_observations = [
        {
            **item,
            "result": float(y_model[index]),
        }
        for index, item in enumerate(valid)
    ]
    return {
        "observations_scaled": scaled_observations,
        "y_mean": 0.0,
        "y_std": 1.0,
        "best_f_scaled": float(np.max(y_model)) if len(y_model) else 0.0,
        "direction": direction,
    }


def _score_candidate_pool(
    *,
    surrogate: BaseSurrogateModel,
    candidate_pool: list[dict[str, Any]],
    best_f_scaled: float,
    y_mean: float,
    y_std: float,
    direction: str,
    seed: int,
) -> dict[str, Any]:
    if isinstance(surrogate, RobustNLLSurrogate):
        return _score_candidate_pool_with_af(
            af_key="qlogei", surrogate=surrogate, candidate_pool=candidate_pool,
            best_f_scaled=best_f_scaled, y_mean=y_mean, y_std=y_std, direction=direction, seed=seed,
        )
    pred_mean_model, pred_std_model = surrogate.predict(candidate_pool)
    pred_mean_model = np.asarray(pred_mean_model, dtype=float)
    pred_std_model = np.maximum(np.asarray(pred_std_model, dtype=float), 1e-6)

    if isinstance(surrogate, CoCaBOGPSurrogate) and surrogate.model is not None:
        try:
            X_pool = surrogate.encode_candidates(candidate_pool)
            acquisition = create_acquisition("log_ei", {})
            acq_values = acquisition.score(surrogate, X_pool, best_f_scaled, np.random.default_rng(seed))
        except Exception:
            acq_values = _analytic_ei(pred_mean_model, pred_std_model, best_f_scaled)
    else:
        acq_values = _analytic_ei(pred_mean_model, pred_std_model, best_f_scaled)

    pred_mean = np.asarray(pred_mean_model, dtype=float)
    pred_std = np.maximum(np.asarray(pred_std_model, dtype=float), 1e-6)
    if direction == "minimize":
        pred_mean = -1.0 * pred_mean

    return {
        "candidate_pool": [dict(candidate) for candidate in candidate_pool],
        "pred_mean_scaled": pred_mean_model,
        "pred_std_scaled": pred_std_model,
        "pred_mean": np.asarray(pred_mean, dtype=float),
        "pred_std": np.asarray(pred_std, dtype=float),
        "acquisition": np.asarray(acq_values, dtype=float),
    }


def _score_candidate_pool_with_af(
    *,
    af_key: str,
    surrogate: BaseSurrogateModel,
    candidate_pool: list[dict[str, Any]],
    best_f_scaled: float,
    y_mean: float,
    y_std: float,
    direction: str,
    seed: int,
    ucb_beta: float | None = None,
) -> dict[str, Any]:
    normalized_af = str(af_key or "qlogei").strip().lower()
    if isinstance(surrogate, RobustNLLSurrogate):
        means, stds = surrogate.component_predictions(candidate_pool)
        pred_mean_model = surrogate.weights @ means
        pred_std_model = np.sqrt(surrogate.weights @ (stds**2 + (means - pred_mean_model)**2))
    else:
        pred_mean_model, pred_std_model = surrogate.predict(candidate_pool)
    pred_mean_model = np.asarray(pred_mean_model, dtype=float)
    pred_std_model = np.maximum(np.asarray(pred_std_model, dtype=float), 1e-6)

    if isinstance(surrogate, RobustNLLSurrogate) and normalized_af != "qucb":
        if normalized_af == "ts":
            index = np.random.default_rng(seed).choice(len(surrogate.models), p=surrogate.weights)
            acq_values = _score_candidate_pool_with_af(
                af_key="ts", surrogate=surrogate.models[index], candidate_pool=candidate_pool,
                best_f_scaled=best_f_scaled, y_mean=y_mean, y_std=y_std, direction=direction, seed=seed,
            )["acquisition"]
        else:
            log_eis = np.asarray([
                gaussian_log_ei(mu, sigma, best_f_scaled)
                for mu, sigma in zip(means, stds)
            ])
            acq_values = logsumexp(log_eis + np.log(surrogate.weights)[:, None], axis=0)
    elif normalized_af == "qucb":
        beta = max(float(ucb_beta if ucb_beta is not None else 1.0), 0.0)
        sigma_multiplier = float(np.sqrt(beta))
        if isinstance(surrogate, CoCaBOGPSurrogate) and surrogate.model is not None:
            try:
                X_pool = surrogate.encode_candidates(candidate_pool)
                acquisition = create_acquisition("ucb", {"beta": beta})
                acq_values = acquisition.score(surrogate, X_pool, best_f_scaled, np.random.default_rng(seed))
            except Exception:
                acq_values = pred_mean_model + sigma_multiplier * pred_std_model
        else:
            acq_values = pred_mean_model + sigma_multiplier * pred_std_model
    elif normalized_af == "ts":
        if isinstance(surrogate, CoCaBOGPSurrogate) and surrogate.model is not None:
            try:
                import torch

                X_pool = surrogate.encode_candidates(candidate_pool)
                with torch.random.fork_rng():
                    torch.manual_seed(int(seed))
                    with torch.no_grad():
                        posterior = surrogate.model.posterior(X_pool)
                        sample = posterior.rsample(sample_shape=torch.Size([1])).squeeze(0).squeeze(-1)
                acq_values = sample.detach().cpu().numpy().reshape(-1)
            except Exception:
                rng = np.random.default_rng(seed)
                acq_values = pred_mean_model + pred_std_model * rng.standard_normal(len(candidate_pool))
        else:
            rng = np.random.default_rng(seed)
            acq_values = pred_mean_model + pred_std_model * rng.standard_normal(len(candidate_pool))
    else:
        if isinstance(surrogate, CoCaBOGPSurrogate) and surrogate.model is not None:
            try:
                X_pool = surrogate.encode_candidates(candidate_pool)
                acquisition = create_acquisition("log_ei", {})
                acq_values = acquisition.score(surrogate, X_pool, best_f_scaled, np.random.default_rng(seed))
            except Exception:
                acq_values = _analytic_ei(pred_mean_model, pred_std_model, best_f_scaled)
        else:
            acq_values = _analytic_ei(pred_mean_model, pred_std_model, best_f_scaled)

    pred_mean = np.asarray(pred_mean_model, dtype=float)
    pred_std = np.maximum(np.asarray(pred_std_model, dtype=float), 1e-6)
    if direction == "minimize":
        pred_mean = -1.0 * pred_mean

    return {
        "candidate_pool": [dict(candidate) for candidate in candidate_pool],
        "pred_mean_scaled": pred_mean_model,
        "pred_std_scaled": pred_std_model,
        "pred_mean": np.asarray(pred_mean, dtype=float),
        "pred_std": np.asarray(pred_std, dtype=float),
        "acquisition": np.asarray(acq_values, dtype=float),
        "af_key": normalized_af,
    }


def _build_hallucinated_observations(
    selected_records: list[dict[str, Any]],
    *,
    hallucination_mode: str,
) -> list[dict[str, Any]]:
    normalized_mode = str(hallucination_mode or "kriging_believer").strip().lower()
    if normalized_mode != "kriging_believer":
        raise ValueError(f"Unsupported hallucination mode: {hallucination_mode}")
    hallucinated: list[dict[str, Any]] = []
    for item in selected_records:
        hallucinated.append(
            {
                "candidate": dict(item.get("candidate", {})),
                "result": float(item.get("_predicted_value_scaled", 0.0) or 0.0),
            }
        )
    return hallucinated


def _fit_fantasized_model(
    *,
    refit_model_factory: Callable[[], BaseSurrogateModel],
    candidates: list[dict[str, Any]],
    y: np.ndarray,
) -> BaseSurrogateModel:
    model = refit_model_factory()
    model.fit(candidates, y)
    return model


def _shortlist_record_from_scores(
    *,
    score_payload: dict[str, Any],
    candidate_index: int,
    selection_step: int,
    selection_mode: str,
    acquisition_value: float,
    acquisition_value_raw: float,
) -> dict[str, Any]:
    index = int(candidate_index)
    return {
        "candidate": dict(score_payload["candidate_pool"][index]),
        "predicted_value": float(score_payload["pred_mean"][index]),
        "uncertainty": float(score_payload["pred_std"][index]),
        "acquisition_value": float(acquisition_value),
        "acquisition_value_raw": float(acquisition_value_raw),
        "selection_step": int(selection_step),
        "selection_mode": str(selection_mode),
        "rank": int(selection_step),
        "_predicted_value_scaled": float(score_payload["pred_mean_scaled"][index]),
    }


def _build_sequential_fantasized_shortlist(
    *,
    active_model: BaseSurrogateModel,
    refit_model_factory: Callable[[], BaseSurrogateModel] | None,
    candidate_pool: list[dict[str, Any]],
    scale_context: dict[str, Any],
    top_k: int,
    prefilter_multiplier: int,
    hallucination_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not candidate_pool:
        return []

    top_k = max(1, int(top_k))
    prefilter_size = min(len(candidate_pool), max(top_k, int(prefilter_multiplier) * top_k))
    raw_scores = _score_candidate_pool(
        surrogate=active_model,
        candidate_pool=candidate_pool,
        best_f_scaled=float(scale_context.get("best_f_scaled", 0.0) or 0.0),
        y_mean=float(scale_context.get("y_mean", 0.0) or 0.0),
        y_std=float(scale_context.get("y_std", 1.0) or 1.0),
        direction=str(scale_context.get("direction") or "maximize"),
        seed=seed,
    )
    raw_acquisition = np.asarray(raw_scores["acquisition"], dtype=float)
    raw_order = np.argsort(raw_acquisition)[::-1]
    prefilter_indices = [int(index) for index in raw_order[:prefilter_size]]
    if not prefilter_indices:
        return []

    shortlist: list[dict[str, Any]] = []
    selected_global_indices: list[int] = []
    top1_index = int(prefilter_indices[0])
    shortlist.append(
        _shortlist_record_from_scores(
            score_payload=raw_scores,
            candidate_index=top1_index,
            selection_step=1,
            selection_mode="raw_top1",
            acquisition_value=float(raw_acquisition[top1_index]),
            acquisition_value_raw=float(raw_acquisition[top1_index]),
        )
    )
    selected_global_indices.append(top1_index)
    remaining_indices = [index for index in prefilter_indices if index != top1_index]

    while remaining_indices and len(shortlist) < top_k:
        fallback_index = max(remaining_indices, key=lambda index: float(raw_acquisition[int(index)]))
        conditioned_record: dict[str, Any] | None = None
        if refit_model_factory is not None:
            try:
                scaled_observations = list(scale_context.get("observations_scaled", []))
                hallucinated = _build_hallucinated_observations(shortlist, hallucination_mode=hallucination_mode)
                train_candidates = [item.get("candidate", {}) for item in scaled_observations + hallucinated]
                train_y = np.asarray([float(item.get("result", 0.0) or 0.0) for item in scaled_observations + hallucinated], dtype=float)
                fantasized_model = _fit_fantasized_model(
                    refit_model_factory=refit_model_factory,
                    candidates=train_candidates,
                    y=train_y,
                )
                remaining_pool = [candidate_pool[index] for index in remaining_indices]
                conditioned_scores = _score_candidate_pool(
                    surrogate=fantasized_model,
                    candidate_pool=remaining_pool,
                    best_f_scaled=float(scale_context.get("best_f_scaled", 0.0) or 0.0),
                    y_mean=float(scale_context.get("y_mean", 0.0) or 0.0),
                    y_std=float(scale_context.get("y_std", 1.0) or 1.0),
                    direction=str(scale_context.get("direction") or "maximize"),
                    seed=seed + len(shortlist),
                )
                local_best = int(np.argmax(np.asarray(conditioned_scores["acquisition"], dtype=float)))
                conditioned_record = _shortlist_record_from_scores(
                    score_payload=conditioned_scores,
                    candidate_index=local_best,
                    selection_step=len(shortlist) + 1,
                    selection_mode="fantasized_greedy",
                    acquisition_value=float(conditioned_scores["acquisition"][local_best]),
                    acquisition_value_raw=float(raw_acquisition[int(remaining_indices[local_best])]),
                )
                fallback_index = int(remaining_indices[local_best])
            except Exception:
                conditioned_record = None

        if conditioned_record is None:
            conditioned_record = _shortlist_record_from_scores(
                score_payload=raw_scores,
                candidate_index=fallback_index,
                selection_step=len(shortlist) + 1,
                selection_mode="fantasized_greedy",
                acquisition_value=float(raw_acquisition[fallback_index]),
                acquisition_value_raw=float(raw_acquisition[fallback_index]),
            )

        shortlist.append(conditioned_record)
        selected_global_indices.append(int(fallback_index))
        remaining_indices = [index for index in remaining_indices if int(index) != int(fallback_index)]

    return shortlist


def _adaptive_ucb_beta(iteration: int, stagnation_length: int, n_obs: int) -> float:
    del iteration
    base = 1.0
    early_boost = max(0.0, (8 - max(int(n_obs), 0)) / 8.0) * 0.75
    stagnation_boost = min(max(float(stagnation_length), 0.0) * 0.25, 1.25)
    return float(round(base + early_boost + stagnation_boost, 4))


def _ensemble_af_slot_targets(top_k: int, weights: dict[str, float] | None = None) -> dict[str, int]:
    total = max(1, int(top_k))
    if weights:
        normalized = {
            key: max(float(weights.get(key, 0.0) or 0.0), 0.0)
            for key in ("qlogei", "qucb", "ts")
        }
        weight_total = sum(normalized.values())
        if weight_total > 0:
            normalized = {key: value / weight_total for key, value in normalized.items()}
            raw = {key: total * value for key, value in normalized.items()}
            targets = {key: int(np.floor(value)) for key, value in raw.items()}
            remaining = total - sum(targets.values())
            fractions = sorted(
                raw,
                key=lambda key: (-(raw[key] - np.floor(raw[key])), ["qlogei", "qucb", "ts"].index(key)),
            )
            for key in fractions:
                if remaining <= 0:
                    break
                targets[key] += 1
                remaining -= 1
            if total >= 3:
                for key in ("qlogei", "qucb", "ts"):
                    if normalized.get(key, 0.0) > 0.05 and targets.get(key, 0) == 0:
                        biggest = max(targets, key=targets.get)
                        if targets.get(biggest, 0) > 1:
                            targets[biggest] -= 1
                            targets[key] = 1
            return {key: int(targets.get(key, 0)) for key in ("qlogei", "qucb", "ts")}
    if total == 1:
        return {"qlogei": 1, "qucb": 0, "ts": 0}
    if total == 2:
        return {"qlogei": 1, "qucb": 1, "ts": 0}

    targets = {"qlogei": 1, "qucb": 1, "ts": 1}
    remaining = total - 3
    weights = {"qlogei": 0.5, "qucb": 0.25, "ts": 0.25}
    raw_allocations = {af_key: remaining * weight for af_key, weight in weights.items()}
    for af_key, raw_value in raw_allocations.items():
        whole = int(np.floor(raw_value))
        targets[af_key] += whole
        remaining -= whole
    remainders = sorted(
        weights.keys(),
        key=lambda af_key: (-float(raw_allocations[af_key] - np.floor(raw_allocations[af_key])), ["qlogei", "qucb", "ts"].index(af_key)),
    )
    for af_key in remainders:
        if remaining <= 0:
            break
        targets[af_key] += 1
        remaining -= 1
    return targets


def _build_ranked_af_candidates(
    *,
    af_key: str,
    active_model: BaseSurrogateModel,
    refit_model_factory: Callable[[], BaseSurrogateModel] | None,
    candidate_pool: list[dict[str, Any]],
    scale_context: dict[str, Any],
    top_k: int,
    prefilter_multiplier: int,
    hallucination_mode: str,
    seed: int,
    ucb_beta: float | None = None,
) -> list[dict[str, Any]]:
    if not candidate_pool:
        return []

    normalized_af = str(af_key or "qlogei").strip().lower()
    top_k = max(1, int(top_k))
    prefilter_size = min(len(candidate_pool), max(top_k, int(prefilter_multiplier) * top_k))
    raw_scores = _score_candidate_pool_with_af(
        af_key=normalized_af,
        surrogate=active_model,
        candidate_pool=candidate_pool,
        best_f_scaled=float(scale_context.get("best_f_scaled", 0.0) or 0.0),
        y_mean=float(scale_context.get("y_mean", 0.0) or 0.0),
        y_std=float(scale_context.get("y_std", 1.0) or 1.0),
        direction=str(scale_context.get("direction") or "maximize"),
        seed=seed,
        ucb_beta=ucb_beta,
    )
    raw_acquisition = np.asarray(raw_scores["acquisition"], dtype=float)
    raw_order = np.argsort(raw_acquisition)[::-1]
    prefilter_indices = [int(index) for index in raw_order[:prefilter_size]]
    if not prefilter_indices:
        return []

    if normalized_af == "ts":
        return [
            {
                "candidate": dict(raw_scores["candidate_pool"][index]),
                "af_key": normalized_af,
                "af_rank": rank + 1,
            }
            for rank, index in enumerate(prefilter_indices[:top_k])
        ]

    ranked: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    top1_index = int(prefilter_indices[0])
    ranked.append({"candidate": dict(raw_scores["candidate_pool"][top1_index]), "af_key": normalized_af, "af_rank": 1})
    selected_records.append(
        _shortlist_record_from_scores(
            score_payload=raw_scores,
            candidate_index=top1_index,
            selection_step=1,
            selection_mode="raw_top1",
            acquisition_value=float(raw_acquisition[top1_index]),
            acquisition_value_raw=float(raw_acquisition[top1_index]),
        )
    )
    remaining_indices = [index for index in prefilter_indices if index != top1_index]

    while remaining_indices and len(ranked) < top_k:
        fallback_index = max(remaining_indices, key=lambda index: float(raw_acquisition[int(index)]))
        conditioned_index: int | None = None
        if refit_model_factory is not None:
            try:
                scaled_observations = list(scale_context.get("observations_scaled", []))
                hallucinated = _build_hallucinated_observations(selected_records, hallucination_mode=hallucination_mode)
                train_candidates = [item.get("candidate", {}) for item in scaled_observations + hallucinated]
                train_y = np.asarray(
                    [float(item.get("result", 0.0) or 0.0) for item in scaled_observations + hallucinated],
                    dtype=float,
                )
                fantasized_model = _fit_fantasized_model(
                    refit_model_factory=refit_model_factory,
                    candidates=train_candidates,
                    y=train_y,
                )
                remaining_pool = [candidate_pool[index] for index in remaining_indices]
                conditioned_scores = _score_candidate_pool_with_af(
                    af_key=normalized_af,
                    surrogate=fantasized_model,
                    candidate_pool=remaining_pool,
                    best_f_scaled=float(scale_context.get("best_f_scaled", 0.0) or 0.0),
                    y_mean=float(scale_context.get("y_mean", 0.0) or 0.0),
                    y_std=float(scale_context.get("y_std", 1.0) or 1.0),
                    direction=str(scale_context.get("direction") or "maximize"),
                    seed=seed + len(ranked),
                    ucb_beta=ucb_beta,
                )
                local_best = int(np.argmax(np.asarray(conditioned_scores["acquisition"], dtype=float)))
                conditioned_index = int(remaining_indices[local_best])
                selected_records.append(
                    _shortlist_record_from_scores(
                        score_payload=conditioned_scores,
                        candidate_index=local_best,
                        selection_step=len(ranked) + 1,
                        selection_mode="fantasized_greedy",
                        acquisition_value=float(conditioned_scores["acquisition"][local_best]),
                        acquisition_value_raw=float(raw_acquisition[conditioned_index]),
                    )
                )
            except Exception:
                conditioned_index = None

        if conditioned_index is None:
            conditioned_index = int(fallback_index)
            selected_records.append(
                _shortlist_record_from_scores(
                    score_payload=raw_scores,
                    candidate_index=conditioned_index,
                    selection_step=len(ranked) + 1,
                    selection_mode="fantasized_greedy",
                    acquisition_value=float(raw_acquisition[conditioned_index]),
                    acquisition_value_raw=float(raw_acquisition[conditioned_index]),
                )
            )

        ranked.append(
            {
                "candidate": dict(candidate_pool[conditioned_index]),
                "af_key": normalized_af,
                "af_rank": len(ranked) + 1,
            }
        )
        remaining_indices = [index for index in remaining_indices if int(index) != conditioned_index]

    return ranked


def _categorical_variables(search_space: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categorical: list[dict[str, Any]] = []
    for variable in search_space:
        if not isinstance(variable, dict):
            continue
        name = str(variable.get("name") or "").strip()
        if not name or variable.get("type", "categorical") == "continuous":
            continue
        labels = [label for label in variable.get("domain", []) if str(label).strip()]
        if labels:
            categorical.append(
                {
                    "name": name,
                    "labels": labels,
                    "role": str(variable.get("role") or variable.get("semantic_role") or name),
                }
            )
    return categorical


def _continuous_grid_variables(search_space: list[dict[str, Any]]) -> list[dict[str, Any]]:
    continuous_grid: list[dict[str, Any]] = []
    for variable in search_space:
        if not isinstance(variable, dict):
            continue
        name = str(variable.get("name") or "").strip()
        if not name or variable.get("type") != "continuous":
            continue
        labels = [label for label in _continuous_allowed_values(variable) if str(label).strip()]
        if labels:
            continuous_grid.append(
                {
                    "name": name,
                    "labels": labels,
                    "role": str(variable.get("role") or variable.get("semantic_role") or name),
                    "option_type": "continuous_grid",
                }
            )
    return continuous_grid


def _bo_round_index(observations: list[dict[str, Any]], warm_start_target: int) -> int:
    return int(len(observations)) - int(warm_start_target or 0) + 1


def _should_trigger_model_schema_evaluation(
    *,
    n_total_obs: int,
    warm_start_target: int,
    last_eval_n: int,
    eval_interval: int,
) -> tuple[bool, str]:
    interval = max(1, int(eval_interval or 1))
    total = max(0, int(n_total_obs or 0))
    warm_start = max(0, int(warm_start_target or 0))
    n_bo_obs = max(0, total - warm_start)
    warm_start_complete = warm_start <= 0 or total >= warm_start
    if not warm_start_complete:
        return False, "before_warm_start_complete"
    if total < 8:
        return False, "evaluation_not_due"
    if int(last_eval_n) < 0:
        return True, "warm_start_complete"
    if n_bo_obs - int(last_eval_n) >= interval:
        return True, "interval"
    return False, "evaluation_not_due"


def _early_post_warm_start_prompt_info(
    *,
    settings,
    observations: list[dict[str, Any]],
    warm_start_target: int,
) -> dict[str, Any]:
    window = max(0, int(getattr(settings, "autobo_unseen_category_window", 5) or 0))
    bo_round = _bo_round_index(observations, warm_start_target)
    return {
        "enabled": bool(1 <= bo_round <= window),
        "bo_round_index": bo_round,
        "window": window,
    }


def _unseen_category_coverage_should_run(
    *,
    settings,
    observations: list[dict[str, Any]],
    warm_start_target: int,
    ensemble_sur_enabled: bool,
    zero_llm_mode: bool,
) -> bool:
    if not _harness_enabled(settings):
        return False
    if not bool(getattr(settings, "autobo_unseen_category_exploration_enabled", True)):
        return False
    if bool(ensemble_sur_enabled) or bool(zero_llm_mode):
        return False
    slots = int(getattr(settings, "autobo_unseen_category_slots", 1) or 0)
    if slots <= 0:
        return False
    window = max(0, int(getattr(settings, "autobo_unseen_category_window", 5) or 0))
    bo_round = _bo_round_index(observations, warm_start_target)
    return 1 <= bo_round <= window


def _unseen_category_coverage_skip_audit(
    *,
    settings,
    observations: list[dict[str, Any]],
    warm_start_target: int,
    ensemble_sur_enabled: bool,
    zero_llm_mode: bool,
) -> dict[str, Any]:
    slots = int(getattr(settings, "autobo_unseen_category_slots", 1) or 0)
    window = max(0, int(getattr(settings, "autobo_unseen_category_window", 5) or 0))
    bo_round = _bo_round_index(observations, warm_start_target)
    skip_reason = ""
    enabled = _unseen_category_coverage_should_run(
        settings=settings,
        observations=observations,
        warm_start_target=warm_start_target,
        ensemble_sur_enabled=ensemble_sur_enabled,
        zero_llm_mode=zero_llm_mode,
    )
    if not _harness_enabled(settings):
        skip_reason = "harness_ablation"
    elif not bool(getattr(settings, "autobo_unseen_category_exploration_enabled", True)):
        skip_reason = "disabled"
    elif ensemble_sur_enabled:
        skip_reason = "ensemble_sur_enabled"
    elif zero_llm_mode:
        skip_reason = "zero_llm_mode"
    elif slots <= 0:
        skip_reason = "no_slots"
    elif bo_round < 1:
        skip_reason = "before_warm_start_complete"
    elif bo_round > window:
        skip_reason = "outside_window"
    return {
        "enabled": enabled,
        "skip_reason": skip_reason,
        "bo_round_index": bo_round,
        "window": window,
        "slots": max(0, slots),
        "unseen_options": {},
        "llm_targets": [],
        "validated_targets": [],
        "inserted_count": 0,
    }


def _build_unseen_category_options(
    *,
    search_space: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    include_continuous_grid: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    coverage_variables = _categorical_variables(search_space)
    if include_continuous_grid:
        coverage_variables.extend(_continuous_grid_variables(search_space))
    if not coverage_variables:
        return {}
    observed_by_name: dict[str, set[str]] = {item["name"]: set() for item in coverage_variables}
    for observation in observations:
        candidate = observation.get("candidate", {}) if isinstance(observation, dict) else {}
        if not isinstance(candidate, dict):
            continue
        for item in coverage_variables:
            name = item["name"]
            if name in candidate:
                observed_by_name[name].add(str(candidate.get(name)))

    unseen_options: dict[str, list[dict[str, Any]]] = {}
    for item in coverage_variables:
        name = item["name"]
        observed = observed_by_name.get(item["name"], set())
        options: list[dict[str, Any]] = []
        for label in item["labels"]:
            if str(label) in observed:
                continue
            legal_count = sum(1 for candidate in candidate_pool if str(candidate.get(name)) == str(label))
            option = {
                "value": label,
                "role": item.get("role") or name,
                "unseen": True,
                "legal_candidate_count": int(legal_count),
            }
            if item.get("option_type") == "continuous_grid":
                option["option_type"] = "continuous_grid"
            options.append(option)
        if options:
            unseen_options[name] = options
    return unseen_options


def _flatten_unseen_category_options(unseen_options: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for variable, options in unseen_options.items():
        for option in options:
            if not isinstance(option, dict):
                continue
            flattened.append(
                {
                    "variable": variable,
                    "value": option.get("value"),
                    "role": option.get("role"),
                    "unseen": True,
                    "legal_candidate_count": int(option.get("legal_candidate_count", 0) or 0),
                    **({"option_type": option.get("option_type")} if option.get("option_type") else {}),
                }
            )
    return sorted(flattened, key=lambda item: (-int(item.get("legal_candidate_count", 0) or 0), str(item["variable"]), str(item["value"])))


def _validate_unseen_category_targets(
    raw_targets: Any,
    unseen_options: dict[str, list[dict[str, Any]]],
    slots: int,
) -> list[dict[str, Any]]:
    max_slots = max(0, int(slots))
    if max_slots <= 0:
        return []
    option_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for variable, options in unseen_options.items():
        for option in options:
            option_lookup[(str(variable), str(option.get("value")))] = option

    if isinstance(raw_targets, dict):
        raw_targets = raw_targets.get("targets")
    if not isinstance(raw_targets, list):
        raw_targets = []

    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_targets:
        if not isinstance(raw, dict):
            continue
        variable = str(raw.get("variable") or "").strip()
        value = raw.get("value")
        key = (variable, str(value))
        if not variable or key in seen or key not in option_lookup:
            continue
        seen.add(key)
        option = option_lookup[key]
        accepted.append(
            {
                "variable": variable,
                "value": option.get("value"),
                "role": option.get("role"),
                "unseen": True,
                "selected_by_llm": True,
                "reasoning": str(raw.get("reasoning") or ""),
                "legal_candidate_count": int(option.get("legal_candidate_count", 0) or 0),
                **({"option_type": option.get("option_type")} if option.get("option_type") else {}),
            }
        )
        if len(accepted) >= max_slots:
            return accepted

    for option in _flatten_unseen_category_options(unseen_options):
        key = (str(option["variable"]), str(option.get("value")))
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            {
                "variable": option["variable"],
                "value": option.get("value"),
                "role": option.get("role"),
                "unseen": True,
                "selected_by_llm": False,
                "reasoning": "Mechanical fallback target after invalid, duplicate, or missing LLM coverage choices.",
                "legal_candidate_count": int(option.get("legal_candidate_count", 0) or 0),
                **({"option_type": option.get("option_type")} if option.get("option_type") else {}),
            }
        )
        if len(accepted) >= max_slots:
            break
    return accepted


def _coverage_records_for_targets(
    *,
    active_model: BaseSurrogateModel,
    candidate_pool: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    unseen_options: dict[str, list[dict[str, Any]]],
    normal_shortlist: list[dict[str, Any]],
    direction: str,
    seed: int,
    top_k: int,
    coverage_slots: int,
) -> list[dict[str, Any]]:
    if not candidate_pool or not targets:
        return []
    scale_context = _build_observation_scale_context(observations, direction=direction)
    normal_keep = max(0, int(top_k))
    reserved_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in normal_shortlist[:normal_keep]
        if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
    }
    selected_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    domain_sizes = {name: len(options) for name, options in unseen_options.items()}

    for target_index, target in enumerate(targets):
        variable = str(target.get("variable") or "").strip()
        value = target.get("value")
        if not variable:
            continue
        matching_pool = [dict(candidate) for candidate in candidate_pool if str(candidate.get(variable)) == str(value)]
        if not matching_pool:
            continue
        scores = _score_candidate_pool_with_af(
            af_key="qlogei",
            surrogate=active_model,
            candidate_pool=matching_pool,
            best_f_scaled=float(scale_context.get("best_f_scaled", 0.0) or 0.0),
            y_mean=float(scale_context.get("y_mean", 0.0) or 0.0),
            y_std=float(scale_context.get("y_std", 1.0) or 1.0),
            direction=direction,
            seed=seed + 31 + target_index,
        )
        acquisition = np.asarray(scores.get("acquisition", []), dtype=float)
        if len(acquisition) == 0:
            continue
        order = np.argsort(acquisition)[::-1]
        rank_lookup = {int(candidate_index): rank + 1 for rank, candidate_index in enumerate(order)}
        selected_index: int | None = None
        for candidate_index in order:
            key = candidate_to_key(scores["candidate_pool"][int(candidate_index)])
            if key in reserved_keys or key in selected_keys:
                continue
            selected_index = int(candidate_index)
            selected_keys.add(key)
            break
        if selected_index is None:
            continue
        coverage_target = {
            "variable": variable,
            "value": value,
            "unseen": True,
            "selected_by_llm": bool(target.get("selected_by_llm")),
            "reasoning": str(target.get("reasoning") or ""),
        }
        if target.get("option_type") == "continuous_grid":
            coverage_target["option_type"] = "continuous_grid"
        records.append(
            {
                "candidate": dict(scores["candidate_pool"][selected_index]),
                "predicted_value": float(np.asarray(scores["pred_mean"], dtype=float)[selected_index]),
                "uncertainty": float(np.asarray(scores["pred_std"], dtype=float)[selected_index]),
                "acquisition_value": float(acquisition[selected_index]),
                "acquisition_value_raw": float(acquisition[selected_index]),
                "selection_step": 0,
                "selection_mode": (
                    "llm_guided_unseen_grid_coverage"
                    if target.get("option_type") == "continuous_grid"
                    else "llm_guided_unseen_category_coverage"
                ),
                "rank": 0,
                "af_sources": ["coverage_qlogei"],
                "af_ranks": {"qlogei": int(rank_lookup.get(selected_index, 0) or 0)},
                "af_consensus_count": 0,
                "ensemble_reference_score": None,
                "ensemble_weighted_rank_score": None,
                "ensemble_diversity_bonus": 0.0,
                "coverage_targets": [coverage_target],
                "coverage_domain_size": int(domain_sizes.get(variable, 0) or 0),
            }
        )
        if len(records) >= max(0, int(coverage_slots)):
            break
    return records


def _build_llm_guided_unseen_category_coverage_records(
    *,
    state: dict[str, Any],
    settings,
    llm,
    invoke_json_node,
    active_model: BaseSurrogateModel,
    candidate_pool: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    search_space: list[dict[str, Any]],
    direction: str,
    normal_shortlist: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    warm_start_target = int(state.get("warm_start_target", 0) or 0)
    slots = max(0, int(getattr(settings, "autobo_unseen_category_slots", 1) or 0))
    audit = _unseen_category_coverage_skip_audit(
        settings=settings,
        observations=observations,
        warm_start_target=warm_start_target,
        ensemble_sur_enabled=False,
        zero_llm_mode=False,
    )
    unseen_options = _build_unseen_category_options(
        search_space=search_space,
        observations=observations,
        candidate_pool=candidate_pool,
        include_continuous_grid=_is_hpo_problem(state.get("problem_spec", {})),
    )
    audit["unseen_options"] = unseen_options
    if not unseen_options or slots <= 0:
        audit["enabled"] = False
        audit["skip_reason"] = "no_unseen_options" if not unseen_options else "no_slots"
        return [], audit, _empty_usage_delta()

    default_targets = {"targets": _flatten_unseen_category_options(unseen_options)[:slots]}
    memory_manager = MemoryManager.from_dict(state.get("memory", {}))
    coverage_state = {**state, "proposal_shortlist": list(normal_shortlist)}
    context = (
        _memoryless_autobo_acquisition_select_context(coverage_state)
        if not _memory_enabled(settings)
        else ContextBuilder.for_autobo_acquisition_select(
            coverage_state,
            memory_manager,
        )
    )
    prompt = build_unseen_category_coverage_prompt(
        reaction_context=context.get("reaction_context", {}),
        top_observations=context.get("top_observations", []),
        bottom_observations=context.get("bottom_observations", []),
        recent_observations=[] if not _memory_enabled(settings) else observations[-6:],
        unseen_options=unseen_options,
        total_observations=len(observations),
        coverage_slots=slots,
        knowledge_cards_text=context.get("knowledge_cards_text", ""),
        memory_rules=context.get("memory_rules", []),
        active_hypotheses=context.get("active_hypotheses", []),
    )
    try:
        parsed, _, usage = invoke_json_node(
            llm,
            state,
            prompt,
            default_targets,
            node_name="run_bo_iteration",
            lightweight=True,
        )
    except Exception as exc:
        parsed = {**default_targets, "error": f"{type(exc).__name__}: {exc}"}
        usage = _empty_usage_delta()
        audit["llm_error"] = parsed["error"]

    raw_targets = parsed.get("targets") if isinstance(parsed, dict) else []
    validated_targets = _validate_unseen_category_targets(raw_targets, unseen_options, slots)
    records = _coverage_records_for_targets(
        active_model=active_model,
        candidate_pool=candidate_pool,
        observations=observations,
        targets=validated_targets,
        unseen_options=unseen_options,
        normal_shortlist=normal_shortlist,
        direction=direction,
        seed=_state_seed(state),
        top_k=top_k,
        coverage_slots=slots,
    )
    audit.update(
        {
            "enabled": True,
            "skip_reason": "" if records else "no_valid_coverage_candidates",
            "llm_targets": raw_targets if isinstance(raw_targets, list) else [],
            "validated_targets": validated_targets,
            "inserted_count": len(records),
        }
    )
    return records, audit, usage


def _merge_shortlist_with_coverage(
    shortlist: list[dict[str, Any]],
    coverage_records: list[dict[str, Any]],
    *,
    top_k: int,
    coverage_slots: int,
) -> list[dict[str, Any]]:
    top_k = max(1, int(top_k))
    coverage_slots = max(0, int(coverage_slots))
    if not coverage_records or coverage_slots <= 0:
        merged = [dict(item) for item in shortlist[:top_k]]
        for index, item in enumerate(merged):
            item["selection_step"] = index + 1
            item["rank"] = index + 1
        return merged
    normal_keep = top_k
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(shortlist[:normal_keep]):
        candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
        key = candidate_to_key(candidate) if isinstance(candidate, dict) else ""
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    inserted = 0
    for item in coverage_records:
        if inserted >= coverage_slots:
            break
        candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
        key = candidate_to_key(candidate) if isinstance(candidate, dict) else ""
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
        inserted += 1
    for item in list(shortlist[normal_keep:]) + list(shortlist[:normal_keep]):
        if len(merged) >= top_k:
            break
        candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
        key = candidate_to_key(candidate) if isinstance(candidate, dict) else ""
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    for index, item in enumerate(merged):
        item["selection_step"] = index + 1
        item["rank"] = index + 1
    return merged[: top_k + coverage_slots]


def _normalized_ensemble_af_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    defaults = {"qlogei": 0.50, "qucb": 0.25, "ts": 0.25}
    if not weights:
        return defaults
    normalized = {
        key: max(float((weights or {}).get(key, 0.0) or 0.0), 0.0)
        for key in ("qlogei", "qucb", "ts")
    }
    total = sum(normalized.values())
    if total <= 0:
        return defaults
    return {key: float(value / total) for key, value in normalized.items()}


def _af_rank_score(rank: Any, top_k: int) -> float:
    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        return 0.0
    if rank_int <= 0:
        return 0.0
    total = max(int(top_k), 1)
    return max(total + 1 - rank_int, 0) / float(total)


def _ensemble_weighted_rank_score(
    record: dict[str, Any],
    *,
    top_k: int,
    weights: dict[str, float] | None = None,
) -> float:
    af_ranks = record.get("af_ranks", {}) if isinstance(record.get("af_ranks"), dict) else {}
    normalized_weights = _normalized_ensemble_af_weights(weights)
    weighted_rank = sum(
        normalized_weights[af_key] * _af_rank_score(af_ranks.get(af_key), top_k)
        for af_key in ("qlogei", "qucb", "ts")
    )
    consensus_fraction = min(max(int(record.get("af_consensus_count", 0) or 0), 0), 3) / 3.0
    return float(weighted_rank + 0.05 * consensus_fraction)


def _candidate_diversity_distance(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    keys = sorted(str(key) for key in candidate.keys())
    if not keys:
        return 0.0
    distances: list[float] = []
    for other in selected:
        differences = sum(1 for key in keys if candidate.get(key) != other.get(key))
        distances.append(float(differences) / float(len(keys)))
    return min(distances) if distances else 0.0


def _ensemble_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    af_ranks = record.get("af_ranks", {}) if isinstance(record.get("af_ranks"), dict) else {}
    missing_rank = 10**9
    return (
        -int(record.get("af_consensus_count", 0) or 0),
        int(af_ranks.get("qlogei", missing_rank)),
        int(af_ranks.get("qucb", missing_rank)),
        int(af_ranks.get("ts", missing_rank)),
        str(record.get("_candidate_key", "")),
    )


def _rank_ensemble_candidates(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    remaining = list(records)
    selected: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    diversity_weight = 0.03
    for _ in range(min(max(int(top_k), 1), len(remaining))):
        best_index = 0
        best_key: tuple[Any, ...] | None = None
        for index, record in enumerate(remaining):
            candidate = record.get("candidate", {}) if isinstance(record.get("candidate"), dict) else {}
            base_score = _ensemble_weighted_rank_score(record, top_k=top_k, weights=weights)
            diversity_bonus = diversity_weight * _candidate_diversity_distance(candidate, selected_candidates)
            score = base_score + diversity_bonus
            sort_key = (
                score,
                base_score,
                int(record.get("af_consensus_count", 0) or 0),
                -_ensemble_sort_key(record)[1],
                -_ensemble_sort_key(record)[2],
                -_ensemble_sort_key(record)[3],
                tuple(-ord(char) for char in str(record.get("_candidate_key", ""))),
            )
            if best_key is None or sort_key > best_key:
                best_key = sort_key
                best_index = index
        chosen = remaining.pop(best_index)
        chosen["_ensemble_weighted_rank_score"] = _ensemble_weighted_rank_score(
            chosen,
            top_k=top_k,
            weights=weights,
        )
        chosen["_ensemble_diversity_bonus"] = (
            best_key[0] - best_key[1]
            if best_key is not None
            else 0.0
        )
        chosen["_ensemble_reference_score"] = best_key[0] if best_key is not None else chosen["_ensemble_weighted_rank_score"]
        selected.append(chosen)
        candidate = chosen.get("candidate", {}) if isinstance(chosen.get("candidate"), dict) else {}
        selected_candidates.append(dict(candidate))
    selected.extend(sorted(remaining, key=_ensemble_sort_key))
    return selected


class EnsembleAcquisitionFlow:
    def __init__(
        self,
        top_k: int = 8,
        prefilter_multiplier: int = 10,
        hallucination_mode: str = "kriging_believer",
        ucb_beta: float | None = None,
        af_weights: dict[str, float] | None = None,
        af_strategy_source: str = "mechanical_default",
    ):
        self.top_k = max(1, int(top_k))
        self.prefilter_multiplier = max(1, int(prefilter_multiplier))
        self.hallucination_mode = str(hallucination_mode or "kriging_believer").strip().lower()
        self.ucb_beta = ucb_beta
        self.af_weights = dict(af_weights or {})
        self.af_strategy_source = str(af_strategy_source or "mechanical_default")
        self.last_prefilter_size = 0
        self.last_af_slot_targets = _ensemble_af_slot_targets(self.top_k, self.af_weights)
        self.last_af_slot_filled = {key: 0 for key in self.last_af_slot_targets}
        self.last_af_failures: list[dict[str, str]] = []
        self.last_ucb_beta: float | None = None
        self.last_ucb_sigma_multiplier: float | None = None

    def propose_candidates(
        self,
        *,
        active_model: BaseSurrogateModel,
        refit_model_factory: Callable[[], BaseSurrogateModel] | None,
        candidate_pool: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        direction: str = "maximize",
        seed: int = 0,
        iteration: int = 0,
        stagnation_length: int = 0,
        cpu_refit_model_factory: Callable[[], BaseSurrogateModel] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidate_pool:
            self.last_prefilter_size = 0
            self.last_af_slot_filled = {key: 0 for key in self.last_af_slot_targets}
            self.last_af_failures = []
            return []

        self.last_prefilter_size = int(
            min(
                len(candidate_pool),
                max(self.top_k, self.prefilter_multiplier * self.top_k),
            )
        )
        self.last_af_slot_targets = _ensemble_af_slot_targets(self.top_k, self.af_weights)
        self.last_af_slot_filled = {key: 0 for key in self.last_af_slot_targets}
        self.last_af_failures = []
        beta = float(self.ucb_beta if self.ucb_beta is not None else _adaptive_ucb_beta(iteration, stagnation_length, len(observations)))
        self.last_ucb_beta = beta
        self.last_ucb_sigma_multiplier = float(np.sqrt(max(beta, 0.0)))

        scale_context = _build_observation_scale_context(observations, direction=direction)
        af_priority = ("qlogei", "qucb", "ts")
        cpu_fallback_model: BaseSurrogateModel | None = None

        def _get_cpu_fallback_model() -> BaseSurrogateModel | None:
            nonlocal cpu_fallback_model
            if cpu_refit_model_factory is None:
                return None
            if cpu_fallback_model is not None:
                return cpu_fallback_model
            train_rows = list(scale_context.get("observations_scaled", []))
            train_candidates = [item.get("candidate", {}) for item in train_rows if item.get("candidate")]
            train_y = np.asarray(
                [float(item.get("result", 0.0) or 0.0) for item in train_rows if item.get("candidate")],
                dtype=float,
            )
            if not train_candidates:
                return None
            model = cpu_refit_model_factory()
            model.fit(train_candidates, train_y)
            cpu_fallback_model = model
            return cpu_fallback_model

        try:
            pred_mean_model, pred_std_model = active_model.predict(candidate_pool)
            pred_mean = np.asarray(pred_mean_model, dtype=float)
            pred_std = np.maximum(np.asarray(pred_std_model, dtype=float), 1e-6)
        except Exception as exc:
            try:
                fallback_model = _get_cpu_fallback_model()
                if fallback_model is None:
                    raise RuntimeError("CPU fallback model is unavailable")
                pred_mean_model, pred_std_model = fallback_model.predict(candidate_pool)
                pred_mean = np.asarray(pred_mean_model, dtype=float)
                pred_std = np.maximum(np.asarray(pred_std_model, dtype=float), 1e-6)
                self.last_af_failures.append(
                    {
                        "af_key": "base_prediction",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recovered_by": "cpu_fallback",
                    }
                )
            except Exception as fallback_exc:
                self.last_af_failures = [
                    {"af_key": af_key, "error_type": type(exc).__name__, "error": str(exc)}
                    for af_key in af_priority
                ]
                raise RuntimeError(
                    "All ensemble acquisition functions failed: "
                    + "; ".join(
                        f"{item['af_key']}={item['error_type']}: {item['error']}"
                        for item in self.last_af_failures
                    )
                ) from fallback_exc

        if direction == "minimize":
            pred_mean = -1.0 * pred_mean
        base_scores = {
            "candidate_pool": [dict(candidate) for candidate in candidate_pool],
            "pred_mean": pred_mean,
            "pred_std": pred_std,
        }
        base_lookup = {
            candidate_to_key(candidate): index
            for index, candidate in enumerate(base_scores["candidate_pool"])
        }

        af_ranked: dict[str, list[dict[str, Any]]] = {}
        for af_offset, af_key in enumerate(af_priority):
            try:
                af_ranked[af_key] = _build_ranked_af_candidates(
                    af_key=af_key,
                    active_model=active_model,
                    refit_model_factory=refit_model_factory,
                    candidate_pool=candidate_pool,
                    scale_context=scale_context,
                    top_k=min(self.top_k, len(candidate_pool)),
                    prefilter_multiplier=self.prefilter_multiplier,
                    hallucination_mode=self.hallucination_mode,
                    seed=seed + (af_offset + 1) * 997,
                    ucb_beta=beta,
                )
                if not af_ranked[af_key]:
                    raise RuntimeError("acquisition function produced no ranked candidates")
            except Exception as exc:
                try:
                    fallback_model = _get_cpu_fallback_model()
                    if fallback_model is None:
                        raise RuntimeError("CPU fallback model is unavailable")
                    af_ranked[af_key] = _build_ranked_af_candidates(
                        af_key=af_key,
                        active_model=fallback_model,
                        refit_model_factory=cpu_refit_model_factory,
                        candidate_pool=candidate_pool,
                        scale_context=scale_context,
                        top_k=min(self.top_k, len(candidate_pool)),
                        prefilter_multiplier=self.prefilter_multiplier,
                        hallucination_mode=self.hallucination_mode,
                        seed=seed + (af_offset + 1) * 997,
                        ucb_beta=beta,
                    )
                    if not af_ranked[af_key]:
                        raise RuntimeError("CPU fallback produced no ranked candidates")
                    self.last_af_failures.append(
                        {
                            "af_key": af_key,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "recovered_by": "cpu_fallback",
                        }
                    )
                except Exception:
                    af_ranked[af_key] = []
                    self.last_af_failures.append(
                        {
                            "af_key": af_key,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

        successful_afs = [af_key for af_key in af_priority if af_ranked.get(af_key)]
        if not successful_afs:
            raise RuntimeError(
                "All ensemble acquisition functions failed: "
                + "; ".join(
                    f"{item['af_key']}={item['error_type']}: {item['error']}"
                    for item in self.last_af_failures
                )
            )

        merged: dict[str, dict[str, Any]] = {}

        def _merge_entry(entry: dict[str, Any]) -> None:
            candidate = dict(entry.get("candidate", {}))
            key = candidate_to_key(candidate)
            base_index = base_lookup.get(key)
            if base_index is None:
                return
            record = merged.get(key)
            if record is None:
                record = {
                    "candidate": candidate,
                    "predicted_value": float(base_scores["pred_mean"][base_index]),
                    "uncertainty": float(base_scores["pred_std"][base_index]),
                    "acquisition_value": None,
                    "acquisition_value_raw": None,
                    "selection_step": 0,
                    "selection_mode": "ensemble_candidate",
                    "rank": 0,
                    "af_sources": [],
                    "af_ranks": {},
                    "af_consensus_count": 0,
                    "ensemble_reference_score": None,
                    "_candidate_key": key,
                }
                merged[key] = record
            af_key = str(entry.get("af_key") or "")
            af_rank = int(entry.get("af_rank") or 0)
            if af_key and af_key not in record["af_ranks"]:
                record["af_sources"].append(af_key)
                record["af_ranks"][af_key] = af_rank
                record["af_consensus_count"] = len(record["af_sources"])

        desired_size = min(self.top_k, len(candidate_pool))
        for af_key in af_priority:
            target = int(self.last_af_slot_targets.get(af_key, 0) or 0)
            primary_entries = list(af_ranked.get(af_key, []))[:target]
            self.last_af_slot_filled[af_key] = len(primary_entries)
            for entry in primary_entries:
                _merge_entry(entry)

        if len(merged) < desired_size:
            for af_key in af_priority:
                overflow_entries = list(af_ranked.get(af_key, []))[int(self.last_af_slot_targets.get(af_key, 0) or 0) :]
                for entry in overflow_entries:
                    _merge_entry(entry)
                    if len(merged) >= desired_size:
                        break
                if len(merged) >= desired_size:
                    break

        if len(merged) < desired_size:
            raise RuntimeError(
                f"Ensemble acquisition functions returned only {len(merged)} unique candidates; "
                f"required {desired_size} from successful AFs: {', '.join(successful_afs)}"
            )

        combined = _rank_ensemble_candidates(
            list(merged.values()),
            top_k=desired_size,
            weights=self.af_weights,
        )
        final_shortlist = combined[:desired_size]
        for index, item in enumerate(final_shortlist):
            item["ensemble_reference_score"] = float(
                item.get("_ensemble_reference_score")
                if item.get("_ensemble_reference_score") is not None
                else _ensemble_weighted_rank_score(item, top_k=self.top_k, weights=self.af_weights)
            )
            item["ensemble_weighted_rank_score"] = float(
                item.get("_ensemble_weighted_rank_score")
                if item.get("_ensemble_weighted_rank_score") is not None
                else _ensemble_weighted_rank_score(item, top_k=self.top_k, weights=self.af_weights)
            )
            item["ensemble_diversity_bonus"] = float(item.get("_ensemble_diversity_bonus", 0.0) or 0.0)
            item["selection_step"] = index + 1
            item["rank"] = index + 1
            if index == 0:
                item["selection_mode"] = "ensemble_reference"
            item.pop("_candidate_key", None)
            item.pop("_ensemble_reference_score", None)
            item.pop("_ensemble_weighted_rank_score", None)
            item.pop("_ensemble_diversity_bonus", None)
        self.last_af_slot_filled = {
            af_key: sum(1 for item in final_shortlist if af_key in list(item.get("af_sources", [])))
            for af_key in af_priority
        }
        return final_shortlist


def _autobo_stagnation_length(performance_log: list[dict[str, Any]]) -> int:
    count = 0
    for item in reversed(list(performance_log or [])):
        if bool(item.get("improved", False)):
            break
        count += 1
    return count


def _default_af_strategy(settings, *, source: str = "mechanical_default", reason: str = "") -> dict[str, Any]:
    beta = getattr(settings, "autobo_ucb_beta", None)
    return {
        "weights": {"qlogei": 0.50, "qucb": 0.25, "ts": 0.25},
        "qucb_beta": beta,
        "reasoning": reason or "Using the mechanical default ensemble allocation.",
        "confidence": 0.5,
        "source": source,
        "valid": True,
    }


def _validate_af_strategy_payload(payload: dict[str, Any], settings, *, source: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw_weights = payload.get("weights", {})
    if not isinstance(raw_weights, dict):
        return None
    weights = {}
    for key in ("qlogei", "qucb", "ts"):
        value = _coerce_float(raw_weights.get(key), default=float("nan"))
        if not np.isfinite(value) or value < 0.0:
            return None
        weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0.0 or abs(total - 1.0) > 0.05:
        return None
    weights = {key: value / total for key, value in weights.items()}
    qlogei_floor = min(max(float(getattr(settings, "autobo_af_qlogei_min_weight", 0.20) or 0.20), 0.0), 1.0)
    if weights["qlogei"] < qlogei_floor:
        deficit = qlogei_floor - weights["qlogei"]
        weights["qlogei"] = qlogei_floor
        donor_total = weights["qucb"] + weights["ts"]
        if donor_total > 0:
            weights["qucb"] = max(0.0, weights["qucb"] - deficit * weights["qucb"] / donor_total)
            weights["ts"] = max(0.0, weights["ts"] - deficit * weights["ts"] / donor_total)
        renorm = sum(weights.values()) or 1.0
        weights = {key: value / renorm for key, value in weights.items()}

    beta = payload.get("qucb_beta")
    if beta is None:
        beta_value = getattr(settings, "autobo_ucb_beta", None)
    else:
        beta_value = _coerce_float(beta, default=float("nan"))
        if not np.isfinite(beta_value):
            return None
        beta_value = min(max(float(beta_value), 0.1), 5.0)
    return {
        "weights": {key: round(float(value), 6) for key, value in weights.items()},
        "qucb_beta": beta_value,
        "reasoning": str(payload.get("reasoning") or "").strip(),
        "confidence": min(max(_coerce_float(payload.get("confidence"), default=0.6), 0.0), 1.0),
        "source": source,
        "valid": True,
    }


def _memoryless_reaction_context(state: dict[str, Any]) -> dict[str, Any]:
    problem_spec = state.get("problem_spec", {}) if isinstance(state.get("problem_spec"), dict) else {}
    return {
        "application_domain": str(problem_spec.get("application_domain") or ""),
        "domain_profile": str(problem_spec.get("domain_profile") or ""),
        "reaction_type": str(problem_spec.get("reaction_type") or ""),
        "target_metric": str(problem_spec.get("target_metric") or "yield"),
        "optimization_direction": str(state.get("optimization_direction", "maximize")).strip().lower(),
    }


def _memoryless_autobo_surrogate_eval_context(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "reaction_context": _memoryless_reaction_context(state),
        "knowledge_cards_text": "",
        "memory_rules": [],
        "active_model": (state.get("autobo_state", {}) or {}).get("active_model", ""),
    }


def _memoryless_autobo_acquisition_select_context(state: dict[str, Any]) -> dict[str, Any]:
    shortlist = list(state.get("proposal_shortlist", []))
    return {
        "reaction_context": _memoryless_reaction_context(state),
        "top_observations": [],
        "bottom_observations": [],
        "knowledge_cards_text": "",
        "knowledge_cards": [],
        "knowledge_mode": "memoryless",
        "memory_rules": [],
        "active_hypotheses": [],
        "total_observations": 0,
        "shortlist": shortlist,
        "recent_override_outcomes": [],
        "memory_packet": {},
    }


def _af_strategy_outcome_digest(state: dict[str, Any]) -> dict[str, Any]:
    observations = [item for item in state.get("observations", []) if item.get("result") is not None]
    recent = observations[-5:]
    overrides = []
    for item in recent:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        rank = metadata.get("autobo_rank") or metadata.get("autobo_shortlist_rank")
        if rank and int(rank) != 1:
            overrides.append(
                {
                    "iteration": item.get("iteration"),
                    "rank": rank,
                    "result": item.get("result"),
                    "best_before": metadata.get("best_before_result"),
                }
            )
    return {
        "recent_results": [item.get("result") for item in recent],
        "recent_improved": [bool((item.get("metadata") or {}).get("improved", False)) for item in recent],
        "recent_overrides": overrides,
    }


def _should_refresh_af_strategy(
    *,
    state: dict[str, Any],
    settings,
    autobo_state: dict[str, Any],
    stagnation_length: int,
    switch_info: dict[str, Any],
) -> tuple[bool, str]:
    current_iter = int(state.get("iteration", 0) or 0)
    current = autobo_state.get("af_strategy", {}) if isinstance(autobo_state.get("af_strategy"), dict) else {}
    if not current:
        return True, "initial_bo_strategy"
    switch_type = str(switch_info.get("switch_type") or "")
    if switch_type in {"deliberate", "active_failed"} and int(current.get("last_switch_iteration", -1) or -1) != current_iter:
        return True, f"switch_type={switch_type}"
    if int(stagnation_length) >= 3:
        return True, f"stagnation_length={stagnation_length}"
    last_decision = int(current.get("last_decision_iteration", -10**9) or -10**9)
    interval = max(1, int(getattr(settings, "autobo_af_strategy_min_interval", 8) or 8))
    if current_iter - last_decision >= interval:
        return True, f"interval={interval}"
    return False, "reuse_cached"


def _resolve_af_strategy(
    *,
    state: dict[str, Any],
    settings,
    llm,
    invoke_json_node,
    autobo_state: dict[str, Any],
    stagnation_length: int,
    switch_info: dict[str, Any],
    zero_llm_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = autobo_state.get("af_strategy", {}) if isinstance(autobo_state.get("af_strategy"), dict) else {}
    if (
        llm is None
        or zero_llm_mode
        or not bool(getattr(settings, "autobo_af_strategy_enabled", True))
    ):
        strategy = cached if cached.get("valid") else _default_af_strategy(settings, source="mechanical_disabled")
        return strategy, _empty_usage_delta()
    should_refresh, refresh_reason = _should_refresh_af_strategy(
        state=state,
        settings=settings,
        autobo_state=autobo_state,
        stagnation_length=stagnation_length,
        switch_info=switch_info,
    )
    if not _memory_enabled(settings):
        should_refresh = True
        refresh_reason = "memory_ablation_stateless"
    if not should_refresh and cached.get("valid"):
        return {**cached, "source": cached.get("source", "llm_cached")}, _empty_usage_delta()

    memory_manager = MemoryManager.from_dict(state.get("memory", {}))
    context = (
        _memoryless_autobo_surrogate_eval_context(state)
        if not _memory_enabled(settings)
        else ContextBuilder.for_autobo_surrogate_eval(state, memory_manager)
    )
    strategy_context = {
        "iteration": 0 if not _memory_enabled(settings) else int(state.get("iteration", 0) or 0),
        "best_result": None if not _memory_enabled(settings) else state.get("best_result"),
        "active_model": autobo_state.get("active_model"),
        "stagnation_length": 0 if not _memory_enabled(settings) else int(stagnation_length),
        "switch_info": {} if not _memory_enabled(settings) else switch_info,
        "refresh_reason": refresh_reason,
        "previous_strategy": {} if not _memory_enabled(settings) else cached,
        "recent_outcome_digest": {} if not _memory_enabled(settings) else _af_strategy_outcome_digest(state),
    }
    prompt = build_af_strategy_prompt(
        reaction_context=context.get("reaction_context", {}),
        strategy_context=strategy_context,
        knowledge_cards_text=context.get("knowledge_cards_text", ""),
        memory_rules=context.get("memory_rules", []),
    )
    default = _default_af_strategy(settings, source="mechanical_default", reason="AF strategy LLM response unavailable.")
    parsed, _, usage = invoke_json_node(
        llm,
        state,
        prompt,
        default,
        node_name="run_bo_iteration",
        lightweight=True,
    )
    strategy = _validate_af_strategy_payload(parsed, settings, source="llm_directed")
    if strategy is None:
        fallback = _default_af_strategy(settings, source="mechanical_fallback", reason="Invalid LLM AF strategy; using defaults.")
        fallback["last_decision_iteration"] = int(state.get("iteration", 0) or 0)
        fallback["refresh_reason"] = refresh_reason
        return fallback, usage
    strategy["last_decision_iteration"] = int(state.get("iteration", 0) or 0)
    strategy["last_switch_iteration"] = int(state.get("iteration", 0) or 0) if switch_info.get("switch_type") in {"deliberate", "active_failed"} else int(cached.get("last_switch_iteration", -1) or -1)
    strategy["refresh_reason"] = refresh_reason
    return strategy, usage


class ReverseCalibrator:
    def __init__(self, window_size: int = 15, degrade_threshold: float = 0.0):
        self.window_size = int(window_size)
        self.degrade_threshold = float(degrade_threshold)
        self.plaus_records: list[dict[str, Any]] = []

    def record_plausibility_eval(
        self,
        model_id: str,
        point_candidate: dict[str, Any],
        llm_score: float,
        observed_y: float | None = None,
        predicted_mu: float | None = None,
    ) -> None:
        self.plaus_records.append(
            {
                "model_id": model_id,
                "candidate_key": candidate_to_key(point_candidate or {}),
                "llm_score": float(llm_score),
                "observed_y": observed_y,
                "predicted_mu": predicted_mu,
            }
        )

    def should_degrade_llm_weight(self) -> tuple[bool, float, str]:
        recent_plaus = [
            item
            for item in self.plaus_records[-self.window_size :]
            if item.get("observed_y") is not None and item.get("predicted_mu") is not None
        ]
        if len(recent_plaus) >= 8:
            llm_scores = np.asarray([float(item.get("llm_score", 0.0)) for item in recent_plaus], dtype=float)
            actual_errors = np.asarray(
                [abs(float(item.get("observed_y", 0.0)) - float(item.get("predicted_mu", 0.0))) for item in recent_plaus],
                dtype=float,
            )
            rho = _safe_spearman(llm_scores, -1.0 * actual_errors)
            if np.isfinite(rho) and rho < self.degrade_threshold:
                return True, 0.05, f"LLM plausibility correlation={rho:.3f} not positive"

        return False, 0.10, "LLM signal appears calibrated"

    def to_dict(self) -> dict[str, Any]:
        return {"plaus_records": self.plaus_records[-50:]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReverseCalibrator":
        instance = cls()
        instance.plaus_records = list(data.get("plaus_records", []))
        return instance


def _analytic_ei(mu: np.ndarray, sigma: np.ndarray, best_f: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    z = (mu - float(best_f)) / sigma
    try:
        from scipy.stats import norm

        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
    except Exception:
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
    ei = sigma * (z * cdf + pdf)
    return np.maximum(ei, 0.0)


def _observations_to_candidates(
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    candidates = [item.get("candidate", {}) for item in observations if item.get("result") is not None]
    y_values = np.asarray([float(item["result"]) for item in observations if item.get("result") is not None], dtype=float)
    return candidates, y_values


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) <= 1 or len(right) <= 1:
        return 0.0
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(left, right)
        return float(rho) if np.isfinite(rho) else 0.0
    except Exception:
        return 0.0


def _z_score_for_ci(ci_level: float) -> float:
    bounded = min(max(float(ci_level), 1e-3), 0.999)
    try:
        from scipy.stats import norm

        return float(norm.ppf(0.5 + bounded / 2.0))
    except Exception:
        return 1.96


def _resolve_autobo_state(autobo_state: dict[str, Any] | None, settings) -> dict[str, Any]:
    current = dict(autobo_state or {})
    return {
        "robust_nll": dict(current.get("robust_nll", {})),
        "active_model": str(current.get("active_model") or getattr(settings, "autobo_initial_active", "gp_indicator_matern52")),
        "fitness_log": dict(current.get("fitness_log", {})),
        "calibration_log": list(current.get("calibration_log", [])),
        "switch_history": list(current.get("switch_history", [])),
        "last_eval_n": int(current.get("last_eval_n", -1)),
        "last_layer2_iteration": int(current.get("last_layer2_iteration", 0)),
        "hysteresis_until": 0,
        "llm_plaus_audit": list(current.get("llm_plaus_audit", [])),
        "effective_llm_weight": 0.0,
        "deep_ensemble_feature_spec": current.get("deep_ensemble_feature_spec"),
        "coverage_history": {
            str(key): [
                float(item)
                for item in value
                if isinstance(item, (int, float)) and np.isfinite(float(item))
            ][-20:]
            for key, value in dict(current.get("coverage_history") or {}).items()
            if isinstance(value, list)
        },
        "last_loocv_fold_hits": {
            str(key): [bool(item) for item in value]
            for key, value in dict(current.get("last_loocv_fold_hits") or {}).items()
            if isinstance(value, list)
        },
        "challenger_lead_streak": {},
        "af_strategy": dict(current.get("af_strategy", {})) if isinstance(current.get("af_strategy"), dict) else {},
    }


def _pure_reasoning_resolved_components() -> dict[str, Any]:
    return {
        "surrogate_model": "pure_reasoning_llm",
        "kernel_config": {
            "key": "none",
            "params": {},
            "rationale": "Pure reasoning ablation disables surrogate kernels and BO scoring.",
        },
        "acquisition_function": "llm_direct_select",
    }


def _pure_reasoning_bo_config(state: dict[str, Any]) -> dict[str, Any]:
    config_version = len(state.get("config_history", [])) + 1
    return {
        "surrogate_model": "pure_reasoning_llm",
        "surrogate_params": {},
        "kernel_config": {
            "key": "none",
            "params": {},
            "rationale": "Pure reasoning ablation does not instantiate a BO kernel.",
        },
        "acquisition_function": "llm_direct_select",
        "af_params": {},
        "rationale": "Pure reasoning ablation: do not use AutoBO or any BO scoring; the LLM selects the next experiment directly from a legal candidate pool.",
        "confidence": 1.0,
        "config_version": config_version,
        "validated": True,
        "selection_source": "pure_reasoning_llm",
        "selection_diagnostics": {},
        "autobo_active_model": None,
        "resolved_components": _pure_reasoning_resolved_components(),
        "proposal_strategy": "pure_reasoning_ablation",
    }


def _pure_reasoning_effective_config(state: dict[str, Any]) -> dict[str, Any]:
    effective_config = dict(state.get("effective_config", {}))
    effective_config.update(
        {
            "runtime_mode": detect_runtime_capabilities()["runtime_mode"],
            "proposal_strategy": "pure_reasoning_ablation",
            "resolved_components": _pure_reasoning_resolved_components(),
            "surrogate_model": "pure_reasoning_llm",
            "kernel_config": {"key": "none", "params": {}},
            "acquisition_function": "llm_direct_select",
            "selection_source": "pure_reasoning_llm",
            "autobo_active_model": None,
            "selection_diagnostics": {},
        }
    )
    return effective_config


def _effective_config_with_components(
    state: dict[str, Any],
    *,
    active_model_id: str,
    resolved_components: dict[str, Any],
    switch_info: dict[str, Any],
    trigger_reason: str,
    acquisition_function: str,
    switch_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_config = dict(state.get("effective_config", {}))
    effective_config.update(
        {
            "resolved_components": resolved_components,
            "surrogate_model": resolved_components.get("surrogate_model"),
            "kernel_config": resolved_components.get("kernel_config"),
            "acquisition_function": acquisition_function,
            "autobo_active_model": active_model_id,
            "selection_source": "autobo",
            "selection_diagnostics": {
                "switch_info": switch_info,
                "trigger_reason": trigger_reason,
                "switch_decision": dict(switch_decision or {}),
            },
        }
    )
    return effective_config


def _bo_config_with_active_model(bo_config: dict[str, Any], active_model_id: str, acquisition_function: str) -> dict[str, Any]:
    next_config = dict(bo_config or {})
    if str(active_model_id) == "ensemble_sur":
        resolved_components = {
            "surrogate_model": "ensemble_sur",
            "kernel_config": {"key": "multi_surrogate", "params": {}, "categorical_kernel": "per_surrogate", "continuous_kernel": "per_surrogate"},
            "acquisition_function": acquisition_function,
        }
    else:
        resolved_components = resolve_recorded_surrogate_components(
            active_model_id,
            acquisition_function=acquisition_function,
        )
    next_config["surrogate_model"] = resolved_components.get("surrogate_model")
    next_config["kernel_config"] = resolved_components.get("kernel_config")
    next_config["autobo_active_model"] = active_model_id
    next_config["acquisition_function"] = acquisition_function
    return next_config


def _empty_usage_delta() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_calls": 0,
        "estimated": False,
    }


def _accumulate_usage_delta(base: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or _empty_usage_delta())
    incoming = dict(addition or _empty_usage_delta())
    for key in ("calls", "input_tokens", "output_tokens", "total_tokens", "estimated_calls"):
        merged[key] = int(merged.get(key, 0) or 0) + int(incoming.get(key, 0) or 0)
    merged["estimated"] = bool(merged.get("estimated_calls", 0))
    return merged


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if np.isfinite(value) else default
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _coerce_selected_id(value: Any, default: int) -> int:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        direct = re.search(r"(?:candidate|cand|choice|select(?:ed)?|#)\s*#?\s*(\d+)", text, re.IGNORECASE)
        if direct:
            return _coerce_int(direct.group(1), default=default)
        leading_hash = re.fullmatch(r"#\s*(\d+)", text)
        if leading_hash:
            return _coerce_int(leading_hash.group(1), default=default)
    return _coerce_int(value, default=default)


def _find_shortlist_index(
    shortlist: list[dict[str, Any]],
    selected_record: dict[str, Any],
    *,
    default: int = 0,
) -> int:
    candidate = selected_record.get("candidate", {}) if isinstance(selected_record, dict) else {}
    if isinstance(candidate, dict) and candidate:
        selected_key = candidate_to_key(candidate)
        for index, item in enumerate(shortlist):
            item_candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
            if isinstance(item_candidate, dict) and candidate_to_key(item_candidate) == selected_key:
                return index
    return int(default)


def _validate_override_evidence(
    raw_evidence: Any,
    *,
    state: dict[str, Any],
    memory_manager: MemoryManager,
    reasoning: str = "",
) -> tuple[dict[str, Any], str]:
    if not isinstance(raw_evidence, dict):
        return {
            "evidence_type": "none",
            "evidence_ids": [],
            "trajectory_references": [],
            "chemistry_argument": "",
            "validated": False,
            "warning": "missing override_evidence",
        }, "missing override_evidence"
    evidence_ids = _evidence_values(
        raw_evidence,
        "evidence_ids",
        "evidence_id",
        "ids",
        "id",
        "rule_ids",
        "rule_id",
        "memory_rules",
        "memory_rule",
        "card_ids",
        "card_id",
        "knowledge_cards",
        "knowledge_card",
    )
    trajectory_references = _evidence_values(
        raw_evidence,
        "trajectory_references",
        "trajectory_reference",
        "trajectory_refs",
        "trajectory_ref",
        "trajectory_iterations",
        "trajectory_iteration",
        "iterations",
        "iteration",
        "iters",
        "iter",
    )
    chemistry_argument = str(
        raw_evidence.get("chemistry_argument")
        or raw_evidence.get("domain_argument")
        or ""
    ).strip()
    evidence_type = _normalize_evidence_type(str(raw_evidence.get("evidence_type") or "").strip())
    if not evidence_type:
        if trajectory_references:
            evidence_type = "trajectory"
        elif any(key in raw_evidence for key in ("rule_ids", "rule_id", "memory_rules", "memory_rule")):
            evidence_type = "memory_rule"
        elif any(key in raw_evidence for key in ("card_ids", "card_id", "knowledge_cards", "knowledge_card")):
            evidence_type = "knowledge_card"
        elif chemistry_argument:
            evidence_type = "domain"
    normalized = {
        "evidence_type": evidence_type,
        "evidence_ids": evidence_ids,
        "trajectory_references": trajectory_references,
        "chemistry_argument": chemistry_argument,
        "domain_argument": chemistry_argument,
        "validated": False,
    }

    active_cards = [
        card
        for card in ((state.get("knowledge_deck", {}) or {}).get("cards", []) if isinstance(state.get("knowledge_deck", {}), dict) else [])
        if isinstance(card, dict) and str(card.get("status") or "active") in {"active", "validated"}
    ]
    graph_nodes = getattr(getattr(memory_manager, "semantic_graph", None), "nodes", {})
    active_rules = list(graph_nodes.values()) if isinstance(graph_nodes, dict) else []
    valid_rule_ids = {
        str(getattr(node, "id", "") or (node.get("id") if isinstance(node, dict) else "")).strip()
        for node in active_rules
        if _semantic_node_status(node) in {"active", "tentative", "validated"}
    }
    if evidence_type == "knowledge_card":
        matched_ids = _match_evidence_ids_to_card_ids(evidence_ids, active_cards)
        if matched_ids:
            normalized["evidence_ids"] = matched_ids
            normalized["validated"] = True
            return normalized, ""
        warning = "knowledge_card evidence did not cite an active card_id"
        normalized["warning"] = warning
        return normalized, warning
    if evidence_type == "memory_rule":
        matched_ids = _match_evidence_ids_to_rule_ids(evidence_ids, active_rules)
        exact_rule_ids = _string_evidence_id_set(evidence_ids) & valid_rule_ids
        if matched_ids or exact_rule_ids:
            normalized["evidence_ids"] = matched_ids or sorted(exact_rule_ids)
            normalized["validated"] = True
            return normalized, ""
        warning = "memory_rule evidence did not cite an active rule id"
        normalized["warning"] = warning
        return normalized, warning
    if evidence_type == "trajectory":
        observed_iters: set[int] = set()
        for item in state.get("observations", []):
            if not isinstance(item, dict) or item.get("iteration") is None:
                continue
            try:
                observed_iters.add(int(item.get("iteration")))
            except (TypeError, ValueError):
                continue
        cited_iters: set[int] = set()
        for ref in trajectory_references:
            parsed_iter = _parse_iteration_reference(ref)
            if parsed_iter is not None:
                cited_iters.add(parsed_iter)
        if not cited_iters:
            for ref in evidence_ids:
                parsed_iter = _parse_iteration_reference(ref)
                if parsed_iter is not None:
                    cited_iters.add(parsed_iter)
        normalized["trajectory_references"] = sorted(cited_iters)
        if cited_iters & observed_iters:
            normalized["validated"] = True
            return normalized, ""
        warning = "trajectory evidence did not cite an observed iteration"
        normalized["warning"] = warning
        return normalized, warning
    if evidence_type in {"chemistry", "domain"}:
        argument = chemistry_argument or str(reasoning or "").strip()
        generic = argument.lower().strip(" .")
        generic_phrases = {"more exploration", "better exploration", "exploration", "more exploratory"}
        if len(argument) >= 40 and generic not in generic_phrases:
            normalized["chemistry_argument"] = argument
            normalized["domain_argument"] = argument
            normalized["validated"] = True
            return normalized, ""
        warning = f"{evidence_type} evidence was too generic"
        normalized["warning"] = warning
        return normalized, warning
    warning = "override_evidence evidence_type must be knowledge_card, memory_rule, trajectory, chemistry, or domain"
    normalized["warning"] = warning
    return normalized, warning


def _normalize_evidence_type(value: str) -> str:
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "card": "knowledge_card",
        "knowledge": "knowledge_card",
        "knowledge_cards": "knowledge_card",
        "knowledge_card": "knowledge_card",
        "memory": "memory_rule",
        "rule": "memory_rule",
        "rules": "memory_rule",
        "memory_rules": "memory_rule",
        "memory_rule": "memory_rule",
        "trajectory_reference": "trajectory",
        "trajectory_references": "trajectory",
        "trajectory_ref": "trajectory",
        "iteration": "trajectory",
        "observed_iteration": "trajectory",
        "chemistry_argument": "chemistry",
        "chemical": "chemistry",
        "chemistry": "chemistry",
        "domain": "domain",
        "domain_argument": "domain",
        "hpo": "domain",
        "ml": "domain",
    }
    return aliases.get(text, text)


def _evidence_values(payload: dict[str, Any], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        if key not in payload:
            continue
        raw = payload.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif isinstance(raw, tuple):
            values.extend(list(raw))
        elif raw not in (None, "", {}, []):
            values.append(raw)
    normalized: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        normalized.append(value)
        seen.add(key)
    return normalized


def _parse_iteration_reference(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("iteration", "iter", "iteration_id", "step"):
            if key in value:
                parsed = _parse_iteration_reference(value.get(key))
                if parsed is not None:
                    return parsed
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if np.isfinite(value) and float(value).is_integer() else None
    text = str(value or "").strip()
    if not text:
        return None
    direct = re.fullmatch(r"\d+", text)
    if direct:
        return int(text)
    match = re.search(r"\b(?:iter|iteration|obs|observation)\s*#?\s*(\d+)\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _semantic_node_status(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("status") or "active").strip()
    return str(getattr(node, "status", "active") or "active").strip()


def _semantic_node_id(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("id") or "").strip()
    return str(getattr(node, "id", "") or "").strip()


def _semantic_node_statement(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("statement") or node.get("rule") or "").strip()
    return str(getattr(node, "statement", "") or "").strip()


def _normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _string_evidence_id_set(evidence_ids: list[Any]) -> set[str]:
    return {str(item).strip() for item in evidence_ids if str(item).strip()}


def _match_evidence_ids_to_rule_ids(evidence_ids: list[Any], active_rules: list[Any]) -> list[str]:
    matches: list[str] = []
    for evidence in evidence_ids:
        text = str(evidence or "").strip()
        if not text:
            continue
        normalized_text = _normalize_match_text(text)
        for node in active_rules:
            rule_id = _semantic_node_id(node)
            if not rule_id or rule_id in matches:
                continue
            statement = _semantic_node_statement(node)
            normalized_statement = _normalize_match_text(statement)
            if text == rule_id or text.upper() == rule_id.upper():
                matches.append(rule_id)
            elif normalized_text and normalized_statement and (
                normalized_text == normalized_statement
                or normalized_text in normalized_statement
                or normalized_statement in normalized_text
            ):
                matches.append(rule_id)
    return matches


def _match_evidence_ids_to_card_ids(evidence_ids: list[Any], active_cards: list[dict[str, Any]]) -> list[str]:
    matches: list[str] = []
    for evidence in evidence_ids:
        text = str(evidence or "").strip()
        if not text:
            continue
        normalized_text = _normalize_match_text(text)
        for card in active_cards:
            card_id = str(card.get("card_id") or "").strip()
            if not card_id or card_id in matches:
                continue
            card_text = str(card.get("text") or "").strip()
            normalized_card_text = _normalize_match_text(card_text)
            if text == card_id or text.lower() == card_id.lower():
                matches.append(card_id)
            elif normalized_text and normalized_card_text and (
                normalized_text == normalized_card_text
                or normalized_text in normalized_card_text
                or normalized_card_text in normalized_text
            ):
                matches.append(card_id)
    return matches


def _first_dataset_backed_shortlist_record(
    shortlist: list[dict[str, Any]],
    oracle: DatasetOracle,
    preferred_index: int = 0,
) -> tuple[int, dict[str, Any]] | None:
    preferred = shortlist[preferred_index:] + shortlist[:preferred_index]
    preferred_indices = list(range(preferred_index, len(shortlist))) + list(range(0, preferred_index))
    for index, item in zip(preferred_indices, preferred):
        candidate = item.get("candidate", {})
        if not isinstance(candidate, dict):
            continue
        if oracle.candidate_exists(candidate):
            normalized = dict(item)
            normalized["candidate"] = oracle.lookup(candidate)["candidate"]
            return index, normalized
    return None


def _run_llm_plausibility_eval(
    *,
    state: dict[str, Any],
    pool: SurrogatePool,
    observations: list[dict[str, Any]],
    fitted_ids: list[str],
    active_model_id: str,
    llm,
    settings,
    invoke_json_node,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    if len(fitted_ids) < 2:
        return {}, [], _empty_usage_delta()

    variables = state.get("problem_spec", {}).get("variables", [])
    observed_keys = {
        candidate_to_key(item.get("candidate", {}))
        for item in observations
        if item.get("candidate")
    }
    dataset_candidate_pool = dataset_candidate_pool_from_spec(state.get("problem_spec", {}).get("dataset", {}))
    candidate_pool = build_bo_candidate_pool(
        variables,
        observed_keys=observed_keys,
        candidate_pool_size=max(128, int(getattr(settings, "autobo_eval_points", 10) or 10) * 32),
        seed=_state_seed(state),
        hard_constraints=[],
        candidate_pool=dataset_candidate_pool,
    )
    if not candidate_pool:
        return {}, [], _empty_usage_delta()

    all_predictions = pool.predict_all(candidate_pool)
    if len(all_predictions) < 2:
        return {}, [], _empty_usage_delta()

    active_model = pool.get_active_model(active_model_id)
    top_acquisition_keys: set[str] = set()
    if active_model is not None:
        active_spec = pool.specs.get(active_model_id)
        refit_model_factory = None
        if active_spec is not None:
            primary_torch_device = _primary_torch_device(settings)
            refit_model_factory = lambda spec=active_spec, ss=variables, fs=pool.feature_spec, td=primary_torch_device: _create_surrogate_from_spec(spec, ss, fs, torch_device=td)
        prefilter_multiplier = int(getattr(settings, "autobo_shortlist_prefilter_multiplier", 10) or 10)
        hallucination_mode = str(getattr(settings, "autobo_shortlist_hallucination_mode", "kriging_believer"))
        if bool(getattr(settings, "ensemble_af", True)):
            acquisition_shortlist = EnsembleAcquisitionFlow(
                top_k=5,
                prefilter_multiplier=prefilter_multiplier,
                hallucination_mode=hallucination_mode,
                ucb_beta=getattr(settings, "autobo_ucb_beta", None),
            ).propose_candidates(
                active_model=active_model,
                refit_model_factory=refit_model_factory,
                candidate_pool=candidate_pool,
                observations=observations,
                direction=state.get("optimization_direction", "maximize"),
                seed=_state_seed(state),
                iteration=int(state.get("iteration", 0)),
                stagnation_length=_autobo_stagnation_length(state.get("performance_log", [])),
            )
        else:
            acquisition_shortlist = AcquisitionFlow(
                top_k=5,
                prefilter_multiplier=prefilter_multiplier,
                hallucination_mode=hallucination_mode,
            ).propose_candidates(
                active_model=active_model,
                refit_model_factory=refit_model_factory,
                candidate_pool=candidate_pool,
                observations=observations,
                direction=state.get("optimization_direction", "maximize"),
                seed=_state_seed(state),
            )
        top_acquisition_keys = {
            candidate_to_key(item.get("candidate", {}))
            for item in acquisition_shortlist
            if item.get("candidate")
        }

    model_means = np.stack([all_predictions[model_id][0] for model_id in all_predictions], axis=0)
    disagreement = np.max(model_means, axis=0) - np.min(model_means, axis=0)
    disagree_indices = np.argsort(disagreement)[::-1][:5]
    eval_indices = list(disagree_indices.astype(int))
    for index, candidate in enumerate(candidate_pool):
        if candidate_to_key(candidate) in top_acquisition_keys and index not in eval_indices:
            eval_indices.append(index)
        if len(eval_indices) >= int(getattr(settings, "autobo_eval_points", 10) or 10):
            break

    direction = state.get("optimization_direction", "maximize")
    observed_results = np.asarray([float(item["result"]) for item in observations if item.get("result") is not None], dtype=float)
    y_model = observed_results if direction != "minimize" else -1.0 * observed_results
    y_mean = float(np.mean(y_model)) if len(y_model) else 0.0
    y_std = float(np.std(y_model)) or 1.0
    anon_map = {model_id: chr(65 + index) for index, model_id in enumerate(fitted_ids[:6])}
    reverse_anon = {value: key for key, value in anon_map.items()}
    eval_points = []
    for point_offset, candidate_index in enumerate(eval_indices[: int(getattr(settings, "autobo_eval_points", 10) or 10)]):
        candidate = candidate_pool[int(candidate_index)]
        predictions = {}
        for model_id in fitted_ids:
            if model_id not in all_predictions or model_id not in anon_map:
                continue
            mean_scaled, sigma_scaled = all_predictions[model_id]
            mean_raw = float(mean_scaled[int(candidate_index)] * y_std + y_mean)
            if direction == "minimize":
                mean_raw = -1.0 * mean_raw
            sigma_raw = float(max(sigma_scaled[int(candidate_index)] * y_std, 1e-6))
            predictions[anon_map[model_id]] = {"mu": mean_raw, "sigma": sigma_raw}
        eval_points.append(
            {
                "point_id": f"P{point_offset + 1}",
                "candidate": candidate,
                "candidate_description": ", ".join(f"{key}={value}" for key, value in candidate.items()),
                "predictions": predictions,
            }
        )

    memory_manager = MemoryManager.from_dict(state.get("memory", {}))
    context = ContextBuilder.for_autobo_surrogate_eval(state, memory_manager)
    prompt = build_surrogate_plausibility_prompt(
        reaction_context=context.get("reaction_context", {}),
        top_observations=context.get("top_observations", []),
        bottom_observations=context.get("bottom_observations", []),
        eval_points=eval_points,
        knowledge_cards_text=context.get("knowledge_cards_text", ""),
        memory_rules=context.get("memory_rules", []),
    )
    parsed, _, usage = invoke_json_node(
        llm,
        state,
        prompt,
        {"evaluations": []},
        node_name="run_bo_iteration",
    )
    model_scores: dict[str, list[float]] = {model_id: [] for model_id in fitted_ids}
    audit_records: list[dict[str, Any]] = []
    point_lookup = {item["point_id"]: item for item in eval_points}
    for evaluation in parsed.get("evaluations", []):
        point_id = str(evaluation.get("point_id") or "")
        prediction_id = str(evaluation.get("prediction_id") or "")
        model_id = reverse_anon.get(prediction_id)
        point = point_lookup.get(point_id)
        if model_id is None or point is None:
            continue
        score = _coerce_float(evaluation.get("score"), default=3.0)
        model_scores.setdefault(model_id, []).append(score)
        prediction_payload = point.get("predictions", {}).get(prediction_id, {})
        audit_records.append(
            {
                "iteration": int(state.get("iteration", 0)),
                "point_id": point_id,
                "model_id": model_id,
                "candidate": dict(point.get("candidate", {})),
                "candidate_key": candidate_to_key(point.get("candidate", {})),
                "llm_score": score,
                "predicted_mu": prediction_payload.get("mu"),
                "predicted_sigma": prediction_payload.get("sigma"),
                "reasoning": str(evaluation.get("reasoning") or ""),
                "observed_y": None,
            }
        )

    scores = {
        model_id: float(np.mean(values)) if values else 3.0
        for model_id, values in model_scores.items()
    }
    return scores, audit_records, usage


def _trim_autobo_list(items: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    return list(items[-max(int(limit), 1) :])


def _trim_autobo_mapping(payload: dict[str, Any], limit: int = 50) -> dict[str, Any]:
    if len(payload) <= limit:
        return dict(payload)
    ordered_keys = sorted(payload.keys(), key=lambda key: int(key) if str(key).isdigit() else key)
    trimmed = ordered_keys[-max(int(limit), 1) :]
    return {key: payload[key] for key in trimmed}


def _recent_calibration_coverage(values: list[float] | list[bool], window: int = 10) -> float | None:
    if not values:
        return None
    sample = values[-max(int(window), 1) :]
    return float(np.mean(sample)) if sample else None


def _autobo_kernel_key(active_model_id: str) -> str:
    return str(resolve_recorded_kernel_config(active_model_id).get("key") or "none")


def _resolve_pending_plausibility_records(
    records: list[dict[str, Any]],
    candidate: dict[str, Any],
    observed_y: float,
) -> list[dict[str, Any]]:
    candidate_key = candidate_to_key(candidate or {})
    updated = []
    for record in records:
        item = dict(record)
        if item.get("candidate_key") == candidate_key and item.get("observed_y") is None:
            item["observed_y"] = float(observed_y)
        updated.append(item)
    return updated


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else default
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return default
        return numeric if np.isfinite(numeric) else default
    return default


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
        return numeric if np.isfinite(numeric) else None
    return None


def _state_seed(state: dict[str, Any], *, offset: int = 0) -> int:
    return int(state.get("random_seed_base", 0) or 0) + int(state.get("iteration", 0) or 0) + int(offset or 0)
