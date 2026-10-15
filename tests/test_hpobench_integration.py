from __future__ import annotations

import csv
from pathlib import Path

from config.settings import Settings
from core.autobo_engine import (
    _effective_active_model_id,
    _effective_autobo_surrogate_pool,
    resolve_recorded_kernel_config,
    resolve_recorded_surrogate_components,
)
from core.autobo_prompts import build_acquisition_selection_prompt, build_surrogate_plausibility_prompt
from core.dataset_oracle import DatasetOracle
from core.problem_loader import load_problem_file
from core.state import create_initial_state
from core.warm_start import _build_warm_start_direct_candidate_pool_prompt
from knowledge.prompts import build_prior_writer_prompt
from pools.component_pools import (
    _continuous_bounds,
    _continuous_scale,
    _denormalize_continuous,
    _normalize_continuous,
    candidate_distance,
)
from pools.deep_ensemble_features import build_deep_ensemble_feature_spec_prompt
from scripts.build_hpobench_tabular import DEFAULT_BENCHMARKS, build_problem_spec, export_table_to_csv
from tools.retrieval_tools import build_retrieval_tools


FORBIDDEN_HPO_PROMPT_WORDS = ("chemical", "chemistry", "reaction", "physicochemical", "RDKit")


def _write_hpo_fixture(tmp_path: Path) -> Path:
    csv_path = tmp_path / "hpobench_svm_146212.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["C", "gamma", "subsample", "test_acc", "test_loss", "seed_count", "seeds", "row_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "C": "1.0",
                "gamma": "0.01",
                "subsample": "0.5",
                "test_acc": "0.82",
                "test_loss": "0.18",
                "seed_count": "2",
                "seeds": "1;2",
                "row_id": "hpo_000001",
            }
        )
    problem = build_problem_spec(
        model="svm",
        task_id=146212,
        csv_path=str(csv_path),
        feature_columns=["C", "gamma"],
        rows=[
            {"C": "1.0", "gamma": "0.01"},
            {"C": "10.0", "gamma": "0.1"},
        ],
        target_metric="test_acc",
    )
    yaml_path = tmp_path / "hpobench_svm_problem.yaml"
    import yaml

    yaml_path.write_text(yaml.safe_dump(problem, sort_keys=False), encoding="utf-8")
    return yaml_path


def test_hpo_problem_loads_with_dataset_oracle(tmp_path: Path) -> None:
    problem = load_problem_file(_write_hpo_fixture(tmp_path))

    assert problem["application_domain"] == "hpo"
    assert problem["domain_profile"] == "ml_hyperparameter_optimization"
    assert problem["hpo_benchmark"]["model"] == "svm"
    assert problem["hpo_benchmark"]["task_id"] == 146212
    assert problem["hpo_benchmark"]["dataset_name"] == "openml_146212"
    assert problem["target_metric"] == "test_acc"
    assert problem["optimization_direction"] == "maximize"

    oracle = DatasetOracle.from_problem_spec(problem)
    assert oracle is not None
    assert oracle.feature_columns == ("C", "gamma")
    match = oracle.lookup({"C": "1.0", "gamma": "0.01"})
    assert match["result"] == 0.82
    assert match["metadata"]["dataset_row_id"] == "hpo_000001"

    state = create_initial_state(problem, Settings())
    assert "machine learning hyperparameter optimization" in state["messages"][0].content


def test_hpo_domain_prompts_do_not_use_chemistry_persona(tmp_path: Path) -> None:
    problem = load_problem_file(_write_hpo_fixture(tmp_path))
    context = {
        "application_domain": "hpo",
        "domain_profile": "ml_hyperparameter_optimization",
        "reaction_type": problem["reaction_type"],
        "target_metric": "test_acc",
        "optimization_direction": "maximize",
    }
    candidates = [
        {
            "id": 1,
            "candidate": {"C": "1.0", "gamma": "0.01"},
            "predicted_value": 0.81,
            "uncertainty": 0.03,
            "acquisition_value": 0.1,
        },
        {
            "id": 2,
            "candidate": {"C": "10.0", "gamma": "0.1"},
            "predicted_value": 0.79,
            "uncertainty": 0.08,
            "acquisition_value": 0.09,
        },
    ]
    prompts = [
        build_acquisition_selection_prompt(context, [], [], candidates, total_observations=2),
        build_surrogate_plausibility_prompt(context, [], [], []),
        _build_warm_start_direct_candidate_pool_prompt(
            context={
                "problem_features": problem,
                "proposal_value_guide": [],
                "constraints": problem.get("constraints", []),
                "warm_start_target": 1,
            },
            candidates=[{"id": 1, "candidate": candidates[0]["candidate"]}],
            target=1,
            total_direct_target=1,
            validation_feedback="",
            accepted_records=[],
        ),
        build_prior_writer_prompt(problem, "ml_hyperparameter_optimization")[0],
    ]
    for prompt in prompts:
        lowered = prompt.lower()
        for word in FORBIDDEN_HPO_PROMPT_WORDS:
            assert word.lower() not in lowered
        assert "machine learning" in lowered or "hpo" in lowered
        assert "hyperparameter" in lowered

    joined = "\n".join(prompts).lower()
    assert "model capacity" in joined
    assert "regularization" in joined
    assert "overfitting" in joined
    assert "validation performance" in joined

    assert '"domain_argument": ""' in prompts[0]
    assert "chemistry_argument" not in prompts[0]


def test_hpo_domain_disables_retrieval_and_deep_ensemble_feature_prompts(tmp_path: Path) -> None:
    problem = load_problem_file(_write_hpo_fixture(tmp_path))

    assert build_retrieval_tools(Settings(), problem, object(), lambda *args: ({}, {})) == []
    assert build_deep_ensemble_feature_spec_prompt(problem["variables"], problem) == ""


def test_hpobench_export_aggregates_seed_rows(tmp_path: Path) -> None:
    output_csv = tmp_path / "hpobench_svm_31.csv"
    rows = [
        {
            "C": 1.0,
            "gamma": 0.01,
            "dataset_fraction": 0.5,
            "seed": 1,
            "result": {"info": {"val_scores": {"acc": 0.8}, "test_scores": {"acc": 0.79}, "model_cost": 2.0}},
        },
        {
            "C": 1.0,
            "gamma": 0.01,
            "dataset_fraction": 0.5,
            "seed": 2,
            "result": {"info": {"val_scores": {"acc": 0.84}, "test_scores": {"acc": 0.81}, "model_cost": 4.0}},
        },
    ]

    exported = export_table_to_csv(
        rows,
        feature_columns=["C", "gamma", "dataset_fraction"],
        output_csv=output_csv,
    )

    assert len(exported) == 1
    assert exported[0]["val_acc"] == "0.82"
    assert exported[0]["val_loss"] == "0.18"
    assert exported[0]["test_acc"] == "0.8"
    assert exported[0]["cost"] == "3"
    assert exported[0]["seed_count"] == "2"
    assert output_csv.exists()


def test_hpobench_default_benchmark_mapping_matches_selected_tasks() -> None:
    assert DEFAULT_BENCHMARKS["svm"]["task_id"] == 146212
    assert DEFAULT_BENCHMARKS["svm"]["dataset_name"] == "openml_146212"
    assert DEFAULT_BENCHMARKS["svm"]["feature_columns"] == ["C", "gamma"]
    assert DEFAULT_BENCHMARKS["svm"]["fidelity_columns"] == ["subsample"]

    assert DEFAULT_BENCHMARKS["rf"]["task_id"] == 146606
    assert DEFAULT_BENCHMARKS["rf"]["dataset_name"] == "openml_146606"
    assert DEFAULT_BENCHMARKS["rf"]["feature_columns"] == [
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
    ]
    assert DEFAULT_BENCHMARKS["rf"]["fidelity_columns"] == ["n_estimators"]

    assert DEFAULT_BENCHMARKS["xgb"]["task_id"] == 146606
    assert DEFAULT_BENCHMARKS["xgb"]["dataset_name"] == "openml_146606"
    assert DEFAULT_BENCHMARKS["xgb"]["feature_columns"] == [
        "colsample_bytree",
        "eta",
        "max_depth",
        "reg_lambda",
    ]
    assert DEFAULT_BENCHMARKS["xgb"]["fidelity_columns"] == ["n_estimators"]

    assert DEFAULT_BENCHMARKS["nn"]["task_id"] == 168912
    assert DEFAULT_BENCHMARKS["nn"]["dataset_name"] == "openml_168912"
    assert DEFAULT_BENCHMARKS["nn"]["feature_columns"] == [
        "alpha",
        "batch_size",
        "depth",
        "learning_rate_init",
        "width",
    ]
    assert DEFAULT_BENCHMARKS["nn"]["fidelity_columns"] == ["iter"]


def test_hpobench_examples_keep_all_variables_continuous_grid() -> None:
    for path in [
        Path("examples/hpobench_svm_146212_problem.yaml"),
        Path("examples/hpobench_rf_146606_problem.yaml"),
        Path("examples/hpobench_xgb_146606_problem.yaml"),
        Path("examples/hpobench_nn_168912_problem.yaml"),
    ]:
        problem = load_problem_file(path)
        assert problem["application_domain"] == "hpo"
        for variable in problem["variables"]:
            assert variable["type"] == "continuous"
            assert variable.get("grid_values")
            assert variable.get("scale")


def test_hpobench_runtime_uses_continuous_gp_pool_and_records_no_categorical_kernel() -> None:
    problem = load_problem_file(Path("examples/hpobench_svm_146212_problem.yaml"))
    settings = Settings()

    pool = _effective_autobo_surrogate_pool(
        problem_spec=problem,
        settings=settings,
        switching_enabled=True,
    )

    assert pool[:3] == ["gp_matern52", "gp_matern32", "gp_smk"]
    assert "catboost" in pool
    assert "deep_ensemble" not in pool
    assert not any(model_id.startswith("gp_indicator_") for model_id in pool)
    assert not any(model_id.startswith("gp_exp_hamming_") for model_id in pool)

    assert _effective_active_model_id(
        autobo_state={"active_model": "gp_indicator_matern52"},
        problem_spec=problem,
        settings=settings,
        switching_enabled=True,
    ) == "gp_matern52"

    for model_id, kernel in [
        ("gp_matern52", "matern52"),
        ("gp_matern32", "matern32"),
        ("gp_smk", "smk"),
    ]:
        assert resolve_recorded_surrogate_components(model_id)["surrogate_model"] == model_id
        kernel_config = resolve_recorded_kernel_config(model_id)
        assert kernel_config["categorical_kernel"] is None
        assert kernel_config["continuous_kernel"] == kernel
        assert kernel_config["key"] == kernel


def _encode_continuous(variable: dict, value: float) -> float:
    low, high = _continuous_bounds(variable)
    return _normalize_continuous(value, low, high, _continuous_scale(variable))


def test_hpobench_log_scale_variables_are_log_normalized() -> None:
    nn_problem = load_problem_file(Path("examples/hpobench_nn_168912_problem.yaml"))
    alpha = next(variable for variable in nn_problem["variables"] if variable["name"] == "alpha")

    assert alpha["scale"] == "log"
    assert abs(_encode_continuous(alpha, alpha["grid_values"][0]) - 0.0) < 1e-12
    assert abs(_encode_continuous(alpha, alpha["grid_values"][-1]) - 1.0) < 1e-12
    assert abs(_encode_continuous(alpha, alpha["grid_values"][1]) - (1.0 / 9.0)) < 1e-4

    svm_problem = load_problem_file(Path("examples/hpobench_svm_146212_problem.yaml"))
    c_value = next(variable for variable in svm_problem["variables"] if variable["name"] == "C")
    assert c_value["scale"] == "log2"
    assert abs(_encode_continuous(c_value, 1.0) - 0.5) < 1e-12

    xgb_problem = load_problem_file(Path("examples/hpobench_xgb_146606_problem.yaml"))
    reg_lambda = next(variable for variable in xgb_problem["variables"] if variable["name"] == "reg_lambda")
    assert reg_lambda["scale"] == "log"
    assert abs(_encode_continuous(reg_lambda, reg_lambda["grid_values"][1]) - (1.0 / 9.0)) < 1e-4


def test_log_scale_distance_and_inverse_transform_use_log_coordinates() -> None:
    variable = {
        "name": "C",
        "type": "continuous",
        "domain": [2.0**-10, 2.0**10],
        "scale": "log2",
    }

    assert abs(_denormalize_continuous(0.5, 2.0**-10, 2.0**10, "log2") - 1.0) < 1e-12
    assert abs(candidate_distance({"C": 2.0**-10}, {"C": 1.0}, [variable]) - 0.5) < 1e-12

    linear_variable = {"name": "x", "type": "continuous", "domain": [0.0, 1.0], "scale": "linear"}
    assert abs(_encode_continuous(linear_variable, 0.25) - 0.25) < 1e-12
