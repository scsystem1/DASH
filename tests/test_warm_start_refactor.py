from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage

from config.settings import Settings
from core.dataset_oracle import DatasetOracle
from core.graph import (
    _delta_best,
    _extract_json_from_response,
    _extract_last_json,
    _merge_llm_usage,
    _rank_and_filter_cards,
    _update_knowledge_deck_after_interpretation,
    _update_hypothesis_statuses,
    build_chembo_graph,
    compute_convergence_state,
)
from core.problem_loader import load_problem_file
from core.state import create_initial_state
from core.warm_start import (
    _build_random_warm_start_pool,
    _build_warm_start_direct_candidate_pool_prompt,
    _build_warm_start_direct_structured_prompt,
    _extract_partial_direct_selection_payloads,
    interpret_warm_start_result,
    plan_warm_start,
    run_warm_start_postmortem,
)
from knowledge.knowledge_card import create_knowledge_card
from knowledge.prompts import build_prior_writer_prompt
from knowledge.prior_writer import write_initial_priors
from memory.memory_manager import MemoryManager
from pools.component_pools import candidate_to_key


class _GraphDummyLLM:
    def bind_tools(self, tools):
        del tools
        return self

    def invoke(self, messages):
        raise AssertionError(f"Unexpected direct LLM invocation: {messages}")


class _CoverageDummyModel:
    def predict(self, candidates):
        means = []
        sigmas = []
        for candidate in candidates:
            ligand = candidate.get("ligand")
            means.append({"A": 90.0, "B": 40.0, "C": 50.0, "D": 30.0}.get(ligand, 10.0))
            sigmas.append({"A": 1.0, "B": 20.0, "C": 5.0, "D": 25.0}.get(ligand, 1.0))
        return means, sigmas


def _example_problem(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return load_problem_file(root / "examples" / f"{name}_problem.yaml")


def test_suzuki_problem_loads_with_dataset_oracle() -> None:
    problem = _example_problem("suzuki")

    oracle = DatasetOracle.from_problem_spec(problem)
    assert oracle is not None
    assert oracle.size == 5760
    assert oracle.feature_columns == (
        "Reactant_1_Name",
        "Reactant_2_Name",
        "Ligand_Short_Hand",
        "Reagent_1_Short_Hand",
        "Solvent_1_Short_Hand",
    )

    expected_domains = {
        "Reactant_1_Name": 7,
        "Reactant_2_Name": 4,
        "Ligand_Short_Hand": 12,
        "Reagent_1_Short_Hand": 8,
        "Solvent_1_Short_Hand": 6,
    }
    assert {name: len(values) for name, values in oracle.domain_values.items()} == expected_domains


def test_oer_problem_exposes_declared_discrete_simplex_spec_with_dataset_validation() -> None:
    from core.autobo_engine import (
        _build_pure_reasoning_space_spec,
        _resolve_structured_pure_reasoning_candidate,
    )

    problem = _example_problem("oer")
    state = {"problem_spec": problem, "observations": []}

    assert all(variable["domain"] == [0.0, 1.0] for variable in problem["variables"])
    assert all(len(variable["allowed_values"]) == 11 for variable in problem["variables"])

    spec = _build_pure_reasoning_space_spec(state)
    assert spec is not None
    assert spec["mode"] == "declared_discrete_simplex"
    assert spec["dataset_backed"] is True
    assert "exact allowed levels = [0.0, 0.1, 0.2" in spec["space_description"]
    assert "arbitrary continuous fractions" in spec["space_description"]
    assert "must equal 1.0" in spec["space_description"]

    valid_candidate, valid_failure = _resolve_structured_pure_reasoning_candidate(
        {
            "variables": {
                "ni_load": 0.2,
                "fe_load": 0.6,
                "co_load": 0.2,
                "mn_load": 0.0,
                "ce_load": 0.0,
                "la_load": 0.0,
            }
        },
        structured_spec=spec,
        state=state,
    )
    assert valid_failure == ""
    assert valid_candidate == {
        "ni_load": "0.2",
        "fe_load": "0.6",
        "co_load": "0.2",
        "mn_load": "0.0",
        "ce_load": "0.0",
        "la_load": "0.0",
    }

    _, sum_failure = _resolve_structured_pure_reasoning_candidate(
        {
            "variables": {
                "ni_load": 0.5,
                "fe_load": 0.5,
                "co_load": 0.5,
                "mn_load": 0.0,
                "ce_load": 0.0,
                "la_load": 0.0,
            }
        },
        structured_spec=spec,
        state=state,
    )
    assert "must equal 1.0" in sum_failure

    _, grid_failure = _resolve_structured_pure_reasoning_candidate(
        {
            "variables": {
                "ni_load": 0.65,
                "fe_load": 0.25,
                "co_load": 0.08,
                "mn_load": 0.0,
                "ce_load": 0.02,
                "la_load": 0.0,
            }
        },
        structured_spec=spec,
        state=state,
    )
    assert "declared discrete simplex levels" in grid_failure

    _, dataset_failure = _resolve_structured_pure_reasoning_candidate(
        {
            "variables": {
                "ni_load": 0.0,
                "fe_load": 0.1,
                "co_load": 0.1,
                "mn_load": 0.1,
                "ce_load": 0.1,
                "la_load": 0.6,
            }
        },
        structured_spec=spec,
        state=state,
    )
    assert "does not correspond to a legal dataset row" in dataset_failure


def _sample_knowledge_cards() -> list[dict]:
    return [
        {
            "card_id": "kc_ligand",
            "text": "For ligand_SMILES, bulky electron-rich ligands are often productive starting points.",
            "card_type": "reagent_property",
            "confidence": 0.85,
            "targets": ["ligand_SMILES"],
            "actionable_for": ["warm_start", "hypothesis_generation"],
            "scope": "target",
            "status": "active",
            "evidence_refs": ["S01"],
            "source_type": "llm_internal_prior",
            "validation": {"used_count": 0, "supported_count": 0, "contradicted_count": 0, "last_used_iter": None},
        },
        {
            "card_id": "kc_temp",
            "text": "For temperature, moderate-to-high values are often needed to activate challenging coupling manifolds.",
            "card_type": "operating_window",
            "confidence": 0.62,
            "targets": ["temperature"],
            "actionable_for": ["warm_start"],
            "scope": "general",
            "status": "active",
            "evidence_refs": ["S02"],
            "source_type": "llm_internal_prior",
            "validation": {"used_count": 0, "supported_count": 0, "contradicted_count": 0, "last_used_iter": None},
        },
    ]


def _usage() -> dict[str, int | bool]:
    return {
        "calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_calls": 0,
        "estimated": False,
    }


def _parse_direct_target(prompt: str) -> int:
    match = re.search(r"Return exactly\s+(\d+)\s+", prompt)
    return int(match.group(1)) if match else 0


def _invoke_tool_loop_factory():
    def _fake_invoke_tool_loop(llm, state, prompt, tool_map, max_turns=6, **kwargs):
        del llm, max_turns, kwargs
        messages = [HumanMessage(content=prompt)]

        if "Generate 3-5 high-value hypotheses" in prompt:
            messages.append(
                AIMessage(
                    content=json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "id": "H1",
                                    "text": "Coverage-first warm starts should reveal productive discrete regions early.",
                                    "mechanism": "Initialization should balance chemistry priors with broad categorical coverage.",
                                    "testable_prediction": "Strong ligands should appear without collapsing diversity.",
                                    "confidence": "medium",
                                    "status": "active",
                                }
                            ],
                            "working_memory_focus": "Use knowledge cards to balance coverage and exploitation.",
                        }
                    )
                )
            )
            return messages, "", _usage()

        if "Select high-value direct warm-start experiments" in prompt:
            target = _parse_direct_target(prompt)
            problem_spec = state.get("problem_spec", {})
            oracle = DatasetOracle.from_problem_spec(problem_spec)
            if oracle is not None and str(problem_spec.get("reaction_type", "")).upper() != "OCM":
                observed = {
                    candidate_to_key(item.get("candidate", {}))
                    for item in state.get("observations", [])
                    if item.get("candidate")
                }
                selections = [
                    {
                        "variables": dict(candidate),
                        "reasoning": f"High-value direct warm-start seed {index + 1}.",
                        "hypothesis_alignment": "Tests a chemically plausible early region.",
                        "information_value": "Adds chemically plausible warm-start coverage.",
                        "concerns": "",
                        "confidence": 0.8,
                        "purpose": "exploit" if index < max(1, target // 3) else "explore",
                    }
                    for index, candidate in enumerate(
                        candidate for candidate in oracle.candidates if candidate_to_key(candidate) not in observed
                    )
                    if index < target
                ]
            else:
                from core.autobo_engine import _build_pure_reasoning_space_spec

                spec = _build_pure_reasoning_space_spec(state)
                if spec and spec.get("mode") == "ocm_encoded_domain":
                    from core.ocm_domain import load_ocm_domain_spec

                    domain = load_ocm_domain_spec(spec["ocm_dataset_path"])
                    selections = []
                    for row in domain.dataframe.itertuples(index=False):
                        condition_key = None
                        row_condition = (
                            str(row.CT).strip(),
                            str(row.Ar_flow).strip(),
                            str(row.CH4_flow).strip(),
                            str(row.O2_flow).strip(),
                        )
                        for (ct, ar_level, ratio_slot), flows in domain.condition_lookup.items():
                            if (str(ct), *(str(value) for value in flows)) == row_condition:
                                condition_key = (ct, ar_level, ratio_slot)
                                break
                        if condition_key is None:
                            continue
                        cat = str(domain.catalyst_list.index(str(row.Name).strip()))
                        ct, ar_level, ratio_slot = condition_key
                        selections.append(
                            {
                                "cat": cat,
                                "Temp": str(row.Temp).strip(),
                                "CT": ct,
                                "ar_level": ar_level,
                                "ch4_o2_ratio": ratio_slot,
                                "reasoning": f"Encoded OCM warm-start seed {len(selections) + 1}.",
                                "purpose": "exploit" if not selections else "explore",
                                "confidence": 0.8,
                            }
                        )
                        if len(selections) >= target:
                            break
                else:
                    default_selection = dict((spec or {}).get("default_response", {}))
                    default_selection["reasoning"] = "Use the default encoded-domain warm-start seed."
                    selections = [dict(default_selection) for _ in range(target)]
            messages.append(
                AIMessage(
                    content=json.dumps(
                        {
                            "strategy_summary": "Choose high-value direct warm-start points with plausible coverage.",
                            "selections": selections,
                        }
                    )
                )
            )
            return messages, "", _usage()

        if "Select the single best embedding method" in prompt:
            messages.append(
                AIMessage(
                    content=json.dumps(
                        {
                            "method": "one_hot",
                            "params": {},
                            "rationale": "Simple discrete baseline.",
                            "confidence": 0.9,
                        }
                    )
                )
            )
            return messages, "", _usage()

        if "Configure the BoTorch surrogate" in prompt:
            messages.append(
                AIMessage(
                    content=json.dumps(
                        {
                            "surrogate_model": "gp",
                            "surrogate_params": {},
                            "kernel_config": {"key": "matern52", "params": {}, "rationale": "Stable default."},
                            "acquisition_function": "log_ei",
                            "af_params": {},
                            "rationale": "Use the default BO stack after warm start.",
                            "confidence": 0.85,
                        }
                    )
                )
            )
            return messages, "", _usage()

        raise AssertionError(f"Unhandled prompt in fake tool loop:\n{prompt}")

    return _fake_invoke_tool_loop


def _direct_extract_last_json(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return json.loads(message.content)
    return None


def _state_messages_identity(messages):
    return messages


def _updated_campaign_summary_stub(state, messages):
    del state
    return "\n".join(getattr(message, "content", "") for message in messages if getattr(message, "content", ""))


def _attach_llm_usage_stub(update, state, node_name, usage):
    del state, node_name
    update["last_llm_usage"] = dict(usage)


def _memory_manager_from_state(state: dict, settings: Settings) -> MemoryManager:
    return MemoryManager.from_dict(
        state.get("memory", {}),
        capacity=settings.episodic_memory_capacity,
        node_budgets=getattr(settings, "memory_node_budgets", {}),
        consolidation_every_n=int(getattr(settings, "memory_consolidation_every_n", 5)),
        enable_llm_consolidation=bool(getattr(settings, "memory_llm_consolidation_enabled", True)),
        llm_cooldown_iters=int(getattr(settings, "memory_llm_consolidation_cooldown_iters", 5)),
        episode_keep_recent=int(getattr(settings, "memory_episode_keep_recent", 24)),
        episode_keep_salient=int(getattr(settings, "memory_episode_keep_salient", 96)),
    )


def _run_to_first_interrupt(monkeypatch, problem_spec: dict, *, cards: list[dict]) -> dict:
    monkeypatch.setattr(
        "core.graph._create_llm",
        lambda settings, enable_thinking_override=None, **kwargs: _GraphDummyLLM(),
    )
    monkeypatch.setattr(
        "core.graph.write_initial_priors",
        lambda problem_spec, settings, llm, invoke_json: (cards, {"needs_evidence": [], "llm_usage": {}}),
    )
    monkeypatch.setattr("core.graph._invoke_tool_loop", _invoke_tool_loop_factory())

    settings = Settings(
        initial_doe_size=20,
        max_bo_iterations=40,
        human_input_mode="dataset_auto",
    )
    settings.knowledge_enabled = True
    graph = build_chembo_graph(settings)
    initial_state = create_initial_state(problem_spec, settings)
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    list(graph.stream(initial_state, config=config, stream_mode="updates"))
    return graph.get_state(config).values


def _toy_variables() -> list[dict]:
    return [
        {"name": "ligand", "type": "categorical", "domain": ["A", "B", "C", "D"]},
        {"name": "solvent", "type": "categorical", "domain": ["S1", "S2"]},
        {"name": "temperature", "type": "continuous", "domain": [60.0, 120.0]},
    ]


def _toy_pool() -> list[dict]:
    return [
        {"ligand": "A", "solvent": "S1", "temperature": 60.0},
        {"ligand": "B", "solvent": "S1", "temperature": 70.0},
        {"ligand": "C", "solvent": "S2", "temperature": 80.0},
        {"ligand": "D", "solvent": "S2", "temperature": 90.0},
        {"ligand": "A", "solvent": "S2", "temperature": 100.0},
        {"ligand": "B", "solvent": "S2", "temperature": 110.0},
    ]


def test_build_random_warm_start_pool_dataset_backed_is_seeded_and_excludes_observed() -> None:
    problem_spec = _example_problem("dar")
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    assert oracle is not None
    observed_keys = {candidate_to_key(dict(oracle.candidates[0])), candidate_to_key(dict(oracle.candidates[1]))}

    pool_a = _build_random_warm_start_pool(
        problem_spec["variables"],
        pool_size=25,
        seed=7,
        observed_keys=observed_keys,
        hard_constraints=[],
        candidate_pool=list(oracle.candidates),
    )
    pool_b = _build_random_warm_start_pool(
        problem_spec["variables"],
        pool_size=25,
        seed=7,
        observed_keys=observed_keys,
        hard_constraints=[],
        candidate_pool=list(oracle.candidates),
    )
    pool_c = _build_random_warm_start_pool(
        problem_spec["variables"],
        pool_size=25,
        seed=8,
        observed_keys=observed_keys,
        hard_constraints=[],
        candidate_pool=list(oracle.candidates),
    )

    assert pool_a == pool_b
    assert pool_a != pool_c
    assert len(pool_a) == 25
    assert len({candidate_to_key(candidate) for candidate in pool_a}) == 25
    assert all(oracle.candidate_exists(candidate) for candidate in pool_a)
    assert not {candidate_to_key(candidate) for candidate in pool_a} & observed_keys


def test_unseen_category_coverage_builds_zero_count_categorical_dictionary() -> None:
    from core.autobo_engine import _build_unseen_category_options

    variables = [
        {"name": "ligand", "type": "categorical", "domain": ["A", "B", "C", "D"]},
        {"name": "solvent", "type": "categorical", "domain": ["S1", "S2"]},
        {"name": "temperature", "type": "continuous", "domain": [90, 120]},
    ]
    observations = [{"candidate": {"ligand": "A", "solvent": "S1", "temperature": 90}, "result": 91.0}]
    candidate_pool = [
        {"ligand": "A", "solvent": "S1", "temperature": 120},
        {"ligand": "B", "solvent": "S1", "temperature": 120},
        {"ligand": "C", "solvent": "S1", "temperature": 120},
        {"ligand": "D", "solvent": "S2", "temperature": 120},
    ]

    unseen = _build_unseen_category_options(
        search_space=variables,
        observations=observations,
        candidate_pool=candidate_pool,
    )

    assert set(unseen) == {"ligand", "solvent"}
    assert [item["value"] for item in unseen["ligand"]] == ["B", "C", "D"]
    assert unseen["ligand"][0]["legal_candidate_count"] == 1
    assert unseen["solvent"] == [
        {"value": "S2", "role": "solvent", "unseen": True, "legal_candidate_count": 1}
    ]
    assert "temperature" not in unseen


def test_hpo_continuous_grid_coverage_builds_zero_count_grid_dictionary() -> None:
    from core.autobo_engine import _build_unseen_category_options

    variables = [
        {
            "name": "C",
            "role": "hyperparameter",
            "type": "continuous",
            "domain": [1.0, 100.0],
            "grid_values": [1.0, 10.0, 100.0],
            "scale": "log",
        },
        {
            "name": "gamma",
            "role": "hyperparameter",
            "type": "continuous",
            "domain": [0.01, 1.0],
            "grid_values": [0.01, 0.1, 1.0],
            "scale": "log",
        },
    ]
    observations = [{"candidate": {"C": "1.0", "gamma": "0.01"}, "result": 0.8}]
    candidate_pool = [
        {"C": "10.0", "gamma": "0.01"},
        {"C": "100.0", "gamma": "0.1"},
        {"C": "1.0", "gamma": "1.0"},
    ]

    unseen = _build_unseen_category_options(
        search_space=variables,
        observations=observations,
        candidate_pool=candidate_pool,
        include_continuous_grid=True,
    )

    assert set(unseen) == {"C", "gamma"}
    assert unseen["C"] == [
        {
            "value": "10.0",
            "role": "hyperparameter",
            "unseen": True,
            "legal_candidate_count": 1,
            "option_type": "continuous_grid",
        },
        {
            "value": "100.0",
            "role": "hyperparameter",
            "unseen": True,
            "legal_candidate_count": 1,
            "option_type": "continuous_grid",
        },
    ]
    assert unseen["gamma"][-1]["value"] == "1.0"
    assert unseen["gamma"][-1]["option_type"] == "continuous_grid"


def test_unseen_category_coverage_window_gating() -> None:
    from core.autobo_engine import _unseen_category_coverage_should_run

    settings = Settings(
        autobo_unseen_category_exploration_enabled=True,
        autobo_unseen_category_window=5,
        autobo_unseen_category_slots=1,
    )
    observations = [{"candidate": {"x": index}, "result": float(index)} for index in range(20)]

    assert _unseen_category_coverage_should_run(
        settings=settings,
        observations=observations[:10],
        warm_start_target=10,
        ensemble_sur_enabled=False,
        zero_llm_mode=False,
    )
    assert _unseen_category_coverage_should_run(
        settings=settings,
        observations=observations[:14],
        warm_start_target=10,
        ensemble_sur_enabled=False,
        zero_llm_mode=False,
    )
    assert not _unseen_category_coverage_should_run(
        settings=settings,
        observations=observations[:15],
        warm_start_target=10,
        ensemble_sur_enabled=False,
        zero_llm_mode=False,
    )
    assert not _unseen_category_coverage_should_run(
        settings=settings,
        observations=observations[:10],
        warm_start_target=10,
        ensemble_sur_enabled=True,
        zero_llm_mode=False,
    )


def test_model_schema_evaluation_triggers_at_warm_start_completion_then_every_interval() -> None:
    from core.autobo_engine import _should_trigger_model_schema_evaluation

    assert _should_trigger_model_schema_evaluation(
        n_total_obs=9,
        warm_start_target=10,
        last_eval_n=-1,
        eval_interval=5,
    ) == (False, "before_warm_start_complete")
    assert _should_trigger_model_schema_evaluation(
        n_total_obs=10,
        warm_start_target=10,
        last_eval_n=-1,
        eval_interval=5,
    ) == (True, "warm_start_complete")
    assert _should_trigger_model_schema_evaluation(
        n_total_obs=14,
        warm_start_target=10,
        last_eval_n=0,
        eval_interval=5,
    ) == (False, "evaluation_not_due")
    assert _should_trigger_model_schema_evaluation(
        n_total_obs=15,
        warm_start_target=10,
        last_eval_n=0,
        eval_interval=5,
    ) == (True, "interval")
    assert _should_trigger_model_schema_evaluation(
        n_total_obs=19,
        warm_start_target=10,
        last_eval_n=5,
        eval_interval=5,
    ) == (False, "evaluation_not_due")
    assert _should_trigger_model_schema_evaluation(
        n_total_obs=20,
        warm_start_target=10,
        last_eval_n=5,
        eval_interval=5,
    ) == (True, "interval")


def test_settings_reads_switch_surrogate(tmp_path: Path) -> None:
    assert Settings().switch_surrogate is True

    config_path = tmp_path / "fixed_surrogate.yaml"
    config_path.write_text("switch_surrogate: false\n", encoding="utf-8")

    loaded = Settings.from_yaml(str(config_path))

    assert loaded.switch_surrogate is False


def test_default_autobo_surrogate_pool_matches_baseline() -> None:
    assert Settings().autobo_surrogate_pool == [
        "gp_indicator_matern52",
        "gp_exp_hamming_matern52",
        "gp_indicator_matern32",
        "catboost",
    ]


def test_bo_torch_device_defaults_to_gpu_for_dataset_runs() -> None:
    settings = Settings()
    assert settings.bo_torch_device == "cuda:5"
    assert settings.bo_torch_devices == ["cuda:5", "cuda:4", "cuda:3", "cuda:2"]

    runner = Path("scripts/run_dataset_3x40_10init_no_pr_no_pure_autobo_ensemble.sh").read_text(encoding="utf-8")
    assert 'BO_TORCH_DEVICE="${BO_TORCH_DEVICE:-cuda:5}"' in runner
    assert 'settings.bo_torch_device = os.environ.get("CHEMBO_BO_TORCH_DEVICE")' in runner


def test_switch_surrogate_false_fixes_matern52_and_skips_surrogate_evaluation(monkeypatch) -> None:
    from core import autobo_engine

    class _DummySurrogate:
        def fit(self, candidates, values):
            del candidates, values

        def predict(self, candidates):
            return [float(index) for index, _ in enumerate(candidates)], [1.0 for _ in candidates]

    monkeypatch.setattr(
        autobo_engine,
        "_parallel_loocv_evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("surrogate LOOCV evaluation should not run")),
    )
    monkeypatch.setattr(
        autobo_engine,
        "_create_surrogate_from_spec",
        lambda *args, **kwargs: _DummySurrogate(),
    )

    prior_fitness_log = {"previous": {"gp_indicator_matern32": {"composite": 0.9}}}
    prior_calibration_log = [{"iteration": 1, "active_model": "gp_indicator_matern32"}]
    prior_switch_history = [{"iteration": 1, "from": "gp_indicator_matern52", "to": "gp_indicator_matern32"}]
    state = {
        "iteration": 2,
        "problem_spec": {
            "variables": [{"name": "ligand", "type": "categorical", "domain": ["A", "B", "C", "D", "E", "F", "G", "H", "I"]}],
            "dataset": {},
        },
        "optimization_direction": "maximize",
        "observations": [
            {"candidate": {"ligand": "A"}, "result": 1.0},
            {"candidate": {"ligand": "B"}, "result": 2.0},
            {"candidate": {"ligand": "C"}, "result": 3.0},
            {"candidate": {"ligand": "D"}, "result": 4.0},
            {"candidate": {"ligand": "E"}, "result": 1.5},
            {"candidate": {"ligand": "F"}, "result": 2.5},
            {"candidate": {"ligand": "G"}, "result": 3.5},
            {"candidate": {"ligand": "H"}, "result": 4.5},
        ],
        "warm_start_target": 8,
        "autobo_state": {
            "active_model": "gp_indicator_matern32",
            "last_eval_n": -1,
            "fitness_log": prior_fitness_log,
            "calibration_log": prior_calibration_log,
            "switch_history": prior_switch_history,
        },
        "performance_log": [],
    }
    settings = Settings(
        switch_surrogate=False,
        ensemble_sur=True,
        ensemble_af=False,
        autobo_unseen_category_exploration_enabled=False,
        autobo_initial_active="gp_indicator_matern32",
        autobo_surrogate_pool=["gp_indicator_matern32", "catboost"],
    )

    runtime = autobo_engine.run_autobo_iteration(
        state=state,
        settings=settings,
        llm=None,
        invoke_json_node=lambda *args, **kwargs: ({}, [], {}),
    )

    metadata = runtime["payload"]["metadata"]
    assert metadata["active_model_internal"] == "gp_indicator_matern52"
    assert metadata["trigger_reason"] == "switch_surrogate_disabled"
    assert metadata["switch_info"]["switched"] is False
    assert metadata["switch_info"]["reason"] == "switch_surrogate disabled; fixed gp_indicator_matern52 surrogate is used."
    assert metadata["ensemble_af_enabled"] is False
    assert runtime["autobo_state"]["active_model"] == "gp_indicator_matern52"
    assert runtime["autobo_state"]["fitness_log"] == prior_fitness_log
    assert runtime["autobo_state"]["calibration_log"] == prior_calibration_log
    assert runtime["autobo_state"]["switch_history"] == prior_switch_history
    assert runtime["autobo_state"]["last_eval_n"] == -1


def test_unseen_category_coverage_validates_llm_targets_and_fills_missing() -> None:
    from core.autobo_engine import _validate_unseen_category_targets

    unseen = {
        "ligand": [
            {"value": "B", "role": "ligand", "unseen": True, "legal_candidate_count": 2},
            {"value": "C", "role": "ligand", "unseen": True, "legal_candidate_count": 1},
        ],
        "solvent": [{"value": "S2", "role": "solvent", "unseen": True, "legal_candidate_count": 1}],
    }
    raw_targets = [
        {"variable": "ligand", "value": "B", "reasoning": "try new ligand"},
        {"variable": "ligand", "value": "B", "reasoning": "duplicate"},
        {"variable": "ligand", "value": "already_seen"},
        {"variable": "missing", "value": "X"},
    ]

    targets = _validate_unseen_category_targets(raw_targets, unseen, slots=2)

    assert targets[0]["variable"] == "ligand"
    assert targets[0]["value"] == "B"
    assert targets[0]["selected_by_llm"] is True
    assert targets[1]["variable"] == "ligand"
    assert targets[1]["value"] == "C"
    assert targets[1]["selected_by_llm"] is False


class _TargetFillDummyModel:
    def predict(self, candidates):
        means = []
        sigmas = []
        for candidate in candidates:
            means.append(float(candidate.get("temperature", 0.0)))
            sigmas.append(1.0)
        return means, sigmas


def test_unseen_category_coverage_fills_targets_with_qlogei_and_preserves_full_bo_shortlist() -> None:
    from core.autobo_engine import _coverage_records_for_targets, _merge_shortlist_with_coverage

    observations = [{"candidate": {"ligand": "A", "solvent": "S1", "temperature": 10}, "result": 10.0}]
    candidate_pool = [
        {"ligand": "B", "solvent": "S1", "temperature": 20},
        {"ligand": "B", "solvent": "S2", "temperature": 80},
        {"ligand": "C", "solvent": "S2", "temperature": 30},
    ]
    normal_shortlist = [
        {
            "candidate": {"ligand": f"A{index}", "solvent": "S1", "temperature": index},
            "predicted_value": float(index),
            "uncertainty": 1.0,
            "acquisition_value": float(100 - index),
            "acquisition_value_raw": float(100 - index),
            "selection_step": index + 1,
            "selection_mode": "ensemble_reference",
            "rank": index + 1,
        }
        for index in range(5)
    ]
    unseen = {
        "ligand": [
            {"value": "B", "role": "ligand", "unseen": True, "legal_candidate_count": 2},
            {"value": "C", "role": "ligand", "unseen": True, "legal_candidate_count": 1},
        ],
    }
    targets = [
        {"variable": "ligand", "value": "B", "unseen": True, "selected_by_llm": True},
        {"variable": "ligand", "value": "C", "unseen": True, "selected_by_llm": True},
    ]

    coverage_records = _coverage_records_for_targets(
        active_model=_TargetFillDummyModel(),
        candidate_pool=candidate_pool,
        observations=observations,
        targets=targets,
        unseen_options=unseen,
        normal_shortlist=normal_shortlist,
        direction="maximize",
        seed=3,
        top_k=5,
        coverage_slots=2,
    )
    merged = _merge_shortlist_with_coverage(normal_shortlist, coverage_records, top_k=5, coverage_slots=2)

    assert len(merged) == 7
    assert [item["candidate"] for item in merged[:5]] == [item["candidate"] for item in normal_shortlist]
    assert merged[5]["selection_mode"] == "llm_guided_unseen_category_coverage"
    assert merged[5]["candidate"] == {"ligand": "B", "solvent": "S2", "temperature": 80}
    assert merged[5]["coverage_targets"] == [
        {
            "variable": "ligand",
            "value": "B",
            "unseen": True,
            "selected_by_llm": True,
            "reasoning": "",
        }
    ]


def test_hpo_grid_coverage_record_preserves_dataset_backed_candidate() -> None:
    from core.autobo_engine import _coverage_records_for_targets

    observations = [{"candidate": {"C": "1.0", "gamma": "0.01"}, "result": 0.8}]
    candidate_pool = [
        {"C": "10.0", "gamma": "0.01"},
        {"C": "10.0", "gamma": "0.1"},
        {"C": "100.0", "gamma": "0.1"},
    ]

    coverage_records = _coverage_records_for_targets(
        active_model=_TargetFillDummyModel(),
        candidate_pool=candidate_pool,
        observations=observations,
        targets=[
            {
                "variable": "C",
                "value": "10.0",
                "selected_by_llm": True,
                "option_type": "continuous_grid",
            }
        ],
        unseen_options={"C": [{"value": "10.0", "option_type": "continuous_grid"}]},
        normal_shortlist=[],
        direction="maximize",
        seed=5,
        top_k=5,
        coverage_slots=1,
    )

    assert len(coverage_records) == 1
    assert coverage_records[0]["candidate"] in candidate_pool
    assert coverage_records[0]["selection_mode"] == "llm_guided_unseen_grid_coverage"
    assert coverage_records[0]["coverage_targets"] == [
        {
            "variable": "C",
            "value": "10.0",
            "unseen": True,
            "selected_by_llm": True,
            "reasoning": "",
            "option_type": "continuous_grid",
        }
    ]


def test_acquisition_selection_prompt_includes_never_tested_coverage_targets() -> None:
    from core.autobo_prompts import build_acquisition_selection_prompt

    prompt = build_acquisition_selection_prompt(
        reaction_context={"reaction_type": "DAR"},
        top_observations=[],
        bottom_observations=[],
        candidates=[
            {
                "id": 1,
                "candidate": {"ligand_SMILES": "L-bo"},
                "predicted_value": 2.0,
                "uncertainty": 1.0,
                "acquisition_value": 4.0,
                "selection_step": 1,
                "selection_mode": "ensemble_reference",
                "unseen_categorical_values": [
                    {
                        "variable": "ligand_SMILES",
                        "value": "L-bo",
                    }
                ],
            },
            {
                "id": 2,
                "candidate": {"ligand_SMILES": "L-new"},
                "predicted_value": 1.0,
                "uncertainty": 2.0,
                "acquisition_value": 3.0,
                "selection_step": 6,
                "selection_mode": "llm_guided_unseen_category_coverage",
                "coverage_targets": [
                    {
                        "variable": "ligand_SMILES",
                        "value": "L-new",
                        "unseen": True,
                        "selected_by_llm": True,
                    }
                ],
            }
        ],
        total_observations=10,
        ensemble_mode=False,
        early_exploration_info={"enabled": True, "bo_round_index": 1, "window": 5},
    )

    assert "BO-proposed unseen categorical value(s): ligand_SMILES=L-bo" in prompt
    assert "coverage target: ligand_SMILES=L-new" in prompt
    assert "this categorical value has never been tested before" in prompt
    assert "[Early Post-Warm-Start Exploration Guardrail]" in prompt
    assert "[Early Unseen Exploration Priority]" in prompt
    assert "First prefer a chemically plausible BO-proposed candidate with unseen categorical value(s)." in prompt
    assert "prefer a chemically plausible coverage candidate" in prompt
    assert "Select a BO-proposed candidate without unseen categorical value(s) only when" in prompt


def test_hpo_acquisition_selection_prompt_uses_balanced_grid_coverage_language() -> None:
    from core.autobo_prompts import build_acquisition_selection_prompt

    prompt = build_acquisition_selection_prompt(
        reaction_context={
            "application_domain": "hpo",
            "domain_profile": "ml_hyperparameter_optimization",
            "reaction_type": "HPOBENCH_SVM",
        },
        top_observations=[{"candidate": {"C": "1.0", "gamma": "0.1"}, "result": 0.9}],
        bottom_observations=[{"candidate": {"C": "0.001", "gamma": "1024.0"}, "result": 0.5}],
        candidates=[
            {
                "id": 1,
                "candidate": {"C": "1.0", "gamma": "0.1"},
                "predicted_value": 0.91,
                "uncertainty": 0.02,
                "acquisition_value": 0.1,
                "selection_step": 1,
                "selection_mode": "ensemble_reference",
            },
            {
                "id": 2,
                "candidate": {"C": "10.0", "gamma": "0.1"},
                "predicted_value": 0.9,
                "uncertainty": 0.05,
                "acquisition_value": 0.09,
                "selection_step": 6,
                "selection_mode": "llm_guided_unseen_grid_coverage",
                "coverage_targets": [
                    {
                        "variable": "C",
                        "value": "10.0",
                        "unseen": True,
                        "selected_by_llm": True,
                        "option_type": "continuous_grid",
                    }
                ],
            },
        ],
        total_observations=10,
        ensemble_mode=False,
        early_exploration_info={"enabled": True, "bo_round_index": 1, "window": 5},
        stagnation_info={"is_stagnant": True, "stagnation_length": 3, "last_improvement_iteration": 7, "best_result": 0.9},
    )

    lowered = prompt.lower()
    assert "machine learning" in lowered or "hpo" in lowered
    assert "balanced campaign logic" in prompt
    assert "plausibility, trajectory evidence, and information value" in prompt
    assert "[Early Grid Coverage Balance]" in prompt
    assert "Do not select a coverage candidate solely because it is unseen" in prompt
    assert "do not choose an implausible extreme HPO regime only to calibrate uncertainty" in prompt
    assert "Do not prematurely collapse into local optimization" not in prompt
    assert "prefer candidates that open" not in prompt
    assert "chemically" not in lowered
    assert "chemistry" not in lowered


def test_acquisition_selection_prompt_omits_coverage_priority_after_early_window() -> None:
    from core.autobo_prompts import build_acquisition_selection_prompt

    prompt = build_acquisition_selection_prompt(
        reaction_context={"reaction_type": "DAR"},
        top_observations=[],
        bottom_observations=[],
        candidates=[
            {
                "id": 1,
                "candidate": {"ligand_SMILES": "L-best"},
                "predicted_value": 2.0,
                "uncertainty": 1.0,
                "acquisition_value": 4.0,
                "selection_step": 1,
                "selection_mode": "ensemble_reference",
            },
            {
                "id": 2,
                "candidate": {"ligand_SMILES": "L-late-coverage"},
                "predicted_value": 1.0,
                "uncertainty": 2.0,
                "acquisition_value": 3.0,
                "selection_step": 6,
                "selection_mode": "llm_guided_unseen_category_coverage",
                "coverage_targets": [
                    {
                        "variable": "ligand_SMILES",
                        "value": "L-late-coverage",
                        "unseen": True,
                        "selected_by_llm": True,
                    }
                ],
            },
        ],
        total_observations=25,
        ensemble_mode=False,
        early_exploration_info={"enabled": False, "bo_round_index": 6, "window": 5},
    )

    assert "coverage target: ligand_SMILES=L-late-coverage" not in prompt
    assert "this categorical value has never been tested before" not in prompt
    assert "[Early Post-Warm-Start Exploration Guardrail]" not in prompt
    assert "[Early Unseen Exploration Priority]" not in prompt
    assert "prefer a chemically plausible coverage candidate" not in prompt


def test_acquisition_selection_prompt_adds_anti_local_guardrail_during_early_window() -> None:
    from core.autobo_prompts import build_acquisition_selection_prompt

    prompt = build_acquisition_selection_prompt(
        reaction_context={"reaction_type": "DAR"},
        top_observations=[],
        bottom_observations=[],
        candidates=[
            {
                "id": 1,
                "candidate": {"ligand_SMILES": "L-best"},
                "predicted_value": 2.0,
                "uncertainty": 1.0,
                "acquisition_value": 4.0,
                "selection_step": 1,
                "selection_mode": "ensemble_reference",
            }
        ],
        total_observations=11,
        ensemble_mode=False,
        early_exploration_info={"enabled": True, "bo_round_index": 2, "window": 5},
    )

    assert "post-warm-start BO round 2 of 5" in prompt
    assert "Do not prematurely collapse into local optimization" in prompt
    assert "[Early Unseen Exploration Priority]" not in prompt


def test_acquisition_selection_prompt_allows_appended_sixth_candidate() -> None:
    from core.autobo_prompts import build_acquisition_selection_prompt

    candidates = [
        {
            "id": index + 1,
            "candidate": {"ligand_SMILES": f"L{index + 1}"},
            "predicted_value": float(index),
            "uncertainty": 1.0,
            "acquisition_value": float(index),
            "selection_step": index + 1,
            "selection_mode": "ensemble_reference" if index == 0 else "ensemble_candidate",
        }
        for index in range(5)
    ]
    candidates.append(
        {
            "id": 6,
            "candidate": {"ligand_SMILES": "L-coverage"},
            "predicted_value": 1.0,
            "uncertainty": 2.0,
            "acquisition_value": 3.0,
            "selection_step": 6,
            "selection_mode": "llm_guided_unseen_category_coverage",
            "coverage_targets": [
                {
                    "variable": "ligand_SMILES",
                    "value": "L-coverage",
                    "unseen": True,
                    "selected_by_llm": True,
                }
            ],
        }
    )

    prompt = build_acquisition_selection_prompt(
        reaction_context={"reaction_type": "DAR"},
        top_observations=[],
        bottom_observations=[],
        candidates=candidates,
        total_observations=10,
        ensemble_mode=True,
    )

    assert "[Candidates (6-candidate ensemble shortlist" in prompt
    assert "You may only choose #1, #2, #3, #4, #5, or #6." in prompt
    assert "If selected_id is 2, 3, 4, 5, or 6" in prompt


class _PredictOnlySurrogate:
    def fit(self, candidates, values):
        del candidates, values

    def predict(self, candidates):
        return [float(candidate["x"]) for candidate in candidates], [0.5 for _ in candidates]


def _ensemble_candidate_pool(size: int = 10) -> list[dict]:
    return [{"x": index} for index in range(size)]


def test_ensemble_af_isolates_one_failed_af_and_fills_shortlist(monkeypatch) -> None:
    from core import autobo_engine

    def _fake_ranked_af_candidates(**kwargs):
        af_key = kwargs["af_key"]
        if af_key == "qlogei":
            raise RuntimeError("qLogEI exploded")
        offset = 0 if af_key == "qucb" else 5
        return [
            {"candidate": {"x": offset + index}, "af_key": af_key, "af_rank": index + 1}
            for index in range(5)
        ]

    monkeypatch.setattr(autobo_engine, "_build_ranked_af_candidates", _fake_ranked_af_candidates)
    flow = autobo_engine.EnsembleAcquisitionFlow(top_k=5)

    shortlist = flow.propose_candidates(
        active_model=_PredictOnlySurrogate(),
        refit_model_factory=None,
        candidate_pool=_ensemble_candidate_pool(),
        observations=[{"candidate": {"x": 0}, "result": 1.0}],
    )

    assert len(shortlist) == 5
    assert all(item["predicted_value"] is not None for item in shortlist)
    assert all(item["uncertainty"] is not None for item in shortlist)
    assert all("qlogei" not in item["af_sources"] for item in shortlist)
    assert all(item["selection_mode"] != "fallback_random" for item in shortlist)
    assert flow.last_af_slot_filled["qlogei"] == 0
    assert flow.last_af_slot_filled["qucb"] > 0
    assert flow.last_af_slot_filled["ts"] > 0
    assert flow.last_af_failures == [
        {"af_key": "qlogei", "error_type": "RuntimeError", "error": "qLogEI exploded"}
    ]


def test_ensemble_af_single_surviving_af_still_fills_shortlist(monkeypatch) -> None:
    from core import autobo_engine

    def _fake_ranked_af_candidates(**kwargs):
        af_key = kwargs["af_key"]
        if af_key != "ts":
            raise RuntimeError(f"{af_key} unavailable")
        return [
            {"candidate": {"x": index}, "af_key": "ts", "af_rank": index + 1}
            for index in range(5)
        ]

    monkeypatch.setattr(autobo_engine, "_build_ranked_af_candidates", _fake_ranked_af_candidates)
    flow = autobo_engine.EnsembleAcquisitionFlow(top_k=5)

    shortlist = flow.propose_candidates(
        active_model=_PredictOnlySurrogate(),
        refit_model_factory=None,
        candidate_pool=_ensemble_candidate_pool(),
        observations=[{"candidate": {"x": 0}, "result": 1.0}],
    )

    assert len(shortlist) == 5
    assert {source for item in shortlist for source in item["af_sources"]} == {"ts"}
    assert [item["rank"] for item in shortlist] == [1, 2, 3, 4, 5]
    assert flow.last_af_slot_filled == {"qlogei": 0, "qucb": 0, "ts": 5}
    assert {item["af_key"] for item in flow.last_af_failures} == {"qlogei", "qucb"}


def test_ensemble_af_recovers_failed_af_with_cpu_fallback(monkeypatch) -> None:
    from core import autobo_engine

    class _Surrogate(_PredictOnlySurrogate):
        def __init__(self, *, device: str):
            self.device = device

    def _fake_ranked_af_candidates(**kwargs):
        af_key = kwargs["af_key"]
        model = kwargs["active_model"]
        if af_key == "qlogei" and model.device == "cuda":
            raise RuntimeError("cuda posterior failed")
        return [
            {"candidate": {"x": index}, "af_key": af_key, "af_rank": index + 1}
            for index in range(5)
        ]

    monkeypatch.setattr(autobo_engine, "_build_ranked_af_candidates", _fake_ranked_af_candidates)
    flow = autobo_engine.EnsembleAcquisitionFlow(top_k=5)

    shortlist = flow.propose_candidates(
        active_model=_Surrogate(device="cuda"),
        refit_model_factory=None,
        cpu_refit_model_factory=lambda: _Surrogate(device="cpu"),
        candidate_pool=_ensemble_candidate_pool(),
        observations=[{"candidate": {"x": 0}, "result": 1.0}],
    )

    assert len(shortlist) == 5
    assert any("qlogei" in item["af_sources"] for item in shortlist)
    assert flow.last_af_failures == [
        {
            "af_key": "qlogei",
            "error_type": "RuntimeError",
            "error": "cuda posterior failed",
            "recovered_by": "cpu_fallback",
        }
    ]


def test_ensemble_af_all_failed_afs_raise_without_random_fallback(monkeypatch) -> None:
    from core import autobo_engine

    def _fake_ranked_af_candidates(**kwargs):
        raise RuntimeError(f"{kwargs['af_key']} failed")

    monkeypatch.setattr(autobo_engine, "_build_ranked_af_candidates", _fake_ranked_af_candidates)
    flow = autobo_engine.EnsembleAcquisitionFlow(top_k=5)

    with pytest.raises(RuntimeError, match="All ensemble acquisition functions failed") as exc_info:
        flow.propose_candidates(
            active_model=_PredictOnlySurrogate(),
            refit_model_factory=None,
            candidate_pool=_ensemble_candidate_pool(),
            observations=[{"candidate": {"x": 0}, "result": 1.0}],
        )

    assert "qlogei=RuntimeError: qlogei failed" in str(exc_info.value)
    assert "qucb=RuntimeError: qucb failed" in str(exc_info.value)
    assert "ts=RuntimeError: ts failed" in str(exc_info.value)
    assert flow.last_af_slot_filled == {"qlogei": 0, "qucb": 0, "ts": 0}
    assert len(flow.last_af_failures) == 3


def test_unseen_category_coverage_promotes_untried_categorical_levels() -> None:
    from core.autobo_engine import _coverage_records_for_targets, _merge_shortlist_with_coverage

    variables = [
        {"name": "ligand", "type": "categorical", "domain": ["A", "B", "C", "D"]},
        {"name": "solvent", "type": "categorical", "domain": ["S1", "S2"]},
        {"name": "temperature", "type": "continuous", "domain": [90, 120]},
    ]
    observations = [{"candidate": {"ligand": "A", "solvent": "S1", "temperature": 90}, "result": 91.0}]
    candidate_pool = [
        {"ligand": "A", "solvent": "S1", "temperature": 120},
        {"ligand": "B", "solvent": "S1", "temperature": 120},
        {"ligand": "C", "solvent": "S1", "temperature": 120},
        {"ligand": "D", "solvent": "S2", "temperature": 120},
    ]
    current_shortlist = [
        {
            "candidate": candidate_pool[0],
            "predicted_value": 90.0,
            "uncertainty": 1.0,
            "acquisition_value": 91.0,
            "acquisition_value_raw": 91.0,
            "selection_step": 1,
            "selection_mode": "ensemble_reference",
            "rank": 1,
        }
    ]
    unseen = {
        "ligand": [
            {"value": "B", "role": "ligand", "unseen": True, "legal_candidate_count": 1},
        ],
    }

    coverage_records = _coverage_records_for_targets(
        active_model=_CoverageDummyModel(),
        candidate_pool=candidate_pool,
        observations=observations,
        targets=[{"variable": "ligand", "value": "B", "selected_by_llm": True}],
        unseen_options=unseen,
        normal_shortlist=current_shortlist,
        direction="maximize",
        seed=3,
        top_k=2,
        coverage_slots=1,
    )
    injected = _merge_shortlist_with_coverage(current_shortlist, coverage_records, top_k=2, coverage_slots=1)

    assert injected[0]["candidate"]["ligand"] == "A"
    assert injected[1]["selection_mode"] == "llm_guided_unseen_category_coverage"
    assert injected[1]["candidate"]["ligand"] == "B"
    assert injected[1]["coverage_targets"][0]["variable"] == "ligand"


def test_build_random_warm_start_pool_mixed_space_without_dataset_is_seeded_and_bounded() -> None:
    variables = _toy_variables()
    pool_a = _build_random_warm_start_pool(
        variables,
        pool_size=12,
        seed=11,
        observed_keys=set(),
        hard_constraints=[],
        candidate_pool=None,
    )
    pool_b = _build_random_warm_start_pool(
        variables,
        pool_size=12,
        seed=11,
        observed_keys=set(),
        hard_constraints=[],
        candidate_pool=None,
    )
    pool_c = _build_random_warm_start_pool(
        variables,
        pool_size=12,
        seed=12,
        observed_keys=set(),
        hard_constraints=[],
        candidate_pool=None,
    )

    assert pool_a == pool_b
    assert pool_a != pool_c
    assert len(pool_a) == 12
    assert len({candidate_to_key(candidate) for candidate in pool_a}) == 12
    assert all(candidate["ligand"] in {"A", "B", "C", "D"} for candidate in pool_a)
    assert all(candidate["solvent"] in {"S1", "S2"} for candidate in pool_a)
    assert all(60.0 <= float(candidate["temperature"]) <= 120.0 for candidate in pool_a)


def test_build_random_warm_start_pool_excludes_initial_selected_candidates() -> None:
    initial_selected = [_toy_pool()[0], _toy_pool()[1]]
    pool = _build_random_warm_start_pool(
        _toy_variables(),
        pool_size=4,
        seed=13,
        observed_keys=set(),
        hard_constraints=[],
        candidate_pool=_toy_pool(),
        initial_selected=initial_selected,
    )

    pool_keys = {candidate_to_key(candidate) for candidate in pool}
    initial_keys = {candidate_to_key(candidate) for candidate in initial_selected}
    assert len(pool) == 4
    assert not pool_keys & initial_keys


def test_settings_default_initial_doe_size_is_10() -> None:
    assert Settings().initial_doe_size == 10


def test_ablation_switches_default_to_enabled() -> None:
    settings = Settings()
    assert settings.llm_acquisition_strategy_ablation is True
    assert settings.memory_ablation is True
    assert settings.harness_ablation is True


def test_harness_ablation_false_uses_seeded_random_initialization(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.graph._create_llm",
        lambda settings, enable_thinking_override=None, **kwargs: _GraphDummyLLM(),
    )
    settings = Settings(
        initial_doe_size=4,
        max_bo_iterations=8,
        human_input_mode="dataset_auto",
        knowledge_enabled=False,
    )
    settings.harness_ablation = False
    problem = {
        "reaction_type": "toy",
        "target_metric": "yield",
        "optimization_direction": "maximize",
        "budget": 8,
        "variables": _toy_variables(),
        "constraints": [],
    }
    graph = build_chembo_graph(settings)
    initial_state = create_initial_state(problem, settings)
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    list(graph.stream(initial_state, config=config, stream_mode="updates"))
    state = graph.get_state(config).values

    queue = state.get("warm_start_queue", [])
    assert len(queue) == 4
    assert state.get("warm_start_target") == 4
    assert state.get("_warm_start_postmortem_done") is True
    assert all(item["warm_start_category"] == "random_initialization" for item in queue)
    assert state.get("proposal_selected", {}).get("selection_source") == "random_initialization"


def test_llm_acquisition_strategy_ablation_false_selects_qlogei_top1_without_llm() -> None:
    from core.autobo_engine import select_autobo_candidate

    settings = Settings()
    settings.llm_acquisition_strategy_ablation = False
    state = {
        "proposal_shortlist": [
            {"candidate": {"x": "A"}, "autobo_rank": 1, "af_sources": ["qlogei"], "af_ranks": {"qlogei": 1}},
            {"candidate": {"x": "B"}, "autobo_rank": 2, "af_sources": ["qlogei"], "af_ranks": {"qlogei": 2}},
        ],
        "effective_config": {"acquisition_function": "qlog_ei"},
        "last_tool_payload": {"metadata": {"ensemble_af_enabled": False}},
        "memory": {"version": 2, "working": {}, "episodic": [], "semantic": {"nodes": [], "edges": []}},
    }

    def _unexpected_invoke(*args, **kwargs):
        raise AssertionError("LLM selection should not be invoked")

    result = select_autobo_candidate(
        state=state,
        settings=settings,
        llm=_GraphDummyLLM(),
        invoke_json_node=_unexpected_invoke,
    )

    assert result["proposal_selected"]["candidate"] == {"x": "A"}
    assert result["proposal_selected"]["selection_source"] == "autobo_qlogei_top1"
    assert result["proposal_selected"]["rationale"]["selection_mode"] == "qlogei_top1_follow"
    assert result["llm_usage"]["calls"] == 0


def test_memory_ablation_selection_prompt_is_stateless() -> None:
    from core.autobo_engine import select_autobo_candidate

    settings = Settings()
    settings.memory_ablation = False
    captured = {}
    state = {
        "problem_spec": {
            "reaction_type": "toy",
            "target_metric": "yield",
            "optimization_direction": "maximize",
            "variables": _toy_variables(),
        },
        "optimization_direction": "maximize",
        "proposal_shortlist": [
            {
                "candidate": {"ligand": "A", "solvent": "S1", "temperature": 60.0},
                "predicted_value": 1.0,
                "uncertainty": 0.2,
                "acquisition_value": 0.5,
                "autobo_rank": 1,
            },
            {
                "candidate": {"ligand": "B", "solvent": "S2", "temperature": 90.0},
                "predicted_value": 0.9,
                "uncertainty": 0.4,
                "acquisition_value": 0.4,
                "autobo_rank": 2,
            },
        ],
        "observations": [
            {"iteration": 1, "candidate": {"ligand": "SECRET_MEMORY_MARKER"}, "result": 99.0}
        ],
        "hypotheses": [{"id": "H_secret", "text": "SECRET_HYPOTHESIS_MARKER", "status": "active"}],
        "knowledge_deck": {"cards": [{"card_id": "K_secret", "text": "SECRET_KNOWLEDGE_MARKER"}]},
        "memory": {
            "version": 2,
            "working": {"current_focus": "SECRET_MEMORY_MARKER"},
            "episodic": [],
            "semantic": {"nodes": [], "edges": []},
        },
        "best_result": 99.0,
        "best_candidate": {"ligand": "SECRET_MEMORY_MARKER"},
        "effective_config": {"acquisition_function": "qlog_ei"},
        "last_tool_payload": {"metadata": {"ensemble_af_enabled": False}},
    }

    def _capture_invoke(llm, current_state, prompt, default, node_name="", lightweight=False):
        del llm, current_state, default, node_name
        captured["prompt"] = prompt
        captured["lightweight"] = lightweight
        return {"selected_id": 1, "reasoning": "Use top candidate.", "comparison_to_top1": "Accept top candidate.", "selection_mode": "top1_follow"}, [], {"calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "estimated_calls": 0, "input_breakdown": {"system": 0, "campaign_summary": 0, "recent_messages": 0, "prompt": 1}}

    result = select_autobo_candidate(
        state=state,
        settings=settings,
        llm=_GraphDummyLLM(),
        invoke_json_node=_capture_invoke,
    )

    prompt = captured["prompt"]
    assert captured["lightweight"] is True
    assert result["proposal_selected"]["candidate"]["ligand"] == "A"
    assert "SECRET_MEMORY_MARKER" not in prompt
    assert "SECRET_HYPOTHESIS_MARKER" not in prompt
    assert "SECRET_KNOWLEDGE_MARKER" not in prompt
    assert "[Campaign Memory Rules]" not in prompt


def test_harness_ablation_false_disables_unseen_category_coverage() -> None:
    from core.autobo_engine import _unseen_category_coverage_should_run

    settings = Settings(
        autobo_unseen_category_exploration_enabled=True,
        autobo_unseen_category_window=5,
        autobo_unseen_category_slots=1,
    )
    settings.harness_ablation = False
    observations = [{"candidate": {"x": index}, "result": float(index)} for index in range(10)]

    assert not _unseen_category_coverage_should_run(
        settings=settings,
        observations=observations,
        warm_start_target=10,
        ensemble_sur_enabled=False,
        zero_llm_mode=False,
    )


def test_delta_best_supports_fast_interpretation_digest() -> None:
    assert _delta_best(40.0, 55.5, "maximize") == 15.5
    assert _delta_best(40.0, 35.0, "minimize") == 5.0
    assert _delta_best(None, 35.0, "maximize") is None
    assert _delta_best(40.0, None, "maximize") is None


@pytest.mark.parametrize(
    ("budget", "expected_target"),
    [
        (35, 17),
        (40, 20),
        (50, 20),
        (1, 1),
    ],
)
def test_plan_warm_start_respects_budget_caps(budget: int, expected_target: int) -> None:
    settings = Settings(initial_doe_size=20, max_bo_iterations=max(budget, 1))
    problem_spec = _example_problem("dar")
    problem_spec["budget"] = budget
    state = create_initial_state(problem_spec, settings)
    state["knowledge_deck"] = {"cards": _sample_knowledge_cards(), "build_summary": {"coverage_level": "partial"}}

    updates = plan_warm_start(
        state,
        settings,
        _GraphDummyLLM(),
        invoke_tool_loop=_invoke_tool_loop_factory(),
        extract_last_json=_direct_extract_last_json,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )

    assert updates["warm_start_target"] == expected_target
    assert len(updates["warm_start_queue"]) == expected_target
    assert [item["warm_start_category"] for item in updates["warm_start_queue"]] == ["llm_direct"] * expected_target


def test_plan_warm_start_all_llm_direct_is_deterministic() -> None:
    settings = Settings(initial_doe_size=20, max_bo_iterations=40)
    problem_spec = _example_problem("dar")
    state = create_initial_state(problem_spec, settings)
    state["knowledge_deck"] = {"cards": _sample_knowledge_cards(), "build_summary": {"coverage_level": "partial"}}

    first = plan_warm_start(
        state,
        settings,
        _GraphDummyLLM(),
        invoke_tool_loop=_invoke_tool_loop_factory(),
        extract_last_json=_direct_extract_last_json,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )
    second = plan_warm_start(
        state,
        settings,
        _GraphDummyLLM(),
        invoke_tool_loop=_invoke_tool_loop_factory(),
        extract_last_json=_direct_extract_last_json,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )

    assert first["warm_start_queue"] == second["warm_start_queue"]
    categories = [item["warm_start_category"] for item in first["warm_start_queue"]]
    assert categories == ["llm_direct"] * 20
    assert len({candidate_to_key(item["candidate"]) for item in first["warm_start_queue"]}) == 20


def test_plan_warm_start_retries_invalid_direct_points_then_underfills_without_random() -> None:
    settings = Settings(initial_doe_size=4, max_bo_iterations=40)
    problem_spec = _example_problem("dar")
    state = create_initial_state(problem_spec, settings)
    state["knowledge_deck"] = {"cards": [], "build_summary": {"coverage_level": "gap"}}
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    assert oracle is not None
    calls = {"direct": 0}

    def _invoke_tool_loop_invalid_then_sparse(llm, state, prompt, tool_map, max_turns=6, **kwargs):
        del llm, state, tool_map, max_turns, kwargs
        if "Select high-value direct warm-start experiments" not in prompt:
            raise AssertionError(f"Unexpected prompt:\n{prompt}")
        calls["direct"] += 1
        if calls["direct"] == 1:
            selections = [
                {
                    "variables": {column: "not-a-real-level" for column in oracle.feature_columns},
                    "reasoning": "Invalid point that should trigger validation feedback.",
                    "confidence": 0.2,
                }
            ]
        else:
            selections = [
                {
                    "variables": dict(oracle.candidates[0]),
                    "reasoning": "Valid replacement direct seed after validation feedback.",
                    "confidence": 0.8,
                }
            ]
        return [
            HumanMessage(content=prompt),
            AIMessage(
                content=json.dumps(
                    {
                        "strategy_summary": "Retry invalid direct warm-start points.",
                        "selections": selections,
                    }
                )
            ),
        ], "", _usage()

    updates = plan_warm_start(
        state,
        settings,
        _GraphDummyLLM(),
        invoke_tool_loop=_invoke_tool_loop_invalid_then_sparse,
        extract_last_json=_direct_extract_last_json,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )

    assert calls["direct"] == 2
    assert len(updates["warm_start_queue"]) == 1
    assert [item["warm_start_category"] for item in updates["warm_start_queue"]].count("llm_direct") == 1
    assert [item["warm_start_category"] for item in updates["warm_start_queue"]].count("random") == 0
    assert len({candidate_to_key(item["candidate"]) for item in updates["warm_start_queue"]}) == 1
    assert updates["warm_start_target"] == 1
    assert "underfilled=true missing=3" in updates["llm_reasoning_log"][-1]


def test_plan_warm_start_repairs_unparseable_direct_response() -> None:
    settings = Settings(initial_doe_size=4, max_bo_iterations=40)
    problem_spec = _example_problem("dar")
    state = create_initial_state(problem_spec, settings)
    state["knowledge_deck"] = {"cards": [], "build_summary": {"coverage_level": "gap"}}
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    assert oracle is not None
    calls = {"direct": 0, "repair": 0}

    def _invoke_tool_loop_needs_repair(llm, state, prompt, tool_map, max_turns=6, **kwargs):
        del llm, state, tool_map, max_turns, kwargs
        if "Select high-value direct warm-start experiments" not in prompt:
            raise AssertionError(f"Unexpected prompt:\n{prompt}")
        if "Return strict JSON only. Do not include prose" in prompt:
            calls["repair"] += 1
            target = _parse_direct_target(prompt)
            content = json.dumps(
                {
                    "strategy_summary": "Repair the draft into strict JSON.",
                    "selections": [
                        {
                            "variables": dict(candidate),
                            "reasoning": f"Recovered direct warm-start seed {index + 1}.",
                            "confidence": 0.8,
                            "purpose": "explore" if index else "exploit",
                        }
                        for index, candidate in enumerate(oracle.candidates[:target])
                    ],
                }
            )
        else:
            calls["direct"] += 1
            content = (
                "I will think step by step before giving JSON.\n"
                '{"strategy_summary":"Draft","selections":[{"variables":{"broken":"response"}}'
            )
        return [HumanMessage(content=prompt), AIMessage(content=content)], "", _usage()

    updates = plan_warm_start(
        state,
        settings,
        _GraphDummyLLM(),
        invoke_tool_loop=_invoke_tool_loop_needs_repair,
        extract_last_json=_extract_last_json,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )

    assert calls["direct"] == 1
    assert calls["repair"] == 1
    assert len(updates["warm_start_queue"]) == 4
    assert [item["warm_start_category"] for item in updates["warm_start_queue"]].count("llm_direct") == 4
    assert len({candidate_to_key(item["candidate"]) for item in updates["warm_start_queue"]}) == 4


def test_warm_start_direct_prompts_request_full_llm_queue_with_chemically_plausible_coverage() -> None:
    context = {
        "warm_start_target": 4,
        "knowledge_cards_text": "[Knowledge Cards]\nKC1: ligand prior",
        "knowledge_cards": [{"card_id": "KC1"}],
    }
    structured_spec = {
        "space_description": "- ligand: L1, L2\n- temperature: continuous in [80, 120]",
        "output_schema": '{"variables": {"ligand": "L1", "temperature": "100"}, "reasoning": "..."}',
    }

    structured_prompt = _build_warm_start_direct_structured_prompt(
        context=context,
        structured_spec=structured_spec,
        target=4,
        total_direct_target=4,
        validation_feedback="",
        accepted_records=[],
    )
    candidate_prompt = _build_warm_start_direct_candidate_pool_prompt(
        context=context,
        candidates=[
            {"id": 1, "candidate": {"ligand": "L1", "temperature": 100}},
            {"id": 2, "candidate": {"ligand": "L2", "temperature": 110}},
        ],
        target=2,
        total_direct_target=2,
        validation_feedback="",
        accepted_records=[],
    )

    for prompt in (structured_prompt, candidate_prompt):
        assert "selecting the entire warm-start queue; there is no random fill" in prompt
        assert "roughly 40% exploit / 60% explore" in prompt
        assert "Do not choose chemically poor" in prompt
        assert "chemically suitable temperature, concentration, flow, ratio" in prompt
        assert "Do not mechanically spread low/mid/high values" in prompt

    assert '"purpose"' in structured_prompt
    assert '"purpose_by_id"' in candidate_prompt
    assert "[exploit]" in candidate_prompt and "[explore]" in candidate_prompt


def test_extract_json_from_response_handles_embedded_prose_and_trailing_text() -> None:
    text = """
    Some reasoning before the answer.

    ```json
    {"strategy_summary": "Recovered", "selections": [{"variables": {"temperature": "120"}}]}
    ```

    Extra commentary after the code block.
    """

    parsed = _extract_json_from_response(text)

    assert parsed is not None
    assert parsed["strategy_summary"] == "Recovered"


def test_initial_prior_writer_requires_four_hypotheses_and_warm_start_actionability() -> None:
    problem = _example_problem("ocm")
    raw_cards = [
        {
            "text": "Methane activation and oxygen availability jointly control coupling versus complete oxidation selectivity.",
            "card_type": "mechanism",
            "scope": "target",
            "confidence": 0.58,
            "targets": [],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Primary redox metals tune lattice oxygen reactivity and therefore methane activation rates.",
            "card_type": "reagent_property",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["M1"],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Alkali promoters can increase surface basicity and suppress nonselective oxidation pathways.",
            "card_type": "reagent_property",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["M2"],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Intermediate oxygen availability should avoid both conversion starvation and excessive total oxidation.",
            "card_type": "operating_window",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["O2_flow"],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Very high operating severity can promote complete oxidation and mobile promoter loss.",
            "card_type": "failure_mode",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["Temp"],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Support identity can interact with alkali promoters through acid base site density.",
            "card_type": "interaction",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["Support", "M2"],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Methane rich feeds should improve C2 plus selectivity when oxygen activation remains sufficient.",
            "card_type": "hypothesis",
            "scope": "target",
            "confidence": 0.55,
            "targets": [],
            "testable_prediction": "Methane rich warm starts outperform oxygen rich conditions for similar catalysts.",
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Sodium tungstate manganese catalysts should outperform support only controls in early warm starts.",
            "card_type": "hypothesis",
            "scope": "target",
            "confidence": 0.55,
            "targets": [],
            "testable_prediction": "Mn Na W catalysts beat blank or support only rows at matched conditions.",
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Silica supported tungstate systems should reveal productive regions before reducible oxide supports.",
            "card_type": "hypothesis",
            "scope": "target",
            "confidence": 0.55,
            "targets": [],
            "testable_prediction": "SiO2 tungstate rows rank above CeO2 or Nb2O5 support only rows.",
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Moderate temperature should balance methane conversion and C2 plus selectivity better than extremes.",
            "card_type": "hypothesis",
            "scope": "target",
            "confidence": 0.55,
            "targets": [],
            "testable_prediction": "Middle temperature candidates outperform low and high temperature matched rows.",
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Lower oxygen fractions can reduce over oxidation when catalyst oxygen mobility is high.",
            "card_type": "operating_window",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["O2_flow"],
            "actionable_for": ["select_candidate"],
        },
        {
            "text": "Basic supports may extend methyl radical lifetimes and increase coupling probability.",
            "card_type": "reagent_property",
            "scope": "target",
            "confidence": 0.55,
            "targets": ["Support"],
            "actionable_for": ["select_candidate"],
        },
    ]

    def _invoke_json(llm, system_prompt, user_prompt, default):
        del llm, system_prompt, user_prompt, default
        return {"cards": raw_cards, "global_notes": ""}, _usage()

    cards, _ = write_initial_priors(problem, Settings(), _GraphDummyLLM(), _invoke_json)

    assert len(cards) == 12
    assert sum(card["card_type"] == "hypothesis" for card in cards) >= 4
    assert all("warm_start" in card["actionable_for"] for card in cards)


def test_prior_writer_prompt_calibrates_confidence_and_downgrades_specific_claims() -> None:
    system_prompt, user_prompt = build_prior_writer_prompt(
        {
            "reaction_type": "GENERIC",
            "reaction": {"family": "GENERIC"},
            "variables": [{"name": "ligand", "role": "ligand", "type": "categorical", "domain": ["L1"]}],
        },
        "generic_fallback",
    )

    assert "CONFIDENCE CALIBRATION" in system_prompt
    assert "UNVERIFIED by any campaign observation" in system_prompt
    assert "0.40-0.45" in system_prompt
    assert "0.35-0.39" in system_prompt
    assert "0.30-0.34" in system_prompt
    assert "Do not default to 0.40" in system_prompt
    assert "0.45+ band is reserved" in system_prompt
    assert "LANGUAGE DISCIPLINE" in system_prompt
    assert "Card.text must NEVER use absolute words" in system_prompt
    assert "DOWNGRADE RULE" in system_prompt
    assert "specific variable value being better or worse" in system_prompt
    assert "card_type hypothesis with testable_prediction" in system_prompt
    assert "testable_prediction" in user_prompt


def test_active_deck_preserves_hypotheses_for_warm_start() -> None:
    cards = [
        {
            "card_id": f"kc_constraint_{index}",
            "text": f"Only propose valid dataset backed candidates during initialization phase {index}.",
            "card_type": "constraint",
            "scope": "target",
            "confidence": 1.0,
            "status": "active",
            "actionable_for": ["warm_start"],
        }
        for index in range(5)
    ]
    cards += [
        {
            "card_id": f"kc_prior_{index}",
            "text": f"Catalyst support and promoter chemistry provide useful warm start evidence {index}.",
            "card_type": "reagent_property",
            "scope": "target",
            "confidence": 0.6,
            "status": "active",
            "actionable_for": ["warm_start"],
        }
        for index in range(8)
    ]
    cards += [
        {
            "card_id": f"kc_hypothesis_{index}",
            "text": f"Warm start should prioritize plausible catalysts over support only controls {index}.",
            "card_type": "hypothesis",
            "scope": "target",
            "confidence": 0.6,
            "status": "active",
            "actionable_for": ["warm_start"],
            "testable_prediction": "Plausible catalysts beat support only controls in early observations.",
        }
        for index in range(4)
    ]

    selected = _rank_and_filter_cards(cards, max_cards=12, min_hypotheses=4)

    assert len(selected) == 12
    assert sum(card["card_type"] == "hypothesis" for card in selected) == 4
    assert all("warm_start" in card["actionable_for"] for card in selected)


def test_knowledge_card_used_by_improved_result_is_not_automatically_supported() -> None:
    card = create_knowledge_card(
        "Bulky ligands can affect catalyst speciation in this reaction family.",
        "reagent_property",
        targets=["ligand"],
        card_id="kc_ligand",
    )
    deck = {"cards": [card], "build_summary": {}}

    updated = _update_knowledge_deck_after_interpretation(
        knowledge_deck=deck,
        hypotheses=[],
        latest_observation={
            "iteration": 3,
            "candidate": {"ligand": "L1"},
            "result": 70.0,
            "metadata": {"best_before_result": 50.0},
        },
        parsed={
            "supported_hypotheses": [],
            "refuted_hypotheses": [],
            "knowledge_conflict": {"has_conflict": False, "conflicting_cards": [], "reason": ""},
        },
        maintenance_new_rules=[],
        direction="maximize",
        problem_spec={"variables": [{"name": "ligand"}]},
    )

    validation = updated["cards"][0]["validation"]
    assert validation["used_count"] == 1
    assert validation["supported_count"] == 0
    assert validation["contradicted_count"] == 0


def test_knowledge_card_conflicting_card_increments_contradiction() -> None:
    card = create_knowledge_card(
        "Polar solvents usually improve catalyst compatibility.",
        "reagent_property",
        targets=["solvent"],
        card_id="kc_solvent",
    )
    deck = {"cards": [card], "build_summary": {}}

    updated = _update_knowledge_deck_after_interpretation(
        knowledge_deck=deck,
        hypotheses=[],
        latest_observation={
            "iteration": 4,
            "candidate": {"solvent": "S1"},
            "result": 30.0,
            "metadata": {"best_before_result": 50.0},
        },
        parsed={
            "supported_hypotheses": [],
            "refuted_hypotheses": [],
            "knowledge_conflict": {"has_conflict": True, "conflicting_cards": ["kc_solvent"], "reason": "Result moved against card."},
        },
        maintenance_new_rules=[],
        direction="maximize",
        problem_spec={"variables": [{"name": "solvent"}]},
    )

    validation = updated["cards"][0]["validation"]
    assert validation["used_count"] == 1
    assert validation["supported_count"] == 0
    assert validation["contradicted_count"] == 1


def test_hypothesis_card_support_requires_explicit_supported_hypothesis() -> None:
    card = create_knowledge_card(
        "Electron-rich ligands should improve early outcomes when oxidative addition is limiting.",
        "hypothesis",
        confidence=0.5,
        testable_prediction="Runs with electron-rich ligands should outperform otherwise comparable less donating ligands.",
        card_id="kc_hypothesis",
    )
    deck = {"cards": [card], "build_summary": {}}
    hypotheses = [{"id": "H1", "text": "Electron-rich ligands help.", "source_card_id": "kc_hypothesis"}]

    updated = _update_knowledge_deck_after_interpretation(
        knowledge_deck=deck,
        hypotheses=hypotheses,
        latest_observation={
            "iteration": 5,
            "candidate": {"ligand": "L2"},
            "result": 80.0,
            "metadata": {"best_before_result": 60.0},
        },
        parsed={
            "supported_hypotheses": ["H1"],
            "refuted_hypotheses": [],
            "knowledge_conflict": {"has_conflict": False, "conflicting_cards": [], "reason": ""},
        },
        maintenance_new_rules=[],
        direction="maximize",
        problem_spec={"variables": [{"name": "ligand"}]},
    )

    validation = updated["cards"][0]["validation"]
    assert validation["used_count"] == 0
    assert validation["supported_count"] == 1
    assert validation["contradicted_count"] == 0
    assert updated["cards"][0]["confidence"] == pytest.approx(0.6)


def test_partial_warm_start_json_recovery_handles_truncated_array() -> None:
    text = (
        'Some analysis that should be ignored {"strategy_summary":"x","selections":['
        '{"cat":"32","Temp":800,"CT":0.38,"ar_level":"low","ch4_o2_ratio":2,'
        '"reasoning":"classic catalyst","confidence":0.8},'
        '{"cat":"28","Temp":850,"CT":0.5,"ar_level":"mid","ch4_o2_ratio":1,'
        '"reasoning":"support contrast","confidence":0.75}'
    )

    recovered = _extract_partial_direct_selection_payloads(text)

    assert [item["cat"] for item in recovered] == ["32", "28"]


def test_plan_warm_start_reduces_target_when_dataset_candidates_are_insufficient() -> None:
    settings = Settings(initial_doe_size=6, max_bo_iterations=40)
    problem_spec = _example_problem("dar")
    state = create_initial_state(problem_spec, settings)
    oracle = DatasetOracle.from_problem_spec(problem_spec)
    assert oracle is not None
    state["observations"] = [
        {"candidate": dict(candidate), "result": float(index)}
        for index, candidate in enumerate(oracle.candidates[:-3])
    ]

    updates = plan_warm_start(
        state,
        settings,
        _GraphDummyLLM(),
        invoke_tool_loop=_invoke_tool_loop_factory(),
        extract_last_json=_direct_extract_last_json,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )

    assert updates["warm_start_target"] == 3
    assert len(updates["warm_start_queue"]) == 3
    assert "Warm-start target reduced from 6 to 3" in updates["campaign_summary"]


@pytest.mark.parametrize("problem_name", ["dar", "ocm"])
def test_graph_warm_start_smoke_uses_llm_direct_queue(problem_name: str, monkeypatch) -> None:
    state = _run_to_first_interrupt(
        monkeypatch,
        _example_problem(problem_name),
        cards=_sample_knowledge_cards(),
    )
    oracle = DatasetOracle.from_problem_spec(state["problem_spec"])

    assert "kb_context" not in state
    assert "kb_priors" not in state
    assert state["knowledge_deck"]["cards"]
    assert "retrieval_artifacts" not in state
    assert len(state["warm_start_queue"]) == 20
    assert state["warm_start_active"] is True
    assert state["proposal_selected"]["selection_source"] == "warm_start_queue"
    assert oracle is not None
    assert oracle.candidate_exists(state["proposal_selected"]["candidate"])
    categories = [item["warm_start_category"] for item in state["warm_start_queue"]]
    assert categories[0] == "llm_direct"
    assert set(categories) == {"llm_direct"}


def test_interpret_warm_start_result_stays_lightweight() -> None:
    settings = Settings(initial_doe_size=20, max_bo_iterations=40)
    problem_spec = _example_problem("dar")
    state = create_initial_state(problem_spec, settings)
    state["observations"] = [
        {
            "iteration": 1,
            "candidate": {
                "base_SMILES": problem_spec["variables"][0]["domain"][0],
                "ligand_SMILES": problem_spec["variables"][1]["domain"][0],
                "solvent_SMILES": problem_spec["variables"][2]["domain"][0],
                "concentration": problem_spec["variables"][3]["domain"][0],
                "temperature": problem_spec["variables"][4]["domain"][0],
            },
            "result": 42.0,
            "metadata": {"selection_source": "warm_start_queue", "best_before_result": 10.0},
        }
    ]
    memory_manager = _memory_manager_from_state(state, settings)
    semantic_before = len(memory_manager.to_dict()["semantic"]["nodes"])

    updates = interpret_warm_start_result(
        state,
        settings,
        _GraphDummyLLM(),
        memory_manager=memory_manager,
        build_context_messages=lambda state, **kwargs: ([HumanMessage(content="context")], "", {"system": 0, "campaign_summary": 0, "recent_messages": 0, "prompt": 0}),
        invoke_llm_with_tracking=lambda llm, messages: (_GraphDummyLLM().invoke(messages), _usage()),
        extract_json_from_response=lambda text: json.loads(text),
        message_text=lambda message: message.content,
        state_messages=_state_messages_identity,
        updated_campaign_summary=_updated_campaign_summary_stub,
        attach_llm_usage=_attach_llm_usage_stub,
    )

    assert len(updates["memory"]["semantic"]["nodes"]) == semantic_before
    assert "[interpret_results:warm_start_light]" in updates["llm_reasoning_log"][-2]
    assert "last_llm_usage" not in updates
    assert "hypotheses" not in updates


def test_run_warm_start_postmortem_only_uses_warm_start_observations_and_updates_memory() -> None:
    settings = Settings(initial_doe_size=20, max_bo_iterations=40)
    problem_spec = _example_problem("dar")
    state = create_initial_state(problem_spec, settings)
    state["hypotheses"] = [{"id": "H1", "text": "Hotter starts help.", "status": "active"}]
    state["iteration"] = 3
    state["observations"] = [
        {
            "iteration": 1,
            "candidate": {"temperature": "90"},
            "result": 10.0,
            "metadata": {"selection_source": "warm_start_queue"},
        },
        {
            "iteration": 2,
            "candidate": {"temperature": "120"},
            "result": 25.0,
            "metadata": {"selection_source": "warm_start_queue"},
        },
        {
            "iteration": 3,
            "candidate": {"temperature": "105"},
            "result": 22.0,
            "metadata": {"selection_source": "autobo"},
        },
    ]
    captured_prompt = {"text": ""}
    memory_manager = _memory_manager_from_state(state, settings)

    def _invoke_llm_with_tracking(llm, messages):
        del llm
        captured_prompt["text"] = messages[-1].content
        return (
            AIMessage(
                content=json.dumps(
                    {
                        "batch_interpretation": "Warm-start established that hotter conditions outperform colder ones.",
                        "supported_hypotheses": ["H1"],
                        "refuted_hypotheses": [],
                        "key_patterns": ["High temperature is favorable."],
                        "semantic_rules": [
                            {
                                "rule_type": "chemical_effect",
                                "statement": "Higher temperature improved early DAR performance.",
                                "variables": ["temperature"],
                                "conditions": {},
                                "confidence": 0.72,
                            }
                        ],
                    }
                )
            ),
            _usage(),
        )

    payload = run_warm_start_postmortem(
        state,
        settings,
        _GraphDummyLLM(),
        None,
        memory_manager=memory_manager,
        build_context_messages=lambda state, **kwargs: ([HumanMessage(content="context")], "", {"system": 0, "campaign_summary": 0, "recent_messages": 0, "prompt": 0}),
        invoke_llm_with_tracking=_invoke_llm_with_tracking,
        extract_json_from_response=lambda text: json.loads(text),
        message_text=lambda message: message.content,
        compute_convergence_state=compute_convergence_state,
        update_hypothesis_statuses=_update_hypothesis_statuses,
        merge_llm_usage=_merge_llm_usage,
    )

    assert '"selection_source":"warm_start_queue"' in captured_prompt["text"]
    assert '"selection_source":"autobo"' not in captured_prompt["text"]
    assert "[CAUSAL ATTRIBUTION RULES FOR WARM-START / DOE ANALYSIS]" in captured_prompt["text"]
    assert "Compare top and bottom performers by variable combinations" in captured_prompt["text"]
    assert "metadata.evidence_basis" in captured_prompt["text"]
    assert "permanently excluded based solely on sparse warm-start data" in captured_prompt["text"]
    assert payload["batch_interpretation"].startswith("Warm-start established")
    assert payload["hypotheses"][0]["status"] == "supported"
    assert payload["added_rule_count"] == 1
    assert payload["memory"]["semantic"]["nodes"]


def test_interpret_results_prompt_uses_general_causal_discipline_without_pairwise_previous_block() -> None:
    source = (Path(__file__).resolve().parents[1] / "core" / "graph.py").read_text(encoding="utf-8")

    assert "[CAUSAL ATTRIBUTION DISCIPLINE]" in source
    assert "find meaningful comparators" in source
    assert "If no isolated or near-isolated comparison exists" in source
    assert "_changed_vars" not in source
    assert "variables changed simultaneously vs the previous" not in source


def test_llm_consolidation_prompt_exposes_causal_attributions() -> None:
    manager = MemoryManager(capacity=16, consolidation_every_n=2, enable_llm_consolidation=True)
    observations: list[dict] = []
    base_state = {
        "problem_spec": {
            "reaction_type": "TOY",
            "target_metric": "yield",
            "optimization_direction": "maximize",
            "variables": _toy_variables(),
        },
        "optimization_direction": "maximize",
        "effective_config": {"acquisition_function": "qlog_ei"},
        "hypotheses": [],
    }
    best_before = None
    for iteration, candidate, result in [
        (1, {"ligand": "A", "solvent": "S1", "temperature": 60.0}, 42.0),
        (2, {"ligand": "B", "solvent": "S2", "temperature": 90.0}, 58.0),
    ]:
        observations.append(
            {
                "iteration": iteration,
                "candidate": candidate,
                "result": result,
                "metadata": {
                    "predicted_value": 50.0,
                    "best_before_result": best_before,
                    "selection_source": "llm_shortlist",
                },
            }
        )
        state = {**base_state, "iteration": iteration, "observations": list(observations)}
        manager.record_result(
            state,
            {
                "interpretation": f"Result {iteration} recorded.",
                "supported_hypotheses": [],
                "refuted_hypotheses": [],
                "archived_hypotheses": [],
                "reflection": "Track joint configuration effects.",
                "knowledge_conflict": {"has_conflict": False, "conflicting_cards": [], "reason": ""},
                "working_focus": "Avoid over-attributing sparse evidence.",
            },
        )
        best_before = max(item["result"] for item in observations)

    captured = {"prompt": ""}

    class _Adapter:
        def invoke_json(self, prompt, default):
            captured["prompt"] = prompt
            return default, {}

    manager.run_maintenance(
        {**base_state, "iteration": 2, "observations": list(observations)},
        trigger="reflection",
        llm_adapter=_Adapter(),
    )

    assert '"causal_attributions":' in captured["prompt"]
    assert '"changed_variable_count":' in captured["prompt"]
    assert '"has_multi_variable_change_evidence":true' in captured["prompt"]
    assert "A single-variable chemical_effect rule requires isolated or near-isolated evidence" in captured["prompt"]
    assert '"metadata":' in captured["prompt"]
    assert '"confound_note"' in captured["prompt"]
