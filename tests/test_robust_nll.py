import numpy as np
import pytest
from scipy.special import logsumexp
from config.settings import Settings
from core import autobo_engine as engine
from core.robust_surrogate import RobustNLLSurrogate, update_weights, consume_predictions


def test_weights_convex_optimum_and_kl_control():
    logp = np.tile([0., -2., -3., -4.], (5, 1))
    old = np.full(4, .25)
    w = update_weights(logp, old)
    slow = update_weights(logp, old, 10.)
    assert .25 < slow[0] < w[0] < .99
    assert w.sum() == pytest.approx(1.)
    r = logsumexp(logp + np.log(old), axis=1) - logsumexp(logp + np.log(w), axis=1)
    assert max((r.sum() - r) / 4) + np.sum(w * np.log(w / old)) <= 0
    # Compare the two-expert solve to an independent dense grid.
    data = np.array([[0., -2.], [-1., 0.], [0., -3.], [-.5, 0.], [0., -1.]])
    w2 = update_weights(data, [.5, .5])
    grid = np.linspace(.001, .999, 10001)
    ws = np.column_stack([grid, 1-grid])
    r = logsumexp(data + np.log(.5), axis=1)[:, None] - logsumexp(data[:, None, :] + np.log(ws), axis=2)
    losses = np.max((r.sum(axis=0) - r) / 4, axis=0) + np.sum(ws * np.log(ws/.5), axis=1)
    assert w2[0] == pytest.approx(grid[np.argmin(losses)], abs=2e-4)


def test_one_lucky_observation_cannot_drive_update():
    data = np.array([[0., -10.]] + [[0., 0.]] * 4)
    np.testing.assert_allclose(update_weights(data, [.5, .5]), [.5, .5])
    np.testing.assert_allclose(update_weights(np.zeros((5, 4)), [.25]*4), [.25]*4)
    assert np.isfinite(update_weights([[-1e8, 0]], [.5, .5])).all()


class Model:
    def __init__(self, mean):
        self.mean = mean
    def fit(self, candidates, y):
        self.trained = list(candidates)
    def predict(self, candidates):
        return np.full(len(candidates), self.mean), np.ones(len(candidates))


def test_mixture_moments_and_exact_ei():
    mixture = RobustNLLSurrogate([lambda: Model(-2), lambda: Model(2)], [.25, .75])
    mixture.fit([], [])
    mu, std = mixture.predict([{}])
    assert mu[0] == pytest.approx(1)
    assert std[0] == pytest.approx(2)
    score = engine._score_candidate_pool(surrogate=mixture, candidate_pool=[{}], best_f_scaled=0, y_mean=0, y_std=1, direction='minimize', seed=0)
    expected = .25*engine._analytic_ei(np.array([-2]), np.array([1]), 0) + .75*engine._analytic_ei(np.array([2]), np.array([1]), 0)
    np.testing.assert_allclose(np.exp(score['acquisition']), expected)
    assert score['pred_mean'][0] == -1


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_runtime_queue_and_prequential_predictions(monkeypatch, direction):
    fits = []
    def factory(spec, *args, **kwargs):
        fits.append(spec.model_id)
        return Model(0 if spec.model_id == 'catboost' else 3)
    monkeypatch.setattr(engine, '_create_surrogate_from_spec', factory)
    monkeypatch.setattr(engine, '_parallel_loocv_evaluate', lambda **kwargs: pytest.fail('legacy evaluation ran'))
    settings = Settings(ensemble_af=False, ensemble_sur=True, zero_llm_ablation_enabled=True,
                        autobo_unseen_category_exploration_enabled=False, shortlist_top_k=1, autobo_acq_top_k=1)
    state = dict(iteration=10, warm_start_target=10, optimization_direction=direction,
                 problem_spec={'variables':[{'name':'x','type':'categorical','domain':list(range(25))}], 'dataset':{}},
                 observations=[{'candidate':{'x':i},'result':0.} for i in range(10)], performance_log=[])
    for t in range(8):
        out = engine.run_autobo_iteration(state=state, settings=settings, llm=None, invoke_json_node=lambda *a, **k: ({}, [], {}))
        cache = out['autobo_state']['robust_nll']
        assert len(cache['history']) == min(t, 5)
        assert len(cache['pending']) == 1
        assert len(cache['model_ids']) == 4 and 'deep_ensemble' not in cache['model_ids']
        assert out['payload']['resolved_components']['surrogate_model'] == 'robust_nll'
        if t:
            assert cache['weights'][-1] > .25
            # Cached forecasts were made before fitting the observed result.
            np.testing.assert_allclose(cache['history'][-1], [-4.5-.5*np.log(2*np.pi)]*3 + [-.5*np.log(2*np.pi)])
        state['autobo_state'] = out['autobo_state']
        repeat = engine.run_autobo_iteration(state=state, settings=settings, llm=None, invoke_json_node=lambda *a, **k: ({}, [], {}))
        assert repeat['autobo_state']['robust_nll']['weights'] == cache['weights']
        state['observations'].append({'candidate':out['payload']['shortlist'][0]['candidate'], 'result':0.})
        state['iteration'] += 1
    assert 'catboost' in fits


def test_defaults_and_legacy_gates():
    settings = Settings()
    assert settings.autobo_surrogate_strategy == 'robust_nll'
    specs = engine.surrogate_specs_from_ids(settings.autobo_surrogate_pool)
    assert len(specs) == 4
    assert len(engine.get_eligible_surrogate_specs(specs, 10, settings)) == 4
    settings.autobo_surrogate_strategy = 'legacy'
    assert len(engine.get_eligible_surrogate_specs(specs, 10, settings)) == 3


def test_queue_discards_oldest_and_only_scores_new_observations():
    rows = [{'candidate': {'x': i}, 'result': float(i)} for i in range(3)]
    cache = {'history': [[100.], [101.], [102.], [103.], [104.]], 'observed_count': 1,
             'pending': {str(i): {'mean': [0.], 'std': [1.]} for i in range(3)}}
    result = consume_predictions(cache, rows, lambda x: str(x['x']), 'maximize')
    assert result[:3] == [[102.], [103.], [104.]]
    np.testing.assert_allclose(np.array(result[-2:]).ravel(), [-.5-.5*np.log(2*np.pi), -2.-.5*np.log(2*np.pi)])
    assert cache['history'][0] == [100.]


def test_solver_failure_preserves_previous_weights(monkeypatch):
    from core import robust_surrogate
    from types import SimpleNamespace
    monkeypatch.setattr(robust_surrogate, 'minimize', lambda *a, **k: SimpleNamespace(success=False))
    np.testing.assert_allclose(update_weights([[0., -10.]], [.4, .6]), [.4, .6])


def test_log_ei_preserves_extreme_tail_ranking():
    from core.robust_surrogate import gaussian_log_ei
    from scipy.integrate import quad
    a = np.array([2., 40., 51., 100., 10000.])
    actual = gaussian_log_ei(-a, np.ones(len(a)), 0.)
    expected = []
    for value in a:
        integral = quad(lambda t: t*np.exp(-t - t*t/(2*value*value)), 0, np.inf)[0]
        expected.append(-.5*value*value - .5*np.log(2*np.pi) - 2*np.log(value) + np.log(integral))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    assert np.all(np.diff(actual) < 0)
    normal_z = np.linspace(-1, 10, 50)
    np.testing.assert_allclose(np.exp(gaussian_log_ei(normal_z, np.ones(50), 0)), engine._analytic_ei(normal_z, np.ones(50), 0))


def test_missing_forecasts_age_out_instead_of_reusing_stale_evidence():
    rows = [{'candidate': {'x': i}, 'result': i} for i in range(5)]
    cache = {'history': [[0., -10.]]*5, 'observed_count': 0, 'pending': {}}
    history = consume_predictions(cache, rows, str, 'maximize')
    assert history == [None]*5
    np.testing.assert_array_equal(update_weights(history, [.4, .6]), [.4, .6])


def test_log_density_common_offsets_do_not_change_weights():
    logp = np.array([[0., -2.], [-1., 0.], [0., -3.], [-.5, 0.], [0., -1.]])
    expected = update_weights(logp, [.5, .5])
    np.testing.assert_allclose(update_weights(logp - 1e12, [.5, .5]), expected, atol=1e-7)


def test_catboost_two_candidate_prediction_layout():
    from pools.component_pools import CatBoostSurrogate
    from types import SimpleNamespace
    surrogate = CatBoostSurrogate([{'name': 'x', 'type': 'continuous', 'domain': [0, 1]}])
    surrogate._model = SimpleNamespace(predict=lambda *a, **kw: np.array([[10., 4.], [20., 9.]]))
    mu, sigma = surrogate.predict([{'x': 0}, {'x': 1}])
    np.testing.assert_array_equal(mu, [10., 20.])
    np.testing.assert_array_equal(sigma, [2., 3.])


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
@pytest.mark.parametrize('ensemble_af', [False, True])
def test_fantasy_refits_preserve_weights_and_never_enter_prediction_cache(monkeypatch, direction, ensemble_af):
    class Memorizer(Model):
        def fit(self, candidates, y):
            self.trained = {engine.candidate_to_key(c): float(v) for c, v in zip(candidates, y)}
        def predict(self, candidates):
            # In-sample predictions are conspicuous, so contaminated cache fails.
            return np.array([999. if engine.candidate_to_key(c) in self.trained else self.mean for c in candidates]), np.ones(len(candidates))
    monkeypatch.setattr(engine, '_create_surrogate_from_spec', lambda spec, *a, **k: Memorizer(2. if spec.model_id == 'catboost' else 0.))
    original_fit = RobustNLLSurrogate.fit
    fits = []
    def fit(self, candidates, y):
        fits.append((len(candidates), self.weights.copy()))
        return original_fit(self, candidates, y)
    monkeypatch.setattr(RobustNLLSurrogate, 'fit', fit)
    settings = Settings(ensemble_af=ensemble_af, zero_llm_ablation_enabled=False, shortlist_top_k=3, autobo_acq_top_k=3,
                        autobo_unseen_category_exploration_enabled=False)
    state = dict(iteration=10, warm_start_target=10, optimization_direction=direction,
                 problem_spec={'variables':[{'name':'x','type':'categorical','domain':list(range(20))}], 'dataset':{}},
                 observations=[{'candidate':{'x':i},'result':0.} for i in range(10)], performance_log=[])
    out = engine.run_autobo_iteration(state=state, settings=settings, llm=None, invoke_json_node=lambda *a, **k: ({}, [], {}))
    cache = out['autobo_state']['robust_nll']
    assert out['payload']['metadata']['ensemble_af_enabled'] == ensemble_af
    assert {size for size, w in fits} >= {10, 11, 12}
    for size, weights in fits:
        np.testing.assert_array_equal(weights, [.25]*4)
    assert len(cache['pending']) == 3
    for forecast in cache['pending'].values():
        assert forecast['mean'] == [0., 0., 0., 2.]
    # Selection need not be top-1. Record through the graph's result-state helper.
    candidate = out['payload']['shortlist'][2]['candidate']
    state['autobo_state'] = out['autobo_state']
    result = engine.record_autobo_result(state=state, settings=settings, selected={}, shortlist=out['payload']['shortlist'], candidate=candidate, result_value=0.)
    state['autobo_state'] = result['autobo_state']
    state['observations'].append({'candidate': candidate, 'result': 0.})
    fits.clear()
    out = engine.run_autobo_iteration(state=state, settings=settings, llm=None, invoke_json_node=lambda *a, **k: ({}, [], {}))
    cache = out['autobo_state']['robust_nll']
    np.testing.assert_allclose(cache['history'][0], [-.5*np.log(2*np.pi)]*3 + [-2.-.5*np.log(2*np.pi)])
    assert cache['weights'][-1] < .25
    for size, weights in fits:
        np.testing.assert_array_equal(weights, cache['weights'])


def test_acquisition_weights_change_ei_ucb_and_ts():
    class Opposing(Model):
        def predict(self, candidates):
            return self.mean*np.array([c['x'] for c in candidates]), np.ones(len(candidates))*.1
    candidates = [{'x': -1.}, {'x': 1.}]
    for weights, expected in [([1., 0.], 0), ([0., 1.], 1)]:
        mixture = RobustNLLSurrogate([lambda: Opposing(-2), lambda: Opposing(2)], weights)
        mixture.fit([], [])
        for af in ['qlogei', 'qucb', 'ts']:
            with np.errstate(divide='ignore'):
                result = engine._score_candidate_pool_with_af(af_key=af, surrogate=mixture, candidate_pool=candidates,
                            best_f_scaled=0., y_mean=0., y_std=1., direction='maximize', seed=42)
            assert np.argmax(result['acquisition']) == expected
