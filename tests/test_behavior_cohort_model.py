import numpy as np
import pytest

from behavior_cohort_model import (
    CohortConfig, CohortTemplate, mass_basis, predict_cohort_curve,
)
from behavior_pace_model import MS_PER_HOUR as H
from scripts.experiment_behavior_cohorts import aggregate

START = 1_700_000_000_000
END = START + 120 * H


def fixture():
    templates = [
        CohortTemplate(1, START - H, "challenge", 500, (1,0,0,0,0,0), 0),
        CohortTemplate(2, START - H, "challenge", 500, (0,0,0,0,0,1), 0),
    ]
    hours = np.arange(0, 73, 6)
    curve = mass_basis(hours, duration_hours=120,
        start_local_hour=(START/H+8)%24, tier=500, reward_tiers=[500])[:, 5]
    observations = np.column_stack((START + hours * H, 1_000_000 * curve))
    args = dict(tier=500, start_at=START, end_at=END, origin_at=START+72*H,
        event_type="challenge", reward_tiers=[500], templates=templates,
        config=CohortConfig(strength=64))
    return observations, args


def test_visible_behavior_selects_deadline_cohort_and_preserves_anchor():
    observations, args = fixture()
    grid = np.arange(START+72*H, END+1, H)
    predicted, diagnostic = predict_cohort_curve(observations, **args, forecast_times=grid)
    assert predicted[0] == pytest.approx(observations[-1,1])
    assert np.all(np.diff(predicted) >= 0)
    assert predicted[-1] == pytest.approx(1_000_000, rel=0.01)
    assert diagnostic["posterior_mean_mass"][5] > 0.99


def test_future_score_suffix_is_never_parsed_and_lag_is_integrated():
    observations, args = fixture()
    lagged = observations[:-1]
    expected, _ = predict_cohort_curve(lagged, **args)
    poisoned = np.vstack((lagged.astype(object), [START+73*H, "FUTURE_INVALID"]))
    actual, _ = predict_cohort_curve(poisoned, **args)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] > lagged[-1,1]


def test_future_training_template_is_rejected():
    observations, args = fixture()
    args["templates"] = [CohortTemplate(3,START,"challenge",500,(1,0,0,0,0,0),0)]
    with pytest.raises(ValueError, match="strictly before"):
        predict_cohort_curve(observations, **args)


def test_type_and_prefix_failures_are_explicit():
    observations, args = fixture()
    args["event_type"] = "unseen"
    with pytest.raises(ValueError, match="no historical template"):
        predict_cohort_curve(observations, **args)
    args["event_type"] = "challenge"
    observations[-1,1] = 1
    with pytest.raises(ValueError, match="must not decrease"):
        predict_cohort_curve(observations, **args)


def test_reward_transfer_increases_deadline_mass_without_moving_anchor():
    observations, args = fixture()
    args["templates"] = [CohortTemplate(1,START-H,"challenge",500,
        (.5,0,0,0,0,.5),0,pressure=.5)]
    args["config"] = CohortConfig(strength=0)
    ordinary, _ = predict_cohort_curve(observations, **args)
    args["config"] = CohortConfig(strength=0,reward_transfer=True)
    reward, diag = predict_cohort_curve(observations, **args)
    assert reward[0] == pytest.approx(ordinary[0])
    assert reward[-1] > ordinary[-1]
    assert diag["posterior_mean_mass"][5] == pytest.approx(2/3)


def test_aggregation_weights_events_then_tiers_and_keeps_paired_support():
    rows = [dict(event_id=1,tier=500,actual=100,predictions={"a":110,"b":100})] * 10
    rows += [dict(event_id=1,tier=1500,actual=100,predictions={"a":130})]
    rows += [dict(event_id=2,tier=500,actual=100,predictions={"a":150,"b":100})]
    assert aggregate(rows,"a")["mape"] == pytest.approx(35)
    assert aggregate(rows,"a",support="b")["mape"] == pytest.approx(30)


def test_template_cache_refits_when_pace_model_changes(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from scripts import experiment_behavior_cohorts as experiment

    for name in ("behavior_cohort_model.py", "behavior_pace_model.py",
                 "behavior_pace_prior.py", "scripts/experiment_behavior_cohorts.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "192.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "CACHE", cache)
    monkeypatch.setattr(experiment, "OUT", tmp_path / "output")
    template = CohortTemplate(192, START-H, "challenge", 500, (1,0,0,0,0,0), 0)
    fitter = Mock(return_value=([template], [], []))
    monkeypatch.setattr(experiment, "build_templates", fitter)

    experiment.get_templates(192, "cohort6")
    experiment.get_templates(192, "cohort6")
    assert fitter.call_count == 1

    (tmp_path / "behavior_pace_model.py").write_text("changed integrator", encoding="utf-8")
    experiment.get_templates(192, "cohort6")
    assert fitter.call_count == 2
