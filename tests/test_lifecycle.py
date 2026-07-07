"""M10: fit/deploy split + run lifecycle (tool change)."""

import numpy as np
import pytest

from domain import Machine, Run, Cut, FeatureRecord, WearLabel
from storage import SQLiteDataStore
from twin import (fit_model_spec, deploy_twin, deploy_from_reference, build_twin,
                  start_new_run, models_from_state, ParticleCloud)


def _reference_store(tmp_path, run_id="c1", n=200):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run(run_id, "phm2010"))
    rng = np.random.default_rng(0)
    for cut in range(1, n + 1):
        wear = 0.05 + 0.12 * (cut / n) ** 2.5
        fz = 1521 * wear ** 2.0 + rng.normal(0, 2.0)
        ds.append_cut(Cut(run_id, cut, f"k{cut}"))
        ds.append_features(FeatureRecord(run_id, cut, {"force_z_rms": fz, "vibration_x_mean_abs": fz}))
        ds.append_wear_label(WearLabel(run_id, cut, wear))
    return ds


def test_fit_spec_is_run_agnostic(tmp_path):
    ds = _reference_store(tmp_path)
    spec = fit_model_spec(ds, "c1")
    assert spec["reference_run_id"] == "c1"
    assert spec["degradation"]["p"] > 1.0 and spec["observation"]["k"] > 1.0
    assert "n_particles" not in spec
    ds.close()

def test_deploy_stamps_fresh_twin_on_target_run(tmp_path):
    ds = _reference_store(tmp_path)
    spec = fit_model_spec(ds, "c1")
    state = deploy_twin(spec, "c4", n_particles=1000)
    assert state.run_id == "c4" and state.cut_index == 0
    cloud = ParticleCloud.from_bytes(state.particles)
    assert cloud.n == 1000 and cloud.mean_wear() < 0.09
    deg, obs = models_from_state(state)
    assert deg.p == pytest.approx(spec["degradation"]["p"])
    ds.close()

def test_build_twin_still_self_references(tmp_path):
    ds = _reference_store(tmp_path)
    state = build_twin(ds, "c1", n_particles=500)
    assert state.run_id == "c1" and state.params["reference_run_id"] == "c1"
    ds.close()

def test_start_new_run_ends_active_and_deploys(tmp_path):
    ds = _reference_store(tmp_path)
    run = start_new_run(ds, "phm2010", "run2", reference_run_id="c1", n_particles=500)
    assert run.run_id == "run2" and run.ended_at is None
    assert ds.get_run("c1").ended_at is not None
    assert ds.get_active_run("phm2010").run_id == "run2"
    st = ds.load_twin_state("run2")
    assert st is not None and st.cut_index == 0
    assert st.params["reference_run_id"] == "c1"
    ds.close()

def test_start_new_run_rejects_duplicate(tmp_path):
    ds = _reference_store(tmp_path)
    with pytest.raises(ValueError):
        start_new_run(ds, "phm2010", "c1", reference_run_id="c1")
    ds.close()


def test_few_shot_calibration_fits_targets_own_map(tmp_path):
    """deploy_with_measurements should fit the TARGET tool's own observation map from a
    few measurements, not reuse the reference's map."""
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    # reference c1: map feat = 1000 * w^2.0 ; target c9: DIFFERENT map feat = 500 * w^1.4
    rng = np.random.default_rng(0)
    ds.create_run(Run("c1", "phm2010"))
    for cut in range(1, 201):
        w = 0.05 + 0.13 * (cut / 200) ** 2.5
        ds.append_cut(Cut("c1", cut, f"c1/{cut}"))
        ds.append_features(FeatureRecord("c1", cut, {"vibration_x_mean_abs": 1000 * w ** 2.0}))
        ds.append_wear_label(WearLabel("c1", cut, w))
    ds.create_run(Run("c9", "phm2010"))
    target_wear = {}
    for cut in range(1, 201):
        w = 0.05 + 0.13 * (cut / 200) ** 2.5
        ds.append_cut(Cut("c9", cut, f"c9/{cut}"))
        ds.append_features(FeatureRecord("c9", cut, {"vibration_x_mean_abs": 500 * w ** 1.4}))
        target_wear[cut] = w
    # provide ~10 evenly-spaced measurements on c9
    meas = {c: target_wear[c] for c in range(20, 201, 20)}
    from twin import deploy_with_measurements, models_from_state
    state = deploy_with_measurements(ds, "c9", meas, reference_run_id="c1", n_particles=500)
    _, obs = models_from_state(state)
    assert obs.k == pytest.approx(1.4, abs=0.25)          # recovered c9's own shape, not c1's 2.0
    assert state.params["calibrated_from"].endswith("on c9")
    ds.close()

def test_few_shot_needs_enough_measurements(tmp_path):
    ds = _reference_store(tmp_path)
    from twin import deploy_with_measurements
    with pytest.raises(ValueError):
        deploy_with_measurements(ds, "c1", {1: 0.05, 2: 0.06}, reference_run_id="c1")  # too few
    ds.close()
