"""M5d: cold-start twin build + ParticleCloud + full TwinState round-trip through storage."""

import numpy as np
import pytest

from domain import Machine, Run, Cut, FeatureRecord, WearLabel
from storage import SQLiteDataStore
from twin import ParticleCloud, build_twin, models_from_state


def test_cloud_weighted_stats():
    cl = ParticleCloud(np.array([0.05, 0.10, 0.15, 0.20]),
                       np.array([0.1, 0.2, 0.3, 0.4]))
    assert cl.mean_wear() == pytest.approx(0.15)          # weighted mean
    assert 0.05 <= cl.quantile_wear(0.5) <= 0.20

def test_cloud_serialization_roundtrip():
    rng = np.random.default_rng(0)
    cl = ParticleCloud(rng.uniform(0, 0.2, 500), np.full(500, 1 / 500))
    back = ParticleCloud.from_bytes(cl.to_bytes())
    assert np.allclose(back.wear, cl.wear) and np.allclose(back.weights, cl.weights)


def _synthetic_reference_store(tmp_path, n=200):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    rng = np.random.default_rng(0)
    for cut in range(1, n + 1):
        wear = 0.05 + 0.12 * (cut / n) ** 2.5              # accelerating wear-out
        fz = 1521 * wear ** 2.0 + rng.normal(0, 2.0)       # obs power law + noise
        ds.append_cut(Cut("c1", cut, f"k{cut}"))
        ds.append_features(FeatureRecord("c1", cut, {"force_z_rms": fz}))
        ds.append_wear_label(WearLabel("c1", cut, wear))
    return ds

def test_build_twin_produces_valid_initial_state(tmp_path):
    ds = _synthetic_reference_store(tmp_path)
    state = build_twin(ds, "c1", n_particles=1500)
    ds.close()
    assert state.cut_index == 0
    assert state.params["degradation"]["p"] > 1.0          # accelerating
    assert state.params["observation"]["k"] > 1.0          # convex
    cloud = ParticleCloud.from_bytes(state.particles)
    assert cloud.n == 1500
    assert np.all(cloud.wear > 0)                          # physical
    assert cloud.weights.sum() == pytest.approx(1.0)

def test_twin_state_survives_sqlite_roundtrip(tmp_path):
    # build -> save -> load -> reconstruct: the particles BLOB path end to end
    ds = _synthetic_reference_store(tmp_path)
    state = build_twin(ds, "c1", n_particles=1000)
    ds.save_twin_state(state)
    loaded = ds.load_twin_state("c1")
    cloud = ParticleCloud.from_bytes(loaded.particles)
    deg, obs = models_from_state(loaded)
    ds.close()
    assert cloud.n == 1000
    assert deg.p == pytest.approx(state.params["degradation"]["p"])
    assert obs.k == pytest.approx(state.params["observation"]["k"])

def test_build_twin_without_labels_raises(tmp_path):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    with pytest.raises(ValueError):
        build_twin(ds, "c1")
    ds.close()
