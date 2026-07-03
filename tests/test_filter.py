"""M6: particle filter mechanics + ParticleTwin. Recovers wear from noisy force."""

import numpy as np
import pytest

from domain import Machine, Run, Cut, FeatureRecord, WearLabel, RULPrediction
from storage import SQLiteDataStore
from twin import (PowerLawWear, PowerLawObservation, ParticleCloud,
                  filter_step, systematic_resample, build_twin, ParticleTwin)


def test_systematic_resample_preserves_count():
    rng = np.random.default_rng(0)
    weights = np.array([0.7, 0.2, 0.05, 0.05])
    idx = systematic_resample(weights, rng)
    assert idx.size == 4 and idx.max() < 4
    assert (idx == 0).sum() >= 2

def test_filter_recovers_synthetic_wear():
    kc = np.array([1,64,127,190,253,315]); kw = np.array([39.6,91.1,95.0,112.5,135.7,165.2])/1000
    cuts = np.arange(1, 316); true = np.interp(cuts, kc, kw)
    deg = PowerLawWear(9.883e-2, 2.727)
    obs = PowerLawObservation("force_z_rms", 1521.3, 2.035, 2.482)
    rng = np.random.default_rng(1)
    observed = obs.expected(true) + rng.normal(0, obs.sigma, cuts.size)

    cloud = ParticleCloud(rng.uniform(0.005, 0.08, 2000), np.full(2000, 1/2000))
    est = []
    for f in observed:
        cloud = filter_step(cloud, deg, obs, f, process_noise=0.002, rng=rng)
        est.append(cloud.mean_wear())
    rmse_um = np.sqrt(np.mean((np.array(est) - true) ** 2)) * 1000
    assert rmse_um < 6.0


def _synthetic_store(tmp_path, n=200):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    rng = np.random.default_rng(0)
    for cut in range(1, n + 1):
        wear = 0.05 + 0.12 * (cut / n) ** 2.5
        fz = 1521 * wear ** 2.0 + rng.normal(0, 2.0)
        ds.append_cut(Cut("c1", cut, f"k{cut}"))
        ds.append_features(FeatureRecord("c1", cut, {"force_z_rms": fz}))
        ds.append_wear_label(WearLabel("c1", cut, wear))
    return ds

def test_particle_twin_update_persists_state_and_returns_rul(tmp_path):
    ds = _synthetic_store(tmp_path)
    ds.save_twin_state(build_twin(ds, "c1", n_particles=1000))
    twin = ParticleTwin(ds, process_noise=0.002)
    rul = twin.update("c1", 1, {"force_z_rms": 6.0})
    assert isinstance(rul, RULPrediction) and rul.cut_index == 1
    assert ds.load_twin_state("c1").cut_index == 1
    assert twin.last_wear_mean is not None
    ds.close()

def test_particle_twin_tracks_wear_over_run(tmp_path):
    ds = _synthetic_store(tmp_path)
    ds.save_twin_state(build_twin(ds, "c1", n_particles=2000))
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels("c1")}
    twin = ParticleTwin(ds, process_noise=0.002)
    final_err = None
    for f in ds.read_all_features("c1"):
        twin.update("c1", f.cut_index, f.features)
        final_err = abs(twin.last_wear_mean - labels[f.cut_index])
    assert final_err < 0.01
    ds.close()

def test_clear_rul(tmp_path):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("m", "phm2010_milling")); ds.create_run(Run("c1", "m"))
    ds.append_cut(Cut("c1", 1, "k"))
    ds.append_rul(RULPrediction("c1", 1, 40, 30, 50))
    assert len(ds.read_all_rul("c1")) == 1
    ds.clear_rul("c1")
    assert len(ds.read_all_rul("c1")) == 0
    ds.close()


def test_sigma_scale_widens_wear_band(tmp_path):
    ds = _synthetic_store(tmp_path)
    ds.save_twin_state(build_twin(ds, "c1", n_particles=2000))
    feats = ds.read_all_features("c1")
    def band_at_cut(scale):
        ds.save_twin_state(build_twin(ds, "c1", n_particles=2000))
        tw = ParticleTwin(ds, process_noise=0.002, sigma_scale=scale, seed=0)
        for f in feats[:30]:
            tw.update("c1", f.cut_index, f.features)
        return tw.last_wear_hi - tw.last_wear_lo
    assert band_at_cut(4.0) > band_at_cut(1.0)
    ds.close()
