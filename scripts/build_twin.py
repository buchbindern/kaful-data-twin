"""
Build and persist the cold-start twin for a reference run. Run from repo root:

    python scripts/build_twin.py --record c1

Fits the degradation + observation models on the run's stored features/labels,
seeds the initial particle cloud, and saves the TwinState (cut_index 0) to the
store. This is the state M6's filter will load and update cut by cut.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from storage import SQLiteDataStore
from twin import build_twin, ParticleCloud


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--feature", default="force_z_rms")
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--onset", type=float, default=None)
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    state = build_twin(ds, args.record, feature_name=args.feature,
                       n_particles=args.n_particles, onset_cut=args.onset)
    ds.save_twin_state(state)
    cloud = ParticleCloud.from_bytes(state.particles)
    ds.close()

    d = state.params
    print(f"built + persisted TwinState for run {args.record!r} at cut {state.cut_index}")
    print(f"  degradation:  dw/dn = {d['degradation']['a']:.3e} * w^{d['degradation']['p']:.3f}"
          f"   (wear-out onset cut {d['onset_cut']:.0f})")
    print(f"  observation:  {d['feature_name']} = {d['observation']['c']:.1f} * w^{d['observation']['k']:.3f}"
          f"   sigma={d['observation']['sigma']:.3f}")
    print(f"  threshold:    {d['threshold_mm']:.3f} mm")
    print(f"  prior cloud:  {cloud.n} particles, mean wear {cloud.mean_wear():.4f} mm "
          f"[{cloud.quantile_wear(0.05):.3f}, {cloud.quantile_wear(0.95):.3f}]")


if __name__ == "__main__":
    main()
