# Kaful — Data-First Streaming RUL Twin for CNC Tools

A production-shaped digital twin that estimates the **remaining useful life** of a
CNC cutting tool, cut by cut, from raw sensor waveforms. Wear is treated as a
hidden state recovered by a particle filter — never read from a label — and every
prediction ships with a calibrated uncertainty interval.

Built and validated on the [PHM 2010 milling challenge](https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010)
dataset (record `c1`).

---

## What it is

Sensor data streams in one *cut* at a time. For each cut the system stores the raw
waveform, extracts scalar features, and updates a per-tool digital twin whose hidden
state is flank wear. A power-law degradation model supplies the wear dynamics; a
power-law observation model links wear to a force feature; a particle filter fuses
the two to estimate wear from noisy signals; and a Monte Carlo projection turns the
posterior into an RUL distribution with confidence bounds — refreshed after every cut.

The guiding principle is **data-first**: the twin models only what the sensors
actually observe (the tool's flank wear), and wear labels are used *only* to score
the system, never to run it — because in production you cannot measure flank wear
mid-operation.

## Results (PHM 2010, record c1)

Two tiers, kept deliberately separate by honesty of the ground truth.

**Tier 1 — wear estimation, scored against *observed* labels (non-circular):**

| metric | wear-out regime | overall |
|---|---|---|
| wear RMSE | **6.3 µm** | 10.5 µm |
| 90% CI coverage | **0.90** (target ~0.90) | 0.73 |

The filter recovers latent wear from a single noisy force channel to ~6 µm in the
wear-out regime — where a *single* observation is ambiguous to ±25 µm — and its
uncertainty interval is calibrated (contains truth ~90% of the time). Running-in is
a known limitation (see below), which is why the overall numbers trail the wear-out
numbers.

**Tier 2 — RUL, scored against *extrapolated* pseudo-truth (flagged):** c1 never
reaches the 0.2 mm failure threshold, so true RUL is extrapolated and shares the
degradation model with the prediction. These numbers are reported for completeness
but are **not** claimed as verified accuracy; an honest RUL accuracy test needs a run
that actually fails (c4/c6), which is future work.

## How it works

```
[edge gateway] --POST compressed waveform--> [Ingest API]
                                                  |
                    +-----------------------------+------------------------+
                    v                                                      v
            [object storage]                                     [feature extractor]
          raw waveforms (immutable)                             waveform -> 42 scalars
                                                                           |
                                                                           v
                                                                   [feature store (SQLite)]
                                                                           |
                                                                           v
                                                    +------------------------------------+
                                                    |            particle twin           |
                                                    | 1. load state (particle cloud)     |
                                                    | 2. predict (degradation model)     |
                                                    | 3. update (observation likelihood) |
                                                    | 4. resample                        |
                                                    | 5. Monte Carlo -> RUL distribution |
                                                    | 6. persist state                   |
                                                    +------------------------------------+
                                                                           |
                                                                           v
                                                                  [RUL predictions] --> dashboard
[wear labels] --> [validation harness]   # scores wear & RUL; the twin never uses them to run
```

**Core modeling.** Wear evolves as a Paris-style power law `dw/dn = a·wᵖ` (a genuine
Markov state model, fit on the wear-out regime). A force feature relates to wear as
`force = f0 + c·wᵏ`. A particle filter estimates the wear posterior each cut; Monte
Carlo forward-simulation (with process noise) projects it to the threshold for a
predictive RUL interval, censoring futures that don't fail within a horizon.

## The build

Built as a **walking skeleton**: M1→M4 proved every pipe end-to-end with a *stub*
twin before any modeling, then M5→M8 dropped in the real science, and M9 wrapped the
proven handler in HTTP.

- **M1–M2** domain models + storage interfaces (`DataStore`/`ObjectStore`), SQLite +
  filesystem implementations — off parquet from day one.
- **M3–M4** feature extractor + ingest handler + replay driver (stub twin) — data
  flowing through every layer.
- **M5** degradation model, observation model, cold-start particle cloud.
- **M6–M7** particle filter (wear tracking) + Monte Carlo RUL.
- **M8** two-tier validation harness — which caught the filter being **overconfident**
  (CI coverage 0.46). A disciplined diagnosis followed: an observation-model
  intercept was hypothesized, *measured, and rejected* (no effect); per-phase
  analysis then localized the problem to running-in; and an observation-noise
  calibration lifted **wear-out coverage 0.55 → 0.90 at zero RMSE cost**, while
  running-in was documented as a model-limited regime rather than papered over.
- **M9** FastAPI transport — the ingest handler was designed at M4 to take raw bytes,
  so the web layer wraps it with *zero* changes to twin/storage logic.

74 tests (4 exercise the real PHM dataset and skip if it's absent).

## Repo layout

```
domain/        dataclasses (Machine, Run, Cut, ...) + abstract DataStore/ObjectStore
storage/       SQLiteDataStore, FilesystemObjectStore  (swap targets: Postgres, S3/MinIO)
features/      FeatureExtractor (6 stats x 7 channels = 42 features)
datasets/      PHM 2010 adapter (waveform + wear-label loading)
ingest/        codec, IngestHandler (the transport-free orchestrator), replay driver
twin/          degradation + observation models, particle cloud, filter, RUL, ParticleTwin
evaluation/    prognostic metrics (RMSE, alpha-lambda, prognostic horizon, coverage)
api/           FastAPI app over the ingest handler
scripts/       run_replay, load_labels, build_twin, run_filter, validate, calibrate, serve, ...
tests/         74 tests
```

## Quickstart

```bash
# environment
conda create -n kaful-data-twin python=3.11 -y && conda activate kaful-data-twin
pip install -e ".[dev]"
python -m pytest -q                       # 74 passed (needs PHM data for 4 of them)

# data: download PHM 2010 record c1, keep it OUT of any cloud-synced folder,
# and symlink it in (data/ is gitignored):
mkdir -p ~/datasets/phm2010 && unzip <c1>.zip -d ~/datasets/phm2010
ln -s ~/datasets/phm2010 data/phm2010

# run the pipeline
python scripts/run_replay.py --record c1  # ingest all cuts (features + raw)
python scripts/load_labels.py --record c1 # load wear labels (validation only)
python scripts/run_filter.py --record c1  # particle filter -> wear + RUL
python scripts/validate.py   --record c1  # two-tier prognostic scorecard
python scripts/calibrate.py  --record c1  # sweep observation-noise scale vs coverage

# run it as a service
python scripts/serve.py                   # POST /machines/{m}/runs/{r}/cuts, GET .../rul
```

## Key design decisions

- **The cut is the transaction unit.** RUL only matters between cuts (keep cutting or
  swap the tool?), so cut-level is the right granularity — not real-time samples, not
  batch.
- **Wear is latent; labels are validation-only.** The twin depends on no supervised
  wear signal at runtime, matching production reality.
- **Data defines the twin's scope.** Only sensor-observed components are modeled
  (here, flank wear) — no open-loop simulation of parts the data is silent on.
- **Storage behind interfaces.** Everything calls `DataStore`/`ObjectStore`, never a
  concrete backend, so SQLite→Postgres and filesystem→S3 are drop-in swaps that touch
  zero twin logic.
- **Uncertainty is first-class.** Every RUL is a distribution; the filter's intervals
  are *calibrated*, and futures that can't fail within the horizon are censored rather
  than reported as fake-precise numbers.

## Honest limitations & what's next

- **RUL accuracy is not yet verified.** c1 never crosses the failure threshold, so RUL
  ground truth is extrapolated. The real test is a run that fails (c4/c6) — **next**.
- **Running-in is undercalibrated.** The force–wear relationship appears to be
  two-regime; a single global observation curve fits wear-out well but misses
  running-in (coverage ~0.25 there). The twin's honest operating envelope is the
  **wear-out regime**, where RUL decisions are actually made.
- **Fixed degradation parameters.** `(a,p)` are fit on the reference run, not
  estimated per-tool. Joint state–parameter estimation would let the twin adapt to a
  tool that wears faster than the reference — a clean extension of the particle cloud.
- **Single feature.** Uses `force_z_rms` (thrust). Fusing an independent modality
  (vibration) is a natural robustness upgrade.
- **Single writer.** Sized for one machine emitting a cut every few minutes; the fleet
  path is per-run locking + Postgres/TimescaleDB + S3/MinIO + a queue (MQTT).

## Tech stack

Python 3.11 · NumPy/SciPy (filter, fitting, Monte Carlo) · SQLite (feature/RUL/state
store) · FastAPI + Uvicorn (transport) · pytest.