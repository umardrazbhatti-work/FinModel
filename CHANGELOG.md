# Changelog

All notable changes, amendments, mistakes, and operational decisions for this project.

**Purpose:** Prevent repeating the same mistakes weeks later. Update this file with **every** change.

Format per entry:
- **Date**
- **What changed**
- **Why**
- **Mistakes / pitfalls / lessons** (if any)

---

## 2026-08-14 — Kaggle re-run with target_clip=5.0 (14-8-26 1300Hrs)

### What changed
1. **Second full Kaggle walk-forward** completed with code from GitHub `7a7e58a` (`target_clip: 5.0` + fixed economic metrics).
2. Analysis pack placed at `Results/exp_eurusd_kaggle_12h - 14-8-26 1300Hrs/exp_eurusd_kaggle_12h/`.
3. Offline rescore + diagnosis regenerated (`rescore_fixed_econ/`, `diagnosis/`, pack-level `VERDICT.md`).
4. Confirmed config on run had `data.target_clip: 5.0`; prediction samples show vol-normalized \(y \in [-5, +5]\).

### Why
- Measure whether winsorizing heavy-tailed targets alone moves multi-TF past single-TF under the same protocol.
- Validate that in-pack economic DD is no longer absurd after the wealth-curve fix.

### Verdict (vs run 13-8 no-clip)
| Check | Run 1 (13-8) | Run 2 (14-8, clip=5) |
|-------|--------------|----------------------|
| Runtime | ~27 min | ~34 min |
| Mean MTP pinball | 0.677 | 0.546* |
| Mean single-TF pinball | 0.661 | 0.536* |
| Δ MTP − single-TF | +0.0168 | **+0.0097** (gap narrowed, still worse) |
| Folds MTP beats single-TF | 1/6 | **1/6** |
| MTP / zero pinball | 0.675 | 0.624 (slightly better vs zero) |
| Best epoch (mean) | ~1.8 | **~1.7** (still early stop peak) |
| corr(y, q50) | ≈ −0.02 | ≈ −0.01 (still ~noise) |
| Re-scored 1h h0 Sharpe (mean) | −2.53 | **−2.87** |
| Official go/nogo (vs zero/mean) | PASS | **PASS** |
| Economic MDD | broken (1e9) | sane fractional wealth DD |

\*Absolute pinball not comparable across runs (label scale changed). Fair bar is **MTP vs single-TF within the same run**.

### Outcome
- **Engineering win:** clip applied correctly; econ metrics usable; pipeline stable.
- **Research fail (multi-TF thesis):** multi-TF still does **not** advance vs single-TF.
- Clip was **label hygiene**, not a multi-TF win. Treat single-TF as champion until MTP wins pinball under same folds/budget.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Absolute pinball drop after clip looks like “big model win” | Both MTP and single-TF train on clipped \(y\); always compare relative to single-TF and ratios to zero. |
| Gap narrowed but still positive Δ | Partial improvement ≠ multi-TF advance; need mean pinball **lower** than single-TF. |
| Epoch 1–2 peak unchanged by clip | Next lever is optim/reg (LR, weight decay, dropout), not more capacity. |
| corr(y, q50) still ~0 | Architecture/fusion not learning useful median skill yet. |

### Controlled upgrade queue (status after this run)
1. **target_clip=5.0** — done + re-run measured; multi-TF still loses.
2. Milder optim (LR 1e-4 / cosine / higher wd) — **next**.
3. Slightly higher dropout — pending.
4. Offline signal-threshold sweep on fixed q50 — pending (no retrain).
5. Gate temperature / entropy retune — only if 2–3 improve val curves.
6. Do **not** add cross-attention or larger `d_model` yet.

---

## 2026-08-13 — First Kaggle full run review, econ fix, diagnosis, target_clip

### What changed
1. **Kaggle first full run completed** (~27 min, not 12h): 6 folds, CUDA, analysis pack extracted under `Results/exp_eurusd_kaggle_12h - 13-8-26 2330Hrs/`.
2. **Economic metrics fix** (`src/evaluation/economic.py`):
   - Max drawdown now uses wealth curve \(W_0=1\), \(W_t=1+\sum r_i\), fractional peak-to-trough.
   - Old formula divided by near-zero cumsum peak → DD values like −1e9.
   - Added `score_position_returns`, position mix fields (`pct_long/short/flat`), `final_wealth`, `max_drawdown_abs`.
3. **Unit tests** `tests/test_economic.py` (7 tests) — regression that MDD stays in a sane band.
4. **Offline tools** (do not need retrain):
   - `scripts/rescore_from_pack.py` — re-score pack `predictions_sample.csv` with fixed econ metrics.
   - `scripts/diagnose_early_stop.py` — train/val curves, target tails, gates, multi-TF vs single-TF.
5. **Controlled upgrade A — target winsorization**:
   - `MultiTFDataset(target_clip=5.0)` clips vol-normalized \(y\) to \([-5, +5]\) (null/0 = off).
   - Wired through configs (`default`, `eurusd_1h`, `kaggle_eurusd_12h`) and `train_fold` / `run_walk_forward` / `evaluate`.
   - Raw returns used for economic eval are **not** clipped (trading path unchanged).
6. GitHub `FinModel` updated with econ fix + tools + target_clip.

### Why
- Trustworthy Sharpe/DD before any model claim.
- Explain epoch-1–2 early stop and multi-TF failing vs single-TF before spending more GPU time.
- Heavy-tailed vol-normalized targets (|y| up to ~50) dominated pinball; clipping is the first controlled upgrade (no capacity increase).

### First Kaggle run — verdict (pinball protocol unchanged)
| Check | Result |
|-------|--------|
| Pipeline / 6 folds / pack | Working |
| Beat zero & hist-mean | 6/6 PASS (official go/nogo) |
| Beat **single-TF** | **FAIL** (MTP 0.677 vs single-TF 0.661; only 1/6 folds better) |
| Best epoch | ~1–2 every fold; then val rises, train falls (overfit) |
| Gates | Near-uniform (not collapsed, not specialized) |
| corr(y, q50) | ≈ −0.02 (median barely tracks target) |
| Re-scored 1h h0 Sharpe (mean) | **−2.53** (econ weak; DD now ~−2.6% wealth, not billions) |
| Wall time ~27 min | Expected: ES patience 12 → ~14 epochs/fold, not 60 |

### Re-score / diagnosis outputs (local Results pack)
- `rescore_fixed_econ/economic_rescore_summary.json`
- `diagnosis/diagnosis.json`, `diagnosis_report.md`
- `VERDICT_post_rescore.md`

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Cumsum equity at 0 + relative DD | Always use wealth starting at 1 for fractional MDD. |
| Official go/nogo only vs zero/mean | Research advance bar is **single-TF** under same protocol. |
| 12h config finished in 27 min | Ceiling ≠ required train time; best epoch 1–2 means more epochs alone will not help. |
| Vol-normalized y has fat tails | Pinball chases extremes; winsorize before scaling model size. |
| Windows zip used `\` in entry names | Kaggle rejects; rebuild zip with POSIX `/` paths. |
| Kaggle P100 + torch cu128 | No sm_60 kernels; use T4 or install torch 2.1.2+cu118. |
| Dataset path `/kaggle/input/datasets/<user>/<slug>/aligned` | Notebook must rglob, not assume flat `/kaggle/input/<slug>`. |
| `predictions_sample` is 500 rows/TF/horizon | Re-scored Sharpe/DD are approximate; full-fold needs checkpoints or full export. |
| Do not jump to cross-attention yet | Next experiments one-at-a-time: target_clip (done+measured) → milder LR/wd → dropout → signal threshold sweep offline. |

### Controlled upgrade queue (status at time of this entry)
1. **target_clip=5.0** — implemented; full re-run on 2026-08-14 (see entry above).
2. Milder optim (LR 1e-4 / cosine / higher wd) — pending.
3. Slightly higher dropout — pending.
4. Offline signal-threshold sweep on fixed q50 — pending (no retrain).
5. Gate temperature / entropy retune — only if 2–3 improve val curves.

---

## 2026-08-13 — Kaggle 12h full run + analysis pack (20–30 files)

### What changed
1. Added `configs/kaggle_eurusd_12h.yaml` sized for a continuous ~12h Kaggle GPU session:
   - `max_epochs=60` (ceiling), `early_stopping_patience=12`
   - `max_folds=6`, `step_bars=1500` (expanding walk-forward)
   - single-TF baseline: `baseline_max_epochs=40`, patience 10
2. Expanded artifact export (`src/evaluation/artifacts.py`):
   - Per fold: `metrics.json`, `history.csv`, `gates.csv`, `stats_detail.json`, `pinball_by_tf.csv`, `economic_by_horizon.csv`, `predictions_sample.csv`, optional single-TF baseline files
   - Experiment-level numbered pack: `00_…` through `18_…` (report, overview, baselines, gates, curves, coverage, economic, go/no-go, checklist, runtime, hyperparams, deltas, fold schedule, pack info)
   - `package_analysis_zip()` builds a downloadable zip (checkpoints excluded by default)
3. `scripts/run_walk_forward.py` now always writes the analysis pack + zip at the end.
4. Kaggle notebook updated to run the 12h config and place the zip at `/kaggle/working/<exp>_analysis_pack.zip`.
5. Default/local configs aligned to the same epoch/fold settings.

### Why
- Use the full Kaggle 12h window for a complete multi-fold experiment, not a tiny smoke test.
- Produce a self-contained zip (≈20–30+ analysis files) that can be downloaded and reviewed offline to decide go/no-go.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Setting only `max_epochs` without early stopping wastes the whole 12h on overfit folds | Keep patience=12; max_epochs is a ceiling. |
| Zipping `.pt` checkpoints makes downloads huge | Default analysis pack excludes checkpoints; opt-in with `--include-checkpoints`. |
| Zero/mean baselines are free; single-TF baseline doubles train time | Budget baseline epochs separately (`baseline_max_epochs=40`). |
| File count grows with folds (fold_k/*) | Root pack has ~18 numbered files; plus ~7 per fold → often 30–50 entries in zip — that is intentional for analysis. |
| Notebook must write zip under `/kaggle/working` | Otherwise hard to find in Kaggle UI download panel. |

### Runtime budget (planning)
- 6 folds × MTP (≤60 epochs, usually early-stops) + single-TF (≤40 epochs)
- Small model (~0.9M params) → designed to fit a 12h GPU session with margin
- If overtime risk: lower `max_folds` or set `SKIP_SINGLE_TF_BASELINE=True` in notebook

---

## 2026-08-13 — Project bootstrap & first implementation

### What changed
1. Created clean project root `Financial_AI_Research/` with structure:
   - `data/aligned/` (final parquet only)
   - `design/mtp_transformer_design/` (locked specs; local only)
   - `src/`, `scripts/`, `tests/`, `configs/`, `outputs/`
2. Implemented full MTP-Transformer v1 pipeline per locked design STEP1–STEP7:
   - Data: `MultiTFDataset` (strict truncation, vol-normalized targets, fold bounds)
   - Model: per-TF patch encoder + RevIN, static TF gates, fusion MLP, multi-horizon quantile heads
   - Loss: multi-quantile pinball + gate entropy regularization
   - Training: pure PyTorch trainer, expanding walk-forward folds, early stopping
   - Evaluation: statistical metrics + simple median-threshold economic metrics
   - Baselines: predict-zero, historical mean, single-TF patch model
   - Scripts: `train_fold.py`, `run_walk_forward.py`, `evaluate.py`
3. Unit tests: leakage, shapes, loss (12 passed).
4. Smoke test: EURUSD fold 0, 2 epochs on CPU — train pinball decreased; test pinball beat zero/mean baselines; gates did not collapse.
5. **Deleted all archive clutter** from parent folder `Forex Data Scrapping/`:
   - Intermediate dirs: `cleaned_1m/`, `multi_tf/`, `macro_fred/`, `economic_calendar/`, residual `aligned/` CSVs
   - All scraper/cleaner/aligner/validator scripts
   - All raw `*_1m_last_*.csv` files
   - Parent folder now contains **only** `Financial_AI_Research/`
6. Created Kaggle dataset zip: `kaggle_upload/mtp_aligned_parquet.zip` (~142 MB of 30 aligned parquets under `aligned/`).
7. Added Kaggle runner notebook (local only, not for GitHub): `notebooks/kaggle_mtp_walk_forward.ipynb`.
8. Added `.gitignore` so GitHub receives **code only** (no data, design docs, notebooks, outputs, changelog).

### Why
- Separate research code from one-off data-prep mess.
- Only final aligned datasets belong in the active project.
- Kaggle needs a zip of the dataset; code will live on GitHub and be pulled by a Kaggle notebook.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Parquet time column is named `time`, not `timestamp` | Dataset renames `time` → `timestamp` on load. Do not assume design-doc column names match files. |
| Daily files are `*_daily_aligned.parquet`, design uses TF name `1d` | `TF_FILE_STEM` maps `1d` → `daily`. Keep this mapping if renaming files. |
| Windows `cp1252` console cannot log Unicode arrows (`→`) | Use ASCII in log messages (`->`) or set `PYTHONIOENCODING=utf-8`. |
| Intermediate data folders are huge and must never re-enter the active tree | Only `*_aligned.parquet` in `data/aligned/`. Everything else is delete/archive. |
| Standardization must use **train-fold only** stats | Never fit scalers on val/test. `fit_standardization` on train → `set_standardization` on val/test. |
| Gate entropy sign convention | Anti-collapse: minimize `pinball - λH` (implemented as `pinball + λ * (-H)`). Do not flip this without re-checking gate collapse. |
| GitHub must not contain design docs, notebooks, or data | `.gitignore` excludes `design/`, `*.ipynb`, `data/aligned/`, `kaggle_upload/`, `outputs/`. CHANGELOG is tracked for experiment history. |
| Never commit as Grok / add AI signatures | Commits use the human git user only. No “Generated by …” footers. |
| Parent workspace had both research project and scrapers mixed | After cleanup, only `Financial_AI_Research` remains under `Forex Data Scrapping`. |

### Smoke-test numbers (fold 0, 2 epochs, CPU)
- Params: ~867k
- Test pinball ≈ 0.70 vs zero ≈ 1.02 / mean ≈ 1.02
- Gates ≈ uniform (~0.16–0.17 each TF)

### Git / GitHub readiness (pending remote URL)
- `git init -b main` inside `Financial_AI_Research/`
- Code-only files staged (42 paths): `src/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`, `pyproject.toml`, `README.md`, `.gitignore`
- **Not staged / ignored:** `design/`, `notebooks/*.ipynb`, `data/aligned/*.parquet`, `kaggle_upload/`, `outputs/`, `CHANGELOG.md`
- No commit yet: waiting for (1) GitHub repo URL, (2) your `user.name` + `user.email` so contributions show **your** name only (no AI author metadata)
- Kaggle zip ready at `kaggle_upload/mtp_aligned_parquet.zip` (~114 MB, 30 parquets under `aligned/`)
- Kaggle notebook ready at `notebooks/kaggle_mtp_walk_forward.ipynb` (upload to Kaggle only)

---

## Template for future entries

```
## YYYY-MM-DD — short title

### What changed
- ...

### Why
- ...

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| ... | ... |
```
