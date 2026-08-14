# FinModel / MTP-Transformer — Master Plan

**Purpose:** Durable roadmap so work can continue if chat history is lost.  
**Living doc:** Stages may be added, split, merged, or dropped when evidence changes.  
**Fixed end goals (do not change):**  
1. **Publication** — defensible multi-horizon / multi-TF forecasting research  
2. **Product deployment** — usable, monitored forecasting/signal service  

**Related files (always read with this plan):**  
- `CHANGELOG.md` — what changed and why (experiment log)  
- `README.md` — how to run  
- `Results/` — offline analysis packs and verdicts (local)  
- Design specs (local only): `design/mtp_transformer_design/`  
- Code: https://github.com/umardrazbhatti-work/FinModel  

**How to update this plan:**  
1. Change stage status: `DONE` | `IN PROGRESS` | `NEXT` | `PENDING` | `BLOCKED` | `DROPPED`  
2. Add a short note under **Findings log** with date + decision  
3. Never rewrite end goals; only rewrite path to them  

**Last plan update:** 2026-08-14  

---

## 0. One-line thesis

A **multi-timeframe gated patch Transformer (MTP)** can improve quantile forecasts of FX (starting with EURUSD) over strong baselines **if and only if** it beats a fair single-TF patch model under leakage-safe walk-forward evaluation—and later shows economically usable risk-adjusted value for a product.

---

## 1. Fixed end goals

### Goal A — Publication
- Clear problem, method, and **leakage-safe** protocol  
- Strong baselines (at least: zero, hist-mean, **single-TF**)  
- Multi-fold walk-forward results with honest negative results where true  
- Ablations (gates, clip, optim, etc.) that justify design choices  
- Reproducible code + configs (this repo)

### Goal B — Product deployment
- Trained model(s) with versioned configs and metrics  
- Inference path on new bars (batch first, then near-real-time if needed)  
- Monitoring: pinball/coverage drift, signal quality, latency, cost model  
- Risk controls: thresholds, position sizing, kill-switch on drawdown  
- Ops: data refresh, retrain cadence, rollback  

**Publication is not the same as product.** A paper can ship on statistical skill; a product needs stable economics and ops. Do not skip research bars to “force” a product.

---

## 2. Non-negotiable decision rules

| Rule | Meaning |
|------|---------|
| **Fair architecture bar** | Multi-TF **advances only if** mean test pinball **&lt;** single-TF under the **same** folds, budget, targets, and optim. |
| **Trivial baselines are not enough** | Beating zero / hist-mean is necessary but not sufficient. |
| **One controlled change at a time** | Clip, optim, dropout, gates, capacity—never bundle. |
| **No capacity jump early** | No cross-attention / larger model until small levers are exhausted. |
| **Absolute pinball across runs is misleading** | After target scaling changes (e.g. clip), compare MTP vs single-TF **within** the run; use ratios to zero as secondary. |
| **Economics must use fixed wealth-curve DD** | Do not trust pre-fix max-drawdown (1e9 bugs). |
| **Single-TF is champion until multi-TF wins** | Product prototypes may use single-TF if multi-TF fails research bar. |

---

## 3. Stage map (start → end)

```text
[1 Data] → [2 Research stack] → [3 Kaggle baseline runs] → [4 Diagnosis]
    → [5 Controlled upgrades] → [6 Multi-TF proven?]
         ├─ YES → [7 Paper-ready evidence] → [8 Write/submit]
         │              ↓
         │         [9 Productize champion model]
         └─ NO  → [5b Pivot champion / new hypothesis] → re-enter 5 or 6
                              ↓
                    [9 Productize] still possible on single-TF
```

---

## 4. Stages (detail)

### Stage 1 — Data foundation
**Status:** `DONE`  

| Item | Notes |
|------|--------|
| Multi-pair multi-TF aligned parquets | EURUSD (+ others) under `data/aligned/` |
| Leakage rules | Strict truncation to ≤ t; past-only vol; purge in walk-forward |
| Kaggle dataset | `mtp-aligned-parquet` (`aligned/*.parquet`); zip must use `/` paths |
| Cost / vol targets | Vol-normalized log-returns; cost_threshold mask |

**Exit criteria:** Reproducible load via `MultiTFDataset`; unit tests for leakage/shapes.

---

### Stage 2 — Research code stack
**Status:** `DONE`  

| Item | Notes |
|------|--------|
| MTP model | Per-TF patch + Transformer encoder, static gates, fusion, quantile heads |
| Loss | Multi-quantile pinball + gate entropy |
| Walk-forward | Expanding folds, purge, val/test blocks |
| Baselines | Zero, hist-mean, single-TF patch |
| Analysis pack | Numbered files + zip for offline review |
| GitHub | https://github.com/umardrazbhatti-work/FinModel |
| Kaggle notebook | Local `notebooks/kaggle_mtp_walk_forward.ipynb` (not in git) |

**Exit criteria:** One full walk-forward completes on GPU and produces a downloadable pack.

---

### Stage 3 — First measured baselines (Kaggle)
**Status:** `DONE`  

| Run | Tag / folder | Result summary |
|-----|----------------|----------------|
| Run 1 | `Results/... 13-8-26 2330Hrs` | Pipeline OK; multi-TF **loses** to single-TF; best epoch ~1–2; econ DD broken |
| Run 2 | `Results/... 14-8-26 1300Hrs` | `target_clip=5`; gap narrowed but multi-TF still loses; DD fixed; best epoch still ~1–2 |

**Exit criteria:** Honest verdict documents + fair comparison numbers (see Results `VERDICT*.md`).

---

### Stage 4 — Metrics integrity & diagnosis
**Status:** `DONE` (tooling); **re-apply** on every new pack  

| Item | Notes |
|------|--------|
| Wealth-curve max drawdown | `src/evaluation/economic.py` |
| Rescore script | `scripts/rescore_from_pack.py` |
| Early-stop / target diagnosis | `scripts/diagnose_early_stop.py` |
| Known issues | `corr(y,q50)≈0`; gates near-uniform; early best_epoch |

**Exit criteria:** Every new pack gets rescore + diagnosis + short verdict before architecture changes.

---

### Stage 5 — Controlled research upgrades (current phase)
**Status:** `IN PROGRESS`  

Execute **one at a time**, full walk-forward, fair MTP vs single-TF.

| # | Upgrade | Status | Notes |
|---|---------|--------|--------|
| 5.1 | `target_clip=5.0` | `DONE` | Label hygiene; multi-TF still loses |
| 5.2 | Milder optim: lr=1e-4, wd=5e-4, cosine | `IMPLEMENTED` / **re-run NEXT** | Code on `main` @ milder optim commit; need Kaggle measure |
| 5.3 | Slightly higher dropout (e.g. 0.15–0.2) | `PENDING` | Only if 5.2 fails or partially helps |
| 5.4 | Offline signal-threshold sweep on fixed q50 | `PENDING` | No retrain; economics only |
| 5.5 | Gate temperature / entropy retune | `PENDING` | Only if val curves improve after 5.2–5.3 |
| 5.6 | Target clip ablation (±8 / off) | `PENDING` | Optional after optim story is clear |
| 5.7 | Capacity / cross-attention | `BLOCKED` | Unlock only if 5.1–5.5 exhausted and single-TF still dominates for wrong reasons |

**Stage 5 exit (research fork):**  
- **PASS multi-TF:** mean MTP pinball &lt; single-TF on ≥ majority of folds and on mean → go to Stage 6.  
- **FAIL multi-TF:** freeze multi-TF thesis as negative/inconclusive; promote **single-TF** (or next hypothesis) as paper/product champion → Stage 5b / 6 adjusted.

**Current next action:** Kaggle full run with milder optim (5.2); place pack under `Results/`; update verdict + this plan.

---

### Stage 5b — Pivot / champion selection
**Status:** `PENDING` (triggered if multi-TF never wins)  

Options (pick with evidence, not preference):  
1. Publish multi-TF as **negative/ablation** result + strong single-TF baseline  
2. New multi-TF hypothesis (learned gates with different objective, hierarchical TFs, etc.)—new stage set  
3. Product path on **single-TF** while research continues separately  

---

### Stage 6 — Multi-asset / multi-regime evidence
**Status:** `PENDING`  

Only after Stage 5 research bar is decided.

| Item | Notes |
|------|--------|
| More pairs | GBPUSD, USDJPY, XAUUSD already in data—run same protocol |
| Longer / more folds | Broader time coverage than 2016–2018 windows only |
| Robustness | Seeds, cost sensitivity, horizon breakdown |
| Optional | Macro/context ablation (`use_context` on/off) |

**Exit criteria:** Results stable enough for paper tables (or clearly scoped limitations).

---

### Stage 7 — Paper-ready package
**Status:** `PENDING`  

| Item | Notes |
|------|--------|
| Figures | Training curves, gates, pinball vs baselines, coverage, econ (fixed) |
| Tables | Per-fold + aggregate; single-TF primary comparison |
| Repro | Tag GitHub release; freeze config YAMLs used in tables |
| Claims discipline | No trading claims without fixed econ metrics and costs |

**Exit criteria:** Draft manuscript + supplementary configs/results.

---

### Stage 8 — Publication submission
**Status:** `PENDING`  

| Item | Notes |
|------|--------|
| Venue choice | Quant finance / ML for time series / workshop—decide later |
| Reviews | Address leakage, baselines, economic interpretation carefully |
| Camera-ready | Final code tag |

**Exit criteria:** Submitted (then accepted/rejected handled as process, not plan rewrite).

---

### Stage 9 — Product deployment
**Status:** `PENDING` (can start light parallel work after Stage 5 champion is clear)  

| Phase | Deliverable |
|-------|-------------|
| 9.1 Champion freeze | Config + weights + metrics for chosen model (multi-TF or single-TF) |
| 9.2 Batch inference | Script/service: latest bars → quantiles + optional signal |
| 9.3 Data ops | Aligned-data refresh pipeline; schema checks |
| 9.4 Monitoring | Pinball/coverage drift, latency, NaNs, gate collapse alerts |
| 9.5 Risk layer | Thresholds, sizing, max DD kill-switch, paper trading first |
| 9.6 Deploy | API or internal batch job; secrets; rollback |
| 9.7 Live evaluation | Shadow mode → limited capital → scale only if metrics hold |

**Exit criteria:** Documented runbook + monitored deployment; product metrics separate from paper metrics.

---

## 5. Current position (snapshot)

| Field | Value |
|-------|--------|
| Active stage | **5.2** — milder optim implemented; **Kaggle re-run pending** |
| Champion model (default) | **Single-TF** until multi-TF wins fair bar |
| Multi-TF thesis | **Not supported yet** (2 full runs) |
| Blocking issues | Early best_epoch ~1–2; weak y–q50 correlation; near-uniform gates |
| Do next | (1) Kaggle run with latest `main` (2) rescore+diagnose (3) update Findings log + stage status |

---

## 6. Findings log (append-only)

| Date | Finding | Decision |
|------|---------|----------|
| 2026-08-13 | First Kaggle full run: multi-TF loses to single-TF; DD bug; best epoch 1–2 | Fix econ; do not enlarge model |
| 2026-08-13 | Vol-normalized \|y\| extremes dominate pinball | Add `target_clip=5` |
| 2026-08-14 | Clip re-run: gap narrower but multi-TF still loses; DD sane | Milder optim next (not capacity) |
| 2026-08-14 | Milder optim coded (lr 1e-4, wd 5e-4, cosine) | Measure on Kaggle before dropout |

*(Add rows after every experiment or major decision.)*

---

## 7. Operational checklist (every experiment)

1. Code on GitHub `main` (or tagged commit recorded in CHANGELOG)  
2. Kaggle: Internet ON; prefer **GPU T4** (P100 needs older torch—notebook handles reinstall)  
3. Dataset attached (`mtp-aligned-parquet` or recursive path under `/kaggle/input/datasets/...`)  
4. Download analysis zip → extract under `Results/<exp_name> - <date>/`  
5. Run:  
   - `python scripts/rescore_from_pack.py --pack-dir <path>`  
   - `python scripts/diagnose_early_stop.py --pack-dir <path>`  
6. Write short verdict (MTP vs single-TF, best_epoch, econ, gates)  
7. Update **Findings log** + stage status in this file + `CHANGELOG.md`  

---

## 8. Repo / path map

| Path | Role |
|------|------|
| `Plan.md` | **This file** — staged roadmap |
| `CHANGELOG.md` | Chronological decisions & mistakes |
| `configs/` | Experiment YAMLs (source of truth for knobs) |
| `src/` | Library code |
| `scripts/` | CLI: train, walk-forward, rescore, diagnose |
| `notebooks/kaggle_mtp_walk_forward.ipynb` | Kaggle runner (local only) |
| `data/aligned/` | Parquets (not in git) |
| `Results/` | Downloaded packs + verdicts (not in git) |
| `design/` | Locked design docs (local only) |

---

## 9. What “done” means for the whole program

| End goal | Done when |
|----------|-----------|
| **Publication** | Manuscript submitted with reproducible configs, fair baselines, and claims matching evidence (including negative multi-TF if that remains true) |
| **Product** | Champion model serving under monitoring with risk controls and a retrain/rollback runbook—not merely a good notebook |

Until both rows are true, keep this plan updated; do not invent a new project narrative in chat only.
