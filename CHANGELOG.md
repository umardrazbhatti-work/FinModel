# Changelog

All notable changes, amendments, mistakes, and operational decisions for this project.

**Purpose:** Prevent repeating the same mistakes weeks later. Update this file with **every** change.

Format per entry:
- **Date**
- **What changed**
- **Why**
- **Mistakes / pitfalls / lessons** (if any)

---

## 2026-08-17 — Module 1 / S-2 measured (PASS) — first Signal

### Decision (Kaggle vs local)
S-2 is a **label/model sweep**, not a Transformer train. It belongs on this machine (same as S-1). Bundling S-2+S-3+nets into one Kaggle session would mix changes. **Do not run Kaggle for this pack.**

### What happened
Local replay (~7 s): horizons {4, 12, 24} × k {0,1,2,3} × persist / logistic-OHLC / logistic+events + oracle ceiling. Same 6 folds, 2-way 1-pip cost, non-overlapping holds.  
Pack: `Results/exp_signal_s2_eurusd_1h - 17-08-26 1623Hrs/`

| Check | Result |
|-------|--------|
| Official S-2 gate | **PASS** |
| Locked Signal | **`h12_k2_logistic_ohlc`** |
| Mean exp / folds | **+8.87e-5** / **5/6** |
| 12h always-long / coin-flip | −3.1e-4 / −4.5e-4 |
| 4h any model | FAIL |
| Persist any H | FAIL |
| Events at 12h | FAIL (hurt) |
| Oracle 12h | +21.6e-4 (ceiling; not a Signal) |

### Decision
- First Signal is locked. Do not swap to 24h+events as default.
- Next default = **S-3** (Handler sizes this Signal) — local.
- Kaggle only if the next *single* question is “can a net beat this logistic on 12h k=2?”

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Oracle looks like a trading system | Ceiling only. Never a winner. |
| Seven winning keys so train seven nets | One champion. 12h k=2 OHLC logistic. |
| Event 24h has higher exp so switch | Fewer trades, extra features, weaker fold fraction. Secondary. |
| Short-heavy mix = “always short EUR” | 2017–18 window caveat. Do not overclaim. |

---

## 2026-08-17 — Module 1 / S-2 implemented (full label sweep)

### What changed
1. `src/signals/labels.py`, `features.py`, `logistic.py`, `s2_eval.py`
2. Config `configs/signal_s2_eurusd_1h.yaml`
3. Runner `scripts/run_signal_s2.py`
4. Tests `tests/test_signal_s2.py`

### How to run
```bash
python -m pytest tests/test_signal_s2.py -q
python scripts/run_signal_s2.py --config configs/signal_s2_eurusd_1h.yaml
```

---

## 2026-08-17 — Module 1 / S-1 measured (FAIL)

### What happened
Local replay (no GPU, ~5 s) of 12 explicit EURUSD 1h rules on the same 6 expanding folds as the locked handler. Cost = 0.0001 one-way. Pack:  
`Results/exp_signal_s1_eurusd_1h - 17-08-26 1615Hrs/`

| Check | Result |
|-------|--------|
| Official S-1 gate | **FAIL** |
| Winning rules | **none** |
| Best non-control | `tod_train_hours` exp **−5.6e-6** (0/6 folds > 0) |
| always_long | −1.24e-5 (1/6) |
| coin_flip | −1.07e-4 (0/6) |
| always_flat | 0 (best of the set) |

### Decision
- These rules are **not** Signals. Do not attach the locked Handler.
- Next = **S-2** (longer-horizon / large-move direction). Not more RV clocks.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Least-negative rule looks like a near-win | `tod_train_hours` is almost always flat. Closer to zero ≠ edge. |
| Attach handler to “shrink losses” | Forbidden until Signal expectancy > 0 alone. |
| Unicode arrows in Windows logs | Use ASCII `->` (same old pitfall). |

---

## 2026-08-17 — Module 1 / S-1 implemented (costed rule baselines)

### What changed
1. `src/signals/` — explicit long/short/flat rules + `signal_verdict`
2. Config `configs/signal_s1_eurusd_1h.yaml`
3. Runner `scripts/run_signal_s1.py` (replay only; same WF as 1h handler)
4. Tests `tests/test_signal_s1.py`

### Why
Handler is locked. Super goal needs a Signal with edge after costs. Rule baselines are the bar a learned Signal must beat.

### How to run
```bash
python -m pytest tests/test_signal_s1.py -q
python scripts/run_signal_s1.py --config configs/signal_s1_eurusd_1h.yaml
```

---

## 2026-08-17 — Module 2 leftover: M-A-15m measured (FAIL)

### What happened
Kaggle T4 full run of `scripts/run_rv_pilot.py` + `configs/eurusd_rv_ma_15m.yaml`. Pack:  
`Results/exp_eurusd_rv_ma_15m_pilot_pack - 17-08-26 1600Hrs/` (~7 min, rc=0).

| Check | Result |
|-------|--------|
| Official specialist gate | **FAIL** |
| Mean primary H=12 (3h) corr | **0.454** (bar 0.15 — clears) |
| Folds pass corr | **6/6** (min 0.257) |
| Mean pinball net / hist-mean / HAR | 0.150 / 0.262 / **0.135** |
| HAR primary corr | **0.417** |
| Folds net wins pinball | 3/6 |
| Best epoch | 11–16 |

### Decision
- **Park the 15m net.** Do not load it into Handler v1.
- Locked handler stays **EURUSD 1h RV**.
- 15m *series* has skill (HAR). Lagged HAR/RV **scalars** may be a later Module-2 upgrade — not this checkpoint.
- **Next remains Module 1 / S-1.** Do not start daily or retry 15m.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Corr 0.45 looks like a specialist | Fair bar includes HAR pinball. Ranking without beating HAR is a fail. |
| 3/6 folds beat HAR so “close enough” | Official bar is **mean** pinball. HAR wins the mean. |
| 15m FAIL means reopen daily as the next clock | Handler is locked. Clock hunt is not the main path. |

---

## 2026-08-17 — Architecture lock: 6 modules; Trade Handler locked (Module 2)

### What changed
1. Super goal is now a **self-sufficient automated trading system**. All work must serve it.
2. Official modules: (1) Signal / Alpha, (2) Trade Handler, (3) Execution, (4) Portfolio, (5) Monitoring, (6) Data.
3. **Module 2 locked:** EURUSD 1h single-TF realized-vol is the Trade Handler (risk/sizing only).
   - Code: `src/handler/` (`VolatilityTradeHandler`, inverse-vol `size_from_vol`)
   - Config: `configs/handler_eurusd_1h.yaml`
   - CLI: `scripts/run_handler.py`
   - Tests: `tests/test_trade_handler.py`
   - Handler **never** sets `side`. Signal owns direction.
4. Series M extra clocks (15m / daily / 5m / B/C/D) are **optional later Module-2 upgrades**, not the main path.
5. **Next research = Module 1 (S-1):** costed rule-baseline Signals on EURUSD.

### Why
A profitable auto-trader needs a Signal with edge after costs. The vol model is a finished risk module. Leaving it as an open specialist-hunt blocked the super goal.

### How to run (handler smoke)
```bash
python -m pytest tests/test_trade_handler.py -q
python scripts/run_handler.py --config configs/handler_eurusd_1h.yaml
# optional: --checkpoint path/to/fold_5/best.pt
```

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| More RV clocks = closer to a trader | Clocks improve a sizer, not a Signal. |
| Combine handler with a coin-flip “to get automation” | Combo waits until Signal expectancy > 0 after costs. |
| Handler emits long/short from vol | Forbidden. `side` is always null. |

---

## 2026-08-17 — Series M-A-15m specialist (code ready; full run = Kaggle)

### What changed
1. Config `configs/eurusd_rv_ma_15m.yaml`:
   - EURUSD **15m only** → 15m log-RV, horizons **[4, 12] bars** (1h / 3h wall-clock)
   - Same optim and walk-forward bar counts as the 1h champion
   - HAR windows **[16, 48, 96, 480]** = calendar match to 1h `[4, 12, 24, 120]` (4h / 12h / 1d / 5d)
   - Lookback 64 (locked 15m design default)
2. Local notebook default mode is `rv_ma_15m` (re-upload required)

### Why
M-A-30m PASSed. 15m is the next independent clock in the locked Series M order.

### How to run
**Full run = Kaggle only** (re-upload notebook; `EXPERIMENT_MODE = "rv_ma_15m"`; Internet ON; GPU T4; clone `main`).

Local smoke only:
```bash
python scripts/run_rv_pilot.py --config configs/eurusd_rv_ma_15m.yaml --max-folds 1 --max-epochs 1
```

Download: `/kaggle/working/exp_eurusd_rv_ma_15m_pilot_pack.zip`

### Success / fail
- **PASS** → 15m is a specialist; next is **M-A-1d** (higher clock; also the remaining M-B candidate). M-D stays closed until 15m is scored.
- **FAIL** → park 15m; do not feed it into B/C/D. Next remaining M-A clock is **daily**.

---

## 2026-08-17 — Series M-A-30m measured (PASS)

### What happened
Kaggle T4 full run of `scripts/run_rv_pilot.py` + `configs/eurusd_rv_ma_30m.yaml`. Pack:  
`Results/exp_eurusd_rv_ma_30m_pilot_pack - 17-08-26 1500Hrs/` (~9 min, rc=0).

| Check | Result |
|-------|--------|
| Official M-A gate | **PASS** |
| Mean primary H=12 (6h) corr | **0.452** (bar 0.15) |
| Folds pass corr | **6/6** (min 0.336) |
| Mean pinball net / hist-mean / HAR | **0.126** / 0.227 / 0.130 |
| HAR primary corr | 0.356 (HAR has skill; net still wins) |
| Persistence corr | 0.086 |
| Best epoch | 13–23 (mean ~19) |
| std(pred)/std(y) | 0.53–0.78 |

### Decision
- **30m is a Series M specialist.** Second passing clock after 1h.
- Champion stays **1h single-TF RV** (different target / wall-clock; 30m does not replace it).
- M-D’s “≥2 specialists” gate is now met. **Do not start M-D yet.**
- M-B stays blocked (30m is a *lower* clock; need daily).
- Next = **M-A-15m**. Do not retry 4h.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| 30m corr 0.452 > 1h 0.437 so swap the champion | Different clocks and horizons (6h vs 12h). Product default stays 1h until B/C/D beat it. |
| Fold 1 loses HAR pinball so “not really a pass” | Official bar is **mean** pinball + majority corr. 5/6 pinball, 6/6 corr. Keep fold 1. |
| Jump to M-D now that two specialists exist | One clock per run. Score 15m first (may become specialist #3). |
| Use 30m as an M-B filter on 1h | Hierarchy needs a *higher* TF. 30m is not that. |

---

## 2026-08-16 — Series M-A-30m specialist (code ready; full run = Kaggle)

### What changed
1. Config `configs/eurusd_rv_ma_30m.yaml`:
   - EURUSD **30m only** → 30m log-RV, horizons **[4, 12] bars** (2h / 6h wall-clock)
   - Same optim and walk-forward bar counts as the 1h champion
   - HAR windows **[8, 24, 48, 240]** = calendar match to 1h `[4, 12, 24, 120]` (4h / 12h / 1d / 5d)
   - Lookback 48 (locked 30m design default)
2. Local notebook default mode is `rv_ma_30m` (re-upload required)

### Why
M-A-4h failed the specialist bar. 30m is the next independent clock in the locked Series M order.

### How to run
**Full run = Kaggle only** (re-upload notebook; `EXPERIMENT_MODE = "rv_ma_30m"`; Internet ON; GPU T4; clone `main`).

Local smoke only:
```bash
python scripts/run_rv_pilot.py --config configs/eurusd_rv_ma_30m.yaml --max-folds 1 --max-epochs 1
```

Download: `/kaggle/working/exp_eurusd_rv_ma_30m_pilot_pack.zip`

### Success / fail
- **PASS** → 30m is a specialist; next is **M-A-15m** (then daily). M-D still needs ≥2 specialists (1h already counts).
- **FAIL** → park 30m; do not feed it into B/C/D. 15m/daily still follow the plan only if the first-wave pair (4h+30m) rule is revisited — default: stop the lower-TF wave and consider daily as the remaining higher clock.

---

## 2026-08-16 — Series M-A-4h measured (FAIL)

### What happened
Kaggle T4 full run of `scripts/run_rv_pilot.py` + `configs/eurusd_rv_ma_4h.yaml`. Pack:  
`Results/exp_eurusd_rv_ma_4h_pilot_pack - 16-8-26 1300Hrs/` (~8 min, rc=0).

| Check | Result |
|-------|--------|
| Official M-A gate | **FAIL** |
| Mean primary H=12 (48h) corr | **0.058** (bar 0.15) |
| Folds pass corr | **0/6** |
| Mean pinball net / hist-mean / HAR | 0.139 / 0.218 / **0.122** |
| HAR primary corr | **0.165** (HAR clears the corr bar) |
| Persistence corr | 0.111 |
| Best epoch | 4–25 (mean ~15) |
| std(pred)/std(y) | 0.05–0.25 (collapsed) |

### Decision
- **Park the 4h specialist.** Do not feed this net into M-B / M-C / M-D.
- Champion stays **1h single-TF RV**.
- Next = **M-A-30m**. Do not retry 4h with more capacity or rescaled folds.
- Honest note: secondary H=16h neural corr mean 0.215 — unofficial; primary stays H=48h.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Net beats hist-mean so “4h works” | Fair bar is HAR + corr. HAR wins pinball 5/6. |
| H=16h corr 0.215 looks like a stealth PASS | Locked primary is H=12 bars. Do not move the horizon after seeing the pack. |
| HAR has skill so use the 4h *checkpoint* as a filter | Use HAR’s lesson later as optional lagged-4h **scalars** (M-C), not this failed net. |

---

## 2026-08-16 — Series M-A-4h specialist (code ready; full run = Kaggle)

### What changed
1. Config `configs/eurusd_rv_ma_4h.yaml`:
   - EURUSD **4h only** → 4h log-RV, horizons **[4, 12] bars** (16h / 48h wall-clock)
   - Same optim and walk-forward **bar counts** as the 1h RV champion (60/12, 6 expanding folds)
   - HAR windows **[1, 3, 6, 30]** = calendar match to 1h `[4, 12, 24, 120]` (4h / 12h / 1d / 5d)
   - Lookback 42 (locked 4h design default)
2. `scripts/run_rv_pilot.py` is now the Series M-A specialist runner:
   - HAR-OLS + lagged-RV persistence scored on every fold
   - Wall-clock horizons written into `10_go_nogo_rv_pilot.json` / report
   - Go/nogo: corr > 0.15 + majority folds **and** pinball < hist-mean **and** < HAR
3. Helpers: `tf_bar_hours` / `horizon_wall_clock` / `specialist_rv_verdict`
4. Tests: `tests/test_rv_specialist.py`
5. Local notebook default mode is `rv_ma_4h` (notebook is not on GitHub)

### Why
Gated all-TF MTP lost the fair bar on RV. Series M starts with independent specialists. 4h is the first new clock after the 1h champion.

### How to run
**Full run = Kaggle only** (re-upload local notebook; `EXPERIMENT_MODE = "rv_ma_4h"`; Internet ON; GPU T4; clone `main`).

Local smoke only:
```bash
python scripts/run_rv_pilot.py --config configs/eurusd_rv_ma_4h.yaml --max-folds 1 --max-epochs 1
```

Download: `/kaggle/working/exp_eurusd_rv_ma_4h_pilot_pack.zip`

### Success / fail
- **PASS** → 4h is a Series M specialist; next is **M-A-30m** (not B/C/D yet)
- **FAIL** → park 4h; do not feed it into hierarchy / features / ensemble

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Same walk-forward *bar* counts on 4h are ~4× longer in calendar time | Intentional (same protocol). Record wall-clock in the verdict; do not silently rescale folds. |
| HAR windows copied as `[4, 12, 24, 120]` 4h-bars | That would be weeks, not the 1h calendar recipe. Use `[1, 3, 6, 30]`. |
| Starting 30m / M-B in the same Kaggle session | One clock per run. 30m waits for the 4h pack. |

---

## 2026-08-15 — Fair multi-TF vs single-TF on RV measured (LOSES)

### What happened
Kaggle T4 full run of `scripts/run_rv_comparison.py`. Pack:  
`Results/exp_eurusd_rv_multi_tf_pilot_pack - 15-08-26 1900Hrs/` (~27 min, rc=0).

| Check | Result |
|-------|--------|
| Fair bar (MTP vs single-TF) | **LOSES** |
| Mean pinball MTP / STF / HAR | 0.1240 / **0.1201** / 0.1248 |
| Mean H=12 corr MTP / STF / HAR | 0.399 / **0.432** / 0.292 |
| Folds MTP wins pinball | **0/6** |
| Folds MTP wins corr | 3/6 |
| Single-TF vs standalone pilot | Matches (pilot 0.1202 / 0.437) |
| Mean best epoch MTP / STF | ~6 / ~14 |
| Gates | Near-uniform (~1/6) |

### Decision
- **Champion = single-TF RV.** Fusion does not help on this target with static gates.
- Stage 2 (capacity / cross-attention) and Stage 3 (pairs) stay **blocked**.
- Persistence is a bad vol baseline (corr −0.24); HAR is real but loses to single-TF.
- Paper can use this pack as an honest negative multi-TF ablation on a learnable target.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| “MTP has skill (corr 0.40) so fusion works” | Fair bar is vs single-TF, not vs zero/HAR. 0.40 < 0.43 and 0/6 pinball. |
| Earlier MTP best epoch looks like “faster learning” | It is earlier overfit (5× params, unused gates). |
| Persistence corr negative | Do not use raw last-H RV as the classical champion; HAR is the one that matters. |

---

## 2026-08-15 — Fair multi-TF vs single-TF on realized vol (code)

### What changed
1. Config `configs/eurusd_rv_multi_tf.yaml`:
   - Same log-RV target, H=4/12, folds, optim as the single-TF pilot
   - Inputs: all 6 TFs; `tradable_tfs: [1h]` only
   - No context (second change stays off)
   - Identical epoch/patience for MTP and single-TF (60 / 12)
2. `scripts/run_rv_comparison.py` trains MTP + single-TF on the same loaders and scores both
3. Classical baselines in the same run (`src/baselines/har_rv.py`):
   - lagged-RV persistence
   - HAR-OLS on past log-RV windows {4, 12, 24, 120} + residual quantiles
4. Go/nogo is three-way: **BEATS / MATCHES / LOSES** vs single-TF on pinball + primary corr
5. Shared `corr_and_r2` / `evaluate_rv_skill` in `src/evaluation/metrics.py`
6. Walk-forward loss now uses `data.tradable_tfs` instead of a hardcoded `30m/1h/4h`
7. Local Kaggle notebook default mode is `rv_multi_tf` (notebook is not on GitHub)

### Why
Pilot showed RV is learnable. Next evidence bar is the original thesis: does multi-TF fusion beat single-TF **on that target**, under identical conditions.

### How to run
**Full run = Kaggle only** (re-upload local notebook; `EXPERIMENT_MODE = "rv_multi_tf"`; Internet ON; GPU T4; clone `main`).

Local smoke only:
```bash
python scripts/run_rv_comparison.py --config configs/eurusd_rv_multi_tf.yaml --max-folds 1 --max-epochs 1
```

### Success / fail
- **BEATS** → Stage 2 architecture may open (one change at a time)
- **MATCHES** → fusion not harmful; do not expand architecture yet
- **LOSES** → single-TF remains champion; no pairs / no capacity

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Weaker single-TF budget (40/10) from the return runs | This comparison uses 60/12 for both |
| Adding macro context at the same time | Would bundle two upgrades; context stays off |
| Official PASS vs hist-mean only | Fair bar is single-TF; HAR is extra context |

---

## 2026-08-15 — Single-TF realized-vol pilot measured (PASS)

### What happened
Full Kaggle T4 run of `scripts/run_rv_pilot.py` (Stage 1.4 Option A). Pack at  
`Results/exp_eurusd_rv_single_tf_pilot_pilot_pack - 15-08-26 1600Hrs/` (~9 min, rc=0).

| Check | Result |
|-------|--------|
| Official RV gate (mean corr > 0.15, majority folds) | **PASS** |
| Mean / median primary H=12 corr(q50, log-RV) | **0.437 / 0.429** |
| Folds pass | **6/6** (min 0.353) |
| Mean test pinball vs hist-mean | **0.120 vs 0.213** (~43% lower) |
| Best epoch | 12–33 (not 1–2) |
| H=4 corr (secondary) | 0.43–0.57, R² always > 0 |

### Why it matters
Same stack that produced corr≈0 on return quantiles now extracts OOS vol-ranking skill. Confirms the deep-diagnostic call: **target was the bottleneck, not plumbing**.

### Verdict constraints
- Under-dispersed (std_pred/std_y ≈ 0.4–0.6); H=12 R² negative on 3/6 folds — ranks well, shrinks too much.
- No HAR / lagged-RV baseline yet; do not claim the Transformer beats classical vol.
- No econ; `directional_accuracy=1.0` is an RV>0 artifact — ignore.
- Multi-TF still untested on this target.

### Decision
- Option B (large-move class.) **dropped**.
- Next: **multi-TF vs single-TF on RV**, same folds/optim; add a cheap lagged-RV/HAR baseline in that run.
- Still **no** capacity, cross-attention, or new pairs.

### Mistakes / pitfalls / lessons
| Pitfall | Lesson |
|---------|--------|
| Celebrate zero-baseline smash on log-RV | Zero is a straw man; hist-mean (and soon HAR) is the bar. |
| Cite dir-acc = 100% | Metric is return-signed; meaningless on strictly positive RV. |
| Jump to multi-TF without a vol champion | Repeat of Stage 0 “PASS vs zero, lose vs single-TF”. Ship HAR/lagged-RV in the same comparison run. |

---

## 2026-08-14 — Target pivot: single-TF realized-vol pilot (post deep diagnostic)

### Decision
Deep diagnostic + three Kaggle return-quantile runs show:
- Pipeline, leakage checks, gradients, and capacity are fine (on-batch fit works).
- Short-horizon **vol-normalized return** quantiles have ~**no OOS skill** (corr≈0, dir≈50%; Ridge probe ≈0).
- Multi-TF consistently loses to single-TF under that objective.

**Stop** grinding multi-horizon return quantiles / multi-TF expansion on 1h EURUSD.  
**Next:** change the supervised target; keep architecture fixed.

### What changed (code)
1. `MultiTFDataset` supports `target_type: return | realized_vol`:
   - RV = √mean(r²) over the next H bars on the primary TF
   - Optional `rv_log_transform: true` (default) for stable training scale
   - `tradable_tfs` override for single-TF pilots
2. Config: `configs/pilot_eurusd_rv_single_tf.yaml` (EURUSD 1h only, horizons 4 & 12)
3. Runner: `scripts/run_rv_pilot.py` — **single-TF only**, walk-forward, reports corr(q50,y) + go/nogo
4. Success bar: mean primary-horizon OOS corr > **0.15** and majority folds pass

### How to run
```bash
python scripts/run_rv_pilot.py --config configs/pilot_eurusd_rv_single_tf.yaml
# optional: --max-folds 6 --device cuda
```
Outputs under `outputs/exp_eurusd_rv_single_tf_pilot/` (`00_pilot_report.md`, `10_go_nogo_rv_pilot.json`).

**Kaggle:** local notebook `notebooks/kaggle_mtp_walk_forward.ipynb` defaults to `EXPERIMENT_MODE = "rv_pilot"` (not in git). Re-upload notebook to Kaggle after pull of pilot code.

### Success / fail branching
- **PASS** → plumbing confirmed; may revisit multi-TF **on RV (or new target)** later  
- **FAIL** → try Option B (large-move classification) or rethink features/horizon — still no multi-TF/pairs

### Why
Attack a target more likely to contain signal (volatility clustering) before spending more GPU on multi-TF fusion.

---

## 2026-08-14 — Milder optimization (LR / weight decay / cosine)

### What changed
1. **Training defaults (controlled upgrade B):**
   - `lr`: `0.0003` → **`0.0001`**
   - `weight_decay`: `0.0001` → **`0.0005`**
   - `lr_scheduler`: **`cosine`** (`CosineAnnealingLR`, `eta_min=1e-6`) — was hard-coded `ReduceLROnPlateau`
2. New factory `src/training/optim.py` → `build_optimizer_and_scheduler()`:
   - Supports `cosine` | `plateau` | `none` via config
   - Used by `scripts/run_walk_forward.py` and `scripts/train_fold.py` (MTP + single-TF share same path)
3. Configs updated: `default.yaml`, `eurusd_1h.yaml`, `kaggle_eurusd_12h.yaml`
4. Unit tests: `tests/test_optim.py`

### Why
- After `target_clip=5.0` re-run, val pinball still peaked at epoch 1–2 (fast overfit).
- Milder step size + stronger L2 + smooth LR decay aims to delay overfit without growing model capacity.
- Same optim settings apply to single-TF baseline for a **fair** comparison under the shared protocol.

### How to run (Kaggle)
- Pull latest `main`, re-run notebook; no notebook edits required if it clones GitHub.
- Confirm runtime config / pack `14_hyperparameters.json` shows `lr: 0.0001`, `weight_decay: 0.0005`, and training logs do not error on scheduler step.

### Success criteria (same fair bar)
- Mean MTP test pinball **&lt;** single-TF under same 6 folds
- Preferably later `best_epoch` (not stuck at 1–2) and smaller train–val gap at end
- Do **not** judge by absolute pinball vs the clip-only run alone

### Controlled upgrade queue (status)
1. target_clip=5.0 — done + measured (multi-TF still loses).
2. Milder optim (LR 1e-4, wd 5e-4, cosine) — **done + measured** (best_epoch better; multi-TF still fails).
3. Return-task multi-TF grinding — **stopped** after deep diagnostic.
4. Single-TF realized-vol pilot — **code ready; full Kaggle run pending**.
5. Dropout / signal-threshold on returns — deprioritized while target pivot runs.

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
2. Milder optim (LR 1e-4 / cosine / higher wd) — see **2026-08-14 milder optim** entry (implemented; re-run pending).
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
