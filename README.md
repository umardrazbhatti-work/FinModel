# MTP-Transformer

Multi-TF Gated Patch Transformer for financial forecasting research.

## Layout

```
configs/          # YAML experiment configs
src/              # library code (data, models, losses, training, evaluation)
scripts/          # CLI entry points
tests/            # unit tests (leakage, shapes, loss)
requirements.txt
```

## Local setup

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -v
```

Place aligned parquet files in `data/aligned/` (see dataset on Kaggle).

```bash
# Series M-A-4h specialist (current experiment) — local smoke only
python scripts/run_rv_pilot.py --config configs/eurusd_rv_ma_4h.yaml --max-folds 1 --max-epochs 1

# 1h RV champion (already PASS)
python scripts/run_rv_pilot.py --config configs/pilot_eurusd_rv_single_tf.yaml

# gated MTP vs single-TF on RV (already LOSES)
python scripts/run_rv_comparison.py --config configs/eurusd_rv_multi_tf.yaml

# single fold / full walk-forward (legacy return task)
python scripts/train_fold.py --config configs/eurusd_1h.yaml --fold 0
python scripts/run_walk_forward.py --config configs/eurusd_1h.yaml
```

## Kaggle (full experiments — not local)

Full walk-forward runs belong on **Kaggle GPU T4**. Local machine: unit tests + 1-fold/1–2-epoch smoke only.

1. Attach dataset `mtp-aligned-parquet` (`aligned/*.parquet`; notebook searches recursively).
2. Re-upload the local notebook `notebooks/kaggle_mtp_walk_forward.ipynb` (not in git) if mode/config cells changed.
3. `GITHUB_REPO_URL` = `https://github.com/umardrazbhatti-work/FinModel.git`, branch `main`.
4. Internet ON + **GPU T4**, then Run All.
   - P100 / sm_60: notebook reinstalls `torch==2.1.2+cu118`.
5. Default mode `EXPERIMENT_MODE = "rv_ma_4h"`:
   - Script: `scripts/run_rv_pilot.py`
   - Config: `configs/eurusd_rv_ma_4h.yaml`
   - Download: `/kaggle/working/exp_eurusd_rv_ma_4h_pilot_pack.zip`

Other modes: `"rv_pilot"` (1h champion), `"rv_multi_tf"` (gated MTP ablation), `"mtp_return"` (legacy).

## Design (v1 locked)

- Six TFs: 5m, 15m, 30m, 1h, 4h, 1d
- Independent patch + small Transformer encoder per TF
- Static TF gates + entropy regularization
- Quantile heads (0.1, 0.5, 0.9) on 30m / 1h / 4h
- Horizons: 1, 4, 12 bars
- Primary TF: 1h (EURUSD first experiment)
