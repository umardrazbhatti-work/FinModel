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
# single fold (return task)
python scripts/train_fold.py --config configs/eurusd_1h.yaml --fold 0

# fair multi-TF vs single-TF on realized vol (current Stage 1 experiment)
python scripts/run_rv_comparison.py --config configs/eurusd_rv_multi_tf.yaml

# single-TF realized-vol pilot (already PASS)
python scripts/run_rv_pilot.py --config configs/pilot_eurusd_rv_single_tf.yaml

# full walk-forward (legacy return task)
python scripts/run_walk_forward.py --config configs/eurusd_1h.yaml
```

## Kaggle (~12h full run)

1. Upload `kaggle_upload/mtp_aligned_parquet.zip` as a Kaggle dataset (`aligned/*.parquet`).
2. Upload the local notebook `notebooks/kaggle_mtp_walk_forward.ipynb` to Kaggle (not in this git repo).
3. Set `GITHUB_REPO_URL` (`https://github.com/umardrazbhatti-work/FinModel.git`) in the notebook.
   Dataset paths may appear as `/kaggle/input/datasets/<user>/mtp-aligned-parquet/aligned/` — the notebook searches recursively.
4. Enable **Internet + GPU**, then Run All.
   - Prefer **GPU T4** (works with stock Kaggle PyTorch).
   - **Tesla P100** is sm_60; recent Kaggle torch (cu128) has no kernels for it. The notebook reinstalls `torch==2.1.2+cu118` automatically when needed.
5. Download `/kaggle/working/exp_eurusd_kaggle_12h_analysis_pack.zip` (20–30+ analysis files) for offline review.

Runtime config: `configs/kaggle_eurusd_12h.yaml`
- `max_epochs=60` (ceiling), `early_stopping_patience=12`
- `max_folds=6`, expanding window
- single-TF baseline with `baseline_max_epochs=40`

```bash
python scripts/run_walk_forward.py --config configs/kaggle_eurusd_12h.yaml
```

## Design (v1 locked)

- Six TFs: 5m, 15m, 30m, 1h, 4h, 1d
- Independent patch + small Transformer encoder per TF
- Static TF gates + entropy regularization
- Quantile heads (0.1, 0.5, 0.9) on 30m / 1h / 4h
- Horizons: 1, 4, 12 bars
- Primary TF: 1h (EURUSD first experiment)
