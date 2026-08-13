"""Training engine for MTP-Transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import save_checkpoint
from src.utils.logging import get_logger

logger = get_logger("mtp.trainer")


class MTPTrainer:
    """Pure-PyTorch trainer with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        grad_clip: float = 1.0,
        log_every: int = 50,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.scheduler = scheduler
        self.history: List[Dict[str, float]] = []

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(batch)
        out["inputs"] = {k: v.to(self.device) for k, v in batch["inputs"].items()}
        out["context"] = batch["context"].to(self.device)
        out["targets"] = {k: v.to(self.device) for k, v in batch["targets"].items()}
        out["target_mask"] = {
            k: v.to(self.device) for k, v in batch["target_mask"].items()
        }
        if "raw_returns" in batch:
            out["raw_returns"] = {
                k: v.to(self.device) for k, v in batch["raw_returns"].items()
            }
        return out

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "pinball_loss": 0.0, "entropy_loss": 0.0}
        n_batches = 0
        gate_acc: Optional[torch.Tensor] = None

        for step, batch in enumerate(dataloader):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch)
            loss_dict = self.loss_fn(
                predictions=out["predictions"],
                targets=batch["targets"],
                target_masks=batch["target_mask"],
                gate_weights=out["gate_weights"],
            )
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at step %s; skipping batch", step)
                continue
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            totals["loss"] += float(loss.item())
            totals["pinball_loss"] += float(loss_dict["pinball_loss"].item())
            totals["entropy_loss"] += float(loss_dict["entropy_loss"].item())
            n_batches += 1

            gw = out["gate_weights"].detach()
            gate_acc = gw if gate_acc is None else gate_acc + gw

            if self.log_every and step > 0 and step % self.log_every == 0:
                logger.info(
                    "step=%d loss=%.5f pinball=%.5f ent=%.5f",
                    step,
                    loss.item(),
                    loss_dict["pinball_loss"].item(),
                    loss_dict["entropy_loss"].item(),
                )

        if n_batches == 0:
            return {k: float("nan") for k in totals}

        metrics = {k: v / n_batches for k, v in totals.items()}
        if gate_acc is not None:
            mean_gates = (gate_acc / n_batches).cpu().numpy().tolist()
            metrics["gate_weights"] = mean_gates  # type: ignore[assignment]
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        totals = {"loss": 0.0, "pinball_loss": 0.0, "entropy_loss": 0.0}
        n_batches = 0
        gate_acc: Optional[torch.Tensor] = None

        all_preds: Dict[str, List[np.ndarray]] = {}
        all_targets: Dict[str, List[np.ndarray]] = {}
        all_masks: Dict[str, List[np.ndarray]] = {}
        all_raw: Dict[str, List[np.ndarray]] = {}

        for batch in dataloader:
            batch = self._move_batch(batch)
            out = self.model(batch)
            loss_dict = self.loss_fn(
                predictions=out["predictions"],
                targets=batch["targets"],
                target_masks=batch["target_mask"],
                gate_weights=out["gate_weights"],
            )
            totals["loss"] += float(loss_dict["loss"].item())
            totals["pinball_loss"] += float(loss_dict["pinball_loss"].item())
            totals["entropy_loss"] += float(loss_dict["entropy_loss"].item())
            n_batches += 1

            gw = out["gate_weights"].detach()
            gate_acc = gw if gate_acc is None else gate_acc + gw

            for tf, pred in out["predictions"].items():
                all_preds.setdefault(tf, []).append(pred.cpu().numpy())
                all_targets.setdefault(tf, []).append(batch["targets"][tf].cpu().numpy())
                all_masks.setdefault(tf, []).append(batch["target_mask"][tf].cpu().numpy())
                if "raw_returns" in batch:
                    all_raw.setdefault(tf, []).append(batch["raw_returns"][tf].cpu().numpy())

        if n_batches == 0:
            return {"metrics": {k: float("nan") for k in totals}}

        metrics: Dict[str, Any] = {k: v / n_batches for k, v in totals.items()}
        if gate_acc is not None:
            metrics["gate_weights"] = (gate_acc / n_batches).cpu().numpy().tolist()

        predictions = {tf: np.concatenate(v, axis=0) for tf, v in all_preds.items()}
        targets = {tf: np.concatenate(v, axis=0) for tf, v in all_targets.items()}
        masks = {tf: np.concatenate(v, axis=0) for tf, v in all_masks.items()}
        raw_returns = {tf: np.concatenate(v, axis=0) for tf, v in all_raw.items()} if all_raw else {}

        return {
            "metrics": metrics,
            "predictions": predictions,
            "targets": targets,
            "masks": masks,
            "raw_returns": raw_returns,
        }

    def fit_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        fold_id: int = 0,
        extra_checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        best_val = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        patience = 0
        history: List[Dict[str, Any]] = []

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, max_epochs + 1):
            train_m = self.train_epoch(train_loader)
            val_out = self.evaluate(val_loader)
            val_m = val_out["metrics"]

            row = {
                "epoch": epoch,
                "train_loss": train_m.get("loss", float("nan")),
                "train_pinball": train_m.get("pinball_loss", float("nan")),
                "val_loss": val_m.get("loss", float("nan")),
                "val_pinball": val_m.get("pinball_loss", float("nan")),
                "gate_weights": val_m.get("gate_weights", train_m.get("gate_weights")),
            }
            history.append(row)
            logger.info(
                "Fold %d Epoch %d | train_pinball=%.5f val_pinball=%.5f gates=%s",
                fold_id,
                epoch,
                row["train_pinball"],
                row["val_pinball"],
                row["gate_weights"],
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(row["val_pinball"])
                else:
                    self.scheduler.step()

            val_score = row["val_pinball"]
            if np.isfinite(val_score) and val_score < best_val:
                best_val = val_score
                patience = 0
                best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_pinball": best_val,
                    "fold_id": fold_id,
                }
                if extra_checkpoint:
                    best_state.update(extra_checkpoint)
                if checkpoint_dir is not None:
                    save_checkpoint(best_state, Path(checkpoint_dir) / "best.pt")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (best val_pinball=%.5f)",
                        epoch,
                        best_val,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])

        return {
            "best_val_pinball": best_val,
            "history": history,
            "best_epoch": best_state["epoch"] if best_state else None,
            "gate_weights": history[-1]["gate_weights"] if history else None,
        }
