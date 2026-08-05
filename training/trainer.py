"""Unified training loop for crisis prediction models."""

import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    """Trainer for TFT / RA-TFT crisis prediction models.

    Supports:
    - Standard PyTorch training loop with gradient clipping
    - ReduceLROnPlateau scheduler
    - Best model checkpoint saving based on validation loss
    - Optional W&B logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataset,
        val_dataset,
        loss_fn: nn.Module,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.use_wandb = use_wandb and wandb is not None
        self.device = config.get("device", "cpu")
        self.task = config.get("task", "B")

        # Training params
        self.lr = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 64)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        # Which head the model is scored by. A regression-only run ("aux") has no
        # trained classification head, so selecting on classifier PR-AUC would
        # checkpoint on noise; it is selected on its own regression objective.
        self.score_from = config.get("predict_from", "clf")

        # Optimizer and scheduler. The scheduler tracks the SELECTION metric
        # (val PR-AUC, mode="max"), so LR drops when ranking plateaus rather
        # than when the (often-exploding) val loss does.
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5,
        )

        self.model.to(self.device)

        # W&B init
        if self.use_wandb:
            wb_project = wandb_config.pop("project", "crisis-prediction") if wandb_config else "crisis-prediction"
            wandb.init(
                project=wb_project,
                config={**config},
            )

    def train(self, epochs: int) -> Dict[str, list]:
        """Train the model for a given number of epochs.

        Returns:
            History dict with keys: train_loss, val_loss, lr
        """
        # Seeded generator so the shuffle order is reproducible across runs
        # (the global RNG is otherwise perturbed by model-init / dataset order).
        g = torch.Generator()
        g.manual_seed(int(self.config.get("seed", 0)))
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            collate_fn=self.train_dataset.collate_fn,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
        )

        history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "lr": []}
        best_val_pr_auc = -float("inf")
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_dict, targets_dict in train_loader:
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

                self.optimizer.zero_grad()
                outputs = self.model(batch_dict)
                loss = self.loss_fn(outputs, targets_dict)
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # --- Validation (loss + PR-AUC; PR-AUC drives selection) ---
            val_loss, val_pr_auc = self._validate(val_loader)

            # --- Scheduler step on the selection metric ---
            self.scheduler.step(val_pr_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Record history ---
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_pr_auc"].append(val_pr_auc)
            history["lr"].append(current_lr)

            # --- Log progress (parseable by dashboard runner) ---
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_pr_auc: {val_pr_auc:.4f}, lr: {current_lr:.6f}")

            # --- Checkpoint + early stopping on MAX val PR-AUC ---
            # val_loss is a poor selector here: it explodes from epoch 1 as the
            # model grows overconfident, while val PR-AUC (ranking) keeps rising.
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, val_pr_auc)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {self.early_stopping_patience} epochs)")
                break

            # --- W&B logging ---
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                })

        # --- Restore best-val checkpoint so downstream eval uses the SELECTED
        #     model, not the last (often worst-val) epoch ---
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Restored best checkpoint: epoch {ckpt.get('epoch')}, "
                  f"val_pr_auc {ckpt.get('val_pr_auc')}, val_loss {ckpt.get('val_loss')}")

        return history

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader):
        """Compute (avg validation loss, val PR-AUC).

        PR-AUC (average precision) is the model-selection metric: it tracks
        ranking quality, which keeps improving while the pos-weighted BCE val
        loss explodes from growing overconfidence on the regime-shifted val set.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        probs_all = []
        labels_all = []

        for batch_dict, targets_dict in val_loader:
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

            outputs = self.model(batch_dict)
            loss = self.loss_fn(outputs, targets_dict)
            total_loss += loss.item()
            n_batches += 1

            logits = outputs["predicted"].squeeze(-1)
            probs_all.append(torch.sigmoid(logits).reshape(-1).cpu())
            labels_all.append(targets_dict["ews_label"].reshape(-1).cpu())

        avg_loss = total_loss / max(n_batches, 1)

        # Regression-only: the classification head is untrained, so its PR-AUC is
        # meaningless. Select on the aux MSE instead, negated to keep the
        # "higher is better" convention used by the scheduler and early stopping.
        if self.score_from == "aux":
            return avg_loss, -avg_loss

        val_pr_auc = float("nan")
        try:
            import numpy as np
            from sklearn.metrics import average_precision_score
            y = torch.cat(labels_all).numpy()
            p = torch.cat(probs_all).numpy()
            if np.unique(y).size > 1:
                val_pr_auc = float(average_precision_score(y, p))
        except Exception:
            pass

        return avg_loss, val_pr_auc

    def _save_checkpoint(self, epoch: int, val_loss: float, val_pr_auc: float = float("nan")) -> None:
        """Save model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        # Save model config (TFTConfig) if available
        from dataclasses import asdict
        model_config = {}
        if hasattr(self.model, "config"):
            model_config = asdict(self.model.config)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "config": self.config,
            "model_config": model_config,
        }, path)

    def evaluate(self, dataset, task: Optional[str] = None) -> Dict[str, float]:
        """Run full evaluation metrics on a dataset."""
        from training.evaluation import evaluate_model
        return evaluate_model(
            self.model, dataset, task=task or self.task, device=self.device
        )
