import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def ews_bce_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """Weighted BCE for EWS labels. Heavy positive weight because crises are rare.

    Args:
        preds: Model output logits (batch, decoder_steps, 1).
        targets: Binary EWS labels (batch, decoder_steps).
        pos_weight: Weight for positive class (crisis onset imminent).
    """
    logits = preds.squeeze(-1)
    pw = torch.tensor([pos_weight], device=logits.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def ews_focal_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 10.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for EWS labels (Lin et al. 2017), with positive-class weighting.

    Down-weights easy (well-classified) windows by (1-p_t)^focal_gamma so the
    rare, hard crisis-onset windows dominate the gradient. focal_gamma=0 reduces
    to weighted BCE.

    Args:
        preds: logits (batch, decoder_steps, 1).
        targets: binary EWS labels (batch, decoder_steps).
        pos_weight: positive-class weight (same role as in weighted BCE).
        focal_gamma: focusing parameter (typical 1-3).
    """
    logits = preds.squeeze(-1)
    pw = torch.tensor([pos_weight], device=logits.device, dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    # p_t = model prob assigned to the TRUE class
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal_factor = (1.0 - p_t).clamp(min=1e-6) ** focal_gamma
    return (focal_factor * bce).mean()


def regime_nll_loss(
    regime_probs: torch.Tensor,
    regime_labels: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    num_states = regime_probs.shape[-1]
    smooth_targets = torch.zeros_like(regime_probs)
    smooth_targets.scatter_(-1, regime_labels.unsqueeze(-1), 1.0)
    smooth_targets = smooth_targets * (1 - label_smoothing) + label_smoothing / num_states
    log_probs = torch.log(regime_probs + 1e-8)
    return -(smooth_targets * log_probs).sum(dim=-1).mean()


class CrisisAwareLoss(nn.Module):
    """Combined EWS loss: L = beta * L_ews + gamma * L_regime.

    L_ews: Weighted BCE on crisis onset probability.
    L_regime: Auxiliary regime classification (NLL with label smoothing).
    """

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.3,
        pos_weight: float = 10.0,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        lambda_aux: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.lambda_aux = lambda_aux

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=outputs["predicted"].device)

        # Primary: EWS binary classification (weighted BCE or focal)
        if self.loss_type == "focal":
            l_ews = ews_focal_loss(outputs["predicted"], targets["ews_label"],
                                   self.pos_weight, self.focal_gamma)
        else:
            l_ews = ews_bce_loss(outputs["predicted"], targets["ews_label"], self.pos_weight)
        total_loss = total_loss + self.beta * l_ews

        # Auxiliary: regime detection
        if "regime_probs" in outputs and "regime_label" in targets:
            l_regime = regime_nll_loss(outputs["regime_probs"], targets["regime_label"])
            total_loss = total_loss + self.gamma * l_regime

        # Auxiliary: forward-drawdown regression (multi-task head)
        if "aux_pred" in outputs and "drawdown_target" in targets:
            aux_pred = outputs["aux_pred"].squeeze(-1)
            l_aux = F.mse_loss(aux_pred, targets["drawdown_target"])
            total_loss = total_loss + self.lambda_aux * l_aux

        return total_loss
