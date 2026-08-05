import torch
from models.crisis_loss import (
    ews_bce_loss,
    regime_nll_loss,
    CrisisAwareLoss,
)

def test_ews_bce_loss():
    preds = torch.randn(8, 5, 1)  # batch, steps, 1
    targets = torch.randint(0, 2, (8, 5)).float()
    loss = ews_bce_loss(preds, targets, pos_weight=10.0)
    assert loss.item() > 0

def test_regime_nll():
    regime_probs = torch.randn(8, 10, 3).softmax(dim=-1)
    regime_labels = torch.randint(0, 3, (8, 10))
    loss = regime_nll_loss(regime_probs, regime_labels, label_smoothing=0.1)
    assert loss.item() > 0

def test_crisis_aware_loss():
    loss_fn = CrisisAwareLoss(beta=1.0, gamma=0.3)
    outputs = {
        "predicted": torch.randn(8, 5, 1),
        "regime_probs": torch.randn(8, 25, 3).softmax(dim=-1),
    }
    targets = {
        "ews_label": torch.randint(0, 2, (8, 5)).float(),
        "regime_label": torch.randint(0, 3, (8, 25)),
    }
    loss = loss_fn(outputs, targets)
    assert loss.item() > 0

def test_crisis_aware_loss_no_regime():
    """Loss should work without regime module outputs."""
    loss_fn = CrisisAwareLoss(beta=1.0, gamma=0.3)
    outputs = {"predicted": torch.randn(8, 5, 1)}
    targets = {"ews_label": torch.randint(0, 2, (8, 5)).float()}
    loss = loss_fn(outputs, targets)
    assert loss.item() > 0
