"""Extract regime probabilities and transition matrix from a trained RA-TFT model."""

import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader


def extract_regime_probabilities(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Extract regime probabilities and transition matrix from trained RA-TFT.

    Args:
        model: Trained RA-TFT model (must have regime_module).
        dataloader: DataLoader yielding batch dicts.
        device: Device to run inference on.
        max_batches: Limit number of batches.

    Returns:
        Dict with:
            - "regime_probs": (n_samples, seq_len, 3) probabilities per timestep
            - "transition_matrix": (3, 3) learned transition matrix
            - "predicted_regimes": (n_samples, seq_len) argmax regime labels
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = model(batch)

            if "regime_probs" in output:
                all_probs.append(output["regime_probs"].cpu().numpy())

    if not all_probs:
        raise ValueError("Model does not produce regime_probs. Is this an RA-TFT with regime module enabled?")

    regime_probs = np.concatenate(all_probs, axis=0)
    predicted_regimes = np.argmax(regime_probs, axis=-1)

    # Extract transition matrix from model parameters
    transition_matrix = None
    if hasattr(model, "regime_module") and model.regime_module is not None:
        transition_matrix = model.regime_module._get_transition_matrix().cpu().numpy()

    result = {
        "regime_probs": regime_probs,
        "predicted_regimes": predicted_regimes,
    }
    if transition_matrix is not None:
        result["transition_matrix"] = transition_matrix

    return result
