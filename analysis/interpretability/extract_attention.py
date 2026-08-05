"""Extract attention weights from a trained RA-TFT or TFT model."""

import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader


def extract_attention_weights(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Extract attention weights from a trained model.

    Args:
        model: Trained TFT or RA-TFT model.
        dataloader: DataLoader yielding batch dicts.
        device: Device to run inference on.
        max_batches: Limit number of batches (for speed).

    Returns:
        Dict with:
            - "attention_scores": (n_samples, num_heads, decoder_steps, total_steps)
            - "historical_weights": (n_samples, num_historical_features)
            - "future_weights": (n_samples, num_future_features)
            - "static_weights": (n_samples, num_static_features)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_attn = []
    all_hist_w = []
    all_future_w = []
    all_static_w = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = model(batch)

            all_attn.append(output["attention_scores"].cpu().numpy())

            if "historical_weights" in output:
                all_hist_w.append(output["historical_weights"].cpu().numpy())
            if "future_weights" in output:
                all_future_w.append(output["future_weights"].cpu().numpy())
            if "static_weights" in output:
                all_static_w.append(output["static_weights"].cpu().numpy())

    result = {
        "attention_scores": np.concatenate(all_attn, axis=0),
    }
    if all_hist_w:
        result["historical_weights"] = np.concatenate(all_hist_w, axis=0)
    if all_future_w:
        result["future_weights"] = np.concatenate(all_future_w, axis=0)
    if all_static_w:
        result["static_weights"] = np.concatenate(all_static_w, axis=0)

    return result
