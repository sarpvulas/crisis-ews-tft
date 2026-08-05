"""Extract Variable Selection Network (VSN) weights from a trained model."""

import torch
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader


def extract_vsn_weights(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Extract VSN weights from a trained model.

    VSN weights indicate the relative importance of each input variable
    at each timestep.

    Args:
        model: Trained TFT or RA-TFT model.
        dataloader: DataLoader yielding batch dicts.
        device: Device to run inference on.
        max_batches: Limit number of batches.

    Returns:
        Dict with:
            - "historical_weights": (n_samples, encoder_steps, n_hist_features) or
              (n_samples, n_hist_features) depending on VSN output shape
            - "future_weights": (n_samples, decoder_steps, n_future_features) or similar
            - "static_weights": (n_samples, n_static_features)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
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

            if "historical_weights" in output:
                all_hist_w.append(output["historical_weights"].cpu().numpy())
            if "future_weights" in output:
                all_future_w.append(output["future_weights"].cpu().numpy())
            if "static_weights" in output:
                all_static_w.append(output["static_weights"].cpu().numpy())

    result = {}
    if all_hist_w:
        result["historical_weights"] = np.concatenate(all_hist_w, axis=0)
    if all_future_w:
        result["future_weights"] = np.concatenate(all_future_w, axis=0)
    if all_static_w:
        result["static_weights"] = np.concatenate(all_static_w, axis=0)

    return result


def aggregate_vsn_importance(
    vsn_weights: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Aggregate VSN weights into per-feature importance scores.

    Args:
        vsn_weights: Array of shape (n_samples, [timesteps,] n_features).
        feature_names: Names corresponding to the last dimension.

    Returns:
        Dict mapping feature name to mean importance.
    """
    # Average over all dims except features (last)
    mean_weights = vsn_weights.mean(axis=tuple(range(vsn_weights.ndim - 1)))
    return {name: float(w) for name, w in zip(feature_names, mean_weights)}
