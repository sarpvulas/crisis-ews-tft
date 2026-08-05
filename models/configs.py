from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TFTConfig:
    """Configuration for TFT and RA-TFT models."""
    # Data shape
    num_historical_numeric: int = 0
    num_historical_categorical: int = 0
    historical_categorical_cardinalities: List[int] = field(default_factory=list)
    num_static_numeric: int = 0
    num_static_categorical: int = 0
    static_categorical_cardinalities: List[int] = field(default_factory=list)
    num_future_numeric: int = 0
    num_future_categorical: int = 0
    future_categorical_cardinalities: List[int] = field(default_factory=list)

    # Architecture
    state_size: int = 64
    hidden_size: int = 64
    attention_heads: int = 4
    lstm_layers: int = 2
    dropout: float = 0.1

    # Sequence
    encoder_steps: int = 60
    decoder_steps: int = 12

    # Output
    task_type: str = "ews"  # "ews" (early warning) or "regression"
    num_outputs: int = 1    # single sigmoid output for P(crisis onset)

    # Regime (RA-TFT only)
    num_regime_states: int = 3
    use_regime_module: bool = False
    use_regime_attention: bool = False

    # Novel TFT-on-top ablations (all default OFF = vanilla behaviour)
    use_aux: bool = False          # auxiliary forward-drawdown regression head (multi-task)
    lambda_aux: float = 0.5        # weight on the aux MSE term (consumed by CrisisAwareLoss)
    use_regime_vsn: bool = False   # regime-conditioned variable selection

    @property
    def total_static_inputs(self) -> int:
        return self.num_static_numeric + self.num_static_categorical

    @property
    def total_historical_inputs(self) -> int:
        return self.num_historical_numeric + self.num_historical_categorical

    @property
    def total_future_inputs(self) -> int:
        return self.num_future_numeric + self.num_future_categorical
