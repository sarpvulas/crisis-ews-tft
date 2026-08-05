import torch
import torch.nn as nn
from models.components.grn import GatedResidualNetwork


class RegimeDetectionModule(nn.Module):
    """Neural HMM with time-varying transitions and differentiable forward algorithm.

    3 states: calm(0) / stress(1) / crisis(2).

    The transition matrix is conditioned on the current hidden state at each timestep,
    allowing the model to learn that regime transitions depend on market conditions
    (e.g., higher probability of calm→stress when volatility is rising).
    """

    def __init__(self, input_dim: int, num_states: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_states = num_states

        # Emission network: maps LSTM hidden state to per-state log-likelihoods
        self.emission_grn = GatedResidualNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=num_states, dropout=0.1,
        )

        # Time-varying transition network: maps hidden state to transition matrix
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_states * num_states),
        )

        # Initialize transition network to output sticky diagonal matrix
        # bias of last layer → high diagonal, low off-diagonal
        with torch.no_grad():
            self.transition_net[-1].weight.fill_(0.0)
            bias = torch.zeros(num_states * num_states)
            for i in range(num_states):
                bias[i * num_states + i] = 3.0       # diagonal → softmax ~0.95
                for j in range(num_states):
                    if i != j:
                        bias[i * num_states + j] = -3.0  # off-diagonal → softmax ~0.025
            self.transition_net[-1].bias.copy_(bias)

        # Initial state distribution (log-space)
        self.initial_logits = nn.Parameter(torch.zeros(num_states))

    def _get_initial_dist(self) -> torch.Tensor:
        return torch.softmax(self.initial_logits, dim=-1)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch, seq_len, input_dim) -- LSTM encoder output
        Returns:
            regime_probs: (batch, seq_len, num_states) -- P(state_t | h_1:t)
            transition_matrices: (batch, seq_len, num_states, num_states)
        """
        batch_size, seq_len, _ = hidden_states.shape
        initial = self._get_initial_dist()

        # Emission log-likelihoods
        emission_logits = self.emission_grn(hidden_states)  # (batch, seq, num_states)
        log_emissions = torch.log_softmax(emission_logits, dim=-1)

        # Compute time-varying transition matrices for all timesteps
        trans_logits = self.transition_net(hidden_states)  # (batch, seq, states*states)
        trans_logits = trans_logits.view(batch_size, seq_len, self.num_states, self.num_states)
        trans_matrices = torch.softmax(trans_logits, dim=-1)  # softmax per row
        log_trans_all = torch.log(trans_matrices + 1e-8)

        # Forward algorithm (differentiable, in log-space)
        log_alpha = torch.log(initial).unsqueeze(0) + log_emissions[:, 0, :]  # (batch, states)
        all_log_alphas = [log_alpha]

        for t in range(1, seq_len):
            log_trans_t = log_trans_all[:, t - 1, :, :]  # (batch, states, states)
            # log_alpha_expanded: (batch, states, 1) + (batch, states, states) → (batch, states, states)
            log_alpha_expanded = log_alpha.unsqueeze(-1) + log_trans_t
            log_alpha = torch.logsumexp(log_alpha_expanded, dim=-2) + log_emissions[:, t, :]
            all_log_alphas.append(log_alpha)

        # Stack and normalize to get posterior probabilities
        log_alphas = torch.stack(all_log_alphas, dim=1)  # (batch, seq, states)
        log_normalizer = torch.logsumexp(log_alphas, dim=-1, keepdim=True)
        regime_probs = torch.exp(log_alphas - log_normalizer)

        # Return average transition matrix for interpretability
        avg_trans = trans_matrices.mean(dim=1).mean(dim=0)  # (states, states)

        return regime_probs, avg_trans
