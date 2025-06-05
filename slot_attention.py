import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm

def _sinkhorn(cost: torch.Tensor, reg: float = 0.05, n_iters: int = 30):
    """
    Simple log-space Sinkhorn that returns a doubly-stochastic plan
    of shape (B, N, K), where rows sum to 1 and columns sum to 1.
    """
    B, N, K = cost.shape
    log_P = -cost / reg          # initialise with exp(−cost/ε) in log-space

    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)   # normalise cols
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)   # normalise rows

    return log_P.exp()                                               # (B,N,K)

def transport_with_noise(cost: torch.Tensor, reg: float = 0.05,
                         noise_std: float = 1e-4, n_iters: int = 30):
    """Tie-breaking Sinkhorn: add tiny i.i.d. Gaussian noise to the cost matrix."""
    noisy_cost = cost + noise_std * torch.randn_like(cost)

    return _sinkhorn(noisy_cost, reg=reg, n_iters=n_iters)

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, base_sigma = 0, temperature = None, 
                 sink_reg = 0.05, noise_std = 1e-4, decay_sinkhorn = False):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.base_sigma = base_sigma

        self.sink_reg = sink_reg
        self.noise_std = noise_std
        self.decay_sinkhorn = decay_sinkhorn

        self.slot_mha = MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.slot_ln = LayerNorm(dim)

        self.step_number = 1

        # self.scale = dim ** -0.5
        # self.additional_scale = 1/0.25
        # self.temperature = 0.5
        # self.temperature = 2

        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = dim ** 0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        # self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)

        sigma = self.base_sigma + F.softplus(self.log_sigma)
        sigma = sigma.expand(b, n_s, -1)

        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        
        if self.decay_sinkhorn:
            # t_frac = max(0.0, (step - 10_000) / 10_000)      # ramp from 10 k → 20 k
            t_frac = self.step_number / 20000
            if t_frac > 1.0:
                t_frac = 1.0
        else:
            t_frac = 0

        current_sink_reg = self.sink_reg * (1.0 - 0.9 * t_frac)
        current_noise_std = self.noise_std * (1.0 - 0.9 * t_frac)
        # current_sink_reg = 0.045 * (1.0 - 0.9 * t_frac)
        # current_noise_std = 8e-5 * (1.0 - 0.9 * t_frac)
        self.step_number += 1

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale * self.additional_scale

            # dots = torch.einsum('bid,bjd->bij', q, k) / self.temperature
            # attn = dots.softmax(dim=1) + self.eps
            # attn = attn / attn.sum(dim=-1, keepdim=True)

            dots = torch.einsum('bid,bjd->bij', q, k) / self.temperature         # (B, S, N)
            cost = -dots.transpose(1, 2)                                         # (B, N, S)
            attn = transport_with_noise(cost,
                                        reg=current_sink_reg,
                                        noise_std=current_noise_std)                # (B, N, S)
            attn = attn.transpose(1, 2)                                          # (B, S, N)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
            slots = slots + self.slot_mha(slots, slots, slots, need_weights=False)[0]
            slots = self.slot_ln(slots)

        return slots
    