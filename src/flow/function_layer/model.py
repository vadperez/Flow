"""The function layer: a shared controller block, and the learned heads of each function module.

    core hidden states --> controller block --> shared features c and the keys of candidate values
            |
            +------------> each resident module's own feature layer --> private features z
                           trigger(z, c, CTRL) --> "call": this module acts now
                                                   "operation": which of its operations
                                                   "final": its result is the answer, no further call needed
                           binder(z, c, CTRL)  --> one hard pointer per argument

The controller block is shared and fitted to one core. A module's heads (its feature layer,
trigger and binder) are small, belong to that module alone, and are loaded and unloaded with it.
The private feature layer reads the core directly, so a module added later does not depend on
what the shared features happened to learn from the modules that were there before it.
Bodies are not here: they are written weights (flow.bodies) and do not depend on the core.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..typesys.values import Kind

KINDS = [Kind.NONE, Kind.NUMBER, Kind.SYMBOL, Kind.FLAG]


class ControllerBlock(nn.Module):
    def __init__(self, core_width: int, n_taps: int, width: int = 128, heads: int = 4, key_dim: int = 64,
                 max_calls: int = 4, max_rank: int = 8) -> None:
        super().__init__()
        self.config = dict(core_width=core_width, n_taps=n_taps, width=width, heads=heads, key_dim=key_dim,
                           max_calls=max_calls, max_rank=max_rank)
        self.width, self.key_dim, self.max_calls, self.max_rank = width, key_dim, max_calls, max_rank
        self.mix = nn.Parameter(torch.zeros(n_taps))  # learned mix of the core's hidden states
        self.inp = nn.Sequential(nn.Linear(core_width, width), nn.LayerNorm(width))
        self.layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.out = nn.LayerNorm(width)
        self.key = nn.Linear(width, key_dim)  # key of a candidate value, from the features at its token
        self.rank = nn.Embedding(max_rank, key_dim)  # 0 = the candidate nearest to the call
        self.res_key = nn.Parameter(torch.randn(key_dim) * 0.02)  # key of the special candidate RES

    @property
    def ctrl_dim(self) -> int:
        return self.max_calls + 1 + len(KINDS)

    def features(self, hidden_states, pad_mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """hidden_states: sequence of [B, T, core_width]; pad_mask: [B, T] True at padding -> [B, T, width]."""
        weights = torch.softmax(self.mix, dim=0)
        mixed = sum(w * h.float() for w, h in zip(weights, hidden_states))
        x = self.inp(mixed)
        padding = None if pad_mask is None else torch.zeros_like(pad_mask, dtype=x.dtype).masked_fill(pad_mask, float("-inf"))
        return self.out(self.layer(x, src_mask=_causal(x.shape[1], x.device), src_key_padding_mask=padding))

    def ctrl(self, step: int, res_kind: Kind, device=None) -> torch.Tensor:
        """CTRL features: how many calls were already made at this position, and the type held in RES."""
        v = torch.zeros(self.ctrl_dim, device=device)
        v[min(step, self.max_calls)] = 1.0
        v[self.max_calls + 1 + KINDS.index(res_kind)] = 1.0
        return v

    def candidate_keys(self, c_seq: torch.Tensor, tokens: torch.Tensor, with_res: bool) -> torch.Tensor:
        """Keys of the candidate values of one call.

        c_seq [T, width]; tokens [A]: token index of each candidate, in text order.
        Returns [A (+1 if with_res), key_dim]; the RES candidate comes last.
        """
        count = tokens.shape[0]
        rank = (count - 1 - torch.arange(count, device=tokens.device)).clamp(max=self.max_rank - 1)
        keys = self.key(c_seq[tokens]) + self.rank(rank) if count else c_seq.new_zeros(0, self.key_dim)
        return torch.cat([keys, self.res_key[None]]) if with_res else keys


def _causal(steps: int, device) -> torch.Tensor:
    return torch.triu(torch.full((steps, steps), float("-inf"), device=device), diagonal=1)


class FunctionHeads(nn.Module):
    """Learned part of one function module: private feature layer, trigger and binder."""

    def __init__(self, core_width: int, n_taps: int, shared_width: int, key_dim: int, ctrl_dim: int, n_operations: int,
                 n_args: int, width: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.config = dict(core_width=core_width, n_taps=n_taps, shared_width=shared_width, key_dim=key_dim, ctrl_dim=ctrl_dim,
                           n_operations=n_operations, n_args=n_args, width=width, heads=heads)
        self.n_args, self.key_dim = n_args, key_dim
        self.mix = nn.Parameter(torch.zeros(n_taps))
        self.inp = nn.Sequential(nn.Linear(core_width, width), nn.LayerNorm(width))
        self.layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.out = nn.LayerNorm(width)
        joint = width + shared_width + ctrl_dim
        # outputs: "call", one per operation, "final"
        self.trigger = nn.Sequential(nn.Linear(joint, 2 * width), nn.GELU(), nn.Linear(2 * width, n_operations + 2))
        self.query = nn.Sequential(nn.Linear(joint, 2 * width), nn.GELU(), nn.Linear(2 * width, n_args * key_dim))

    def features(self, hidden_states, pad_mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """Private features z [B, T, width] from the core's hidden states (causal)."""
        weights = torch.softmax(self.mix, dim=0)
        x = self.inp(sum(w * h.float() for w, h in zip(weights, hidden_states)))
        padding = None if pad_mask is None else torch.zeros_like(pad_mask, dtype=x.dtype).masked_fill(pad_mask, float("-inf"))
        return self.out(self.layer(x, src_mask=_causal(x.shape[1], x.device), src_key_padding_mask=padding))

    def outputs(self, z: torch.Tensor, c: torch.Tensor, ctrl: torch.Tensor):
        """z [..., width], c [..., shared_width], ctrl [ctrl_dim] or [..., ctrl_dim].

        Returns three logits (or groups of logits):
            call       [...]                 whether this module acts now (sigmoid, compared with its threshold)
            operation  [..., n_operations]   which operation, if it acts (the largest wins)
            final      [...]                 whether that call completes the chain

        "Whether" and "which" are separate on purpose: whether a comparison is due does not depend
        on which comparison it is, so a pair of calls that was never shown can still work.
        "final" is read only when the module is called. A chain whose last call is not final is
        abandoned and the model abstains: this is what happens when the module that should
        continue is not loaded.
        """
        out = self.trigger(torch.cat([z, c, ctrl.expand(*c.shape[:-1], -1)], dim=-1))
        return out[..., 0], out[..., 1:-1], out[..., -1]

    def queries(self, z: torch.Tensor, c: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """z [width], c [shared_width], ctrl [ctrl_dim] -> [n_args, key_dim]."""
        return self.query(torch.cat([z, c, ctrl], dim=-1)).view(self.n_args, self.key_dim)

    @staticmethod
    def scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """query [key_dim], keys [A, key_dim] -> [A]."""
        return keys @ query / math.sqrt(query.shape[-1])
