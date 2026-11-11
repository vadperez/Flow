"""Answer-masked pretraining of a core (pipeline step S4.2, in miniature).

The examples come from flow.data.align.lm_example: the label at a call position is <abstain>,
and labels that are result tokens are ignored. Nothing else is special: it is ordinary
next-token training.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..data.align import IGNORE


def pad_batch(examples: "list[tuple[list[int], list[int]]]", pad: int, device=None):
    length = max(len(x) for x, _ in examples)
    ids = torch.full((len(examples), length), pad, dtype=torch.long)
    labels = torch.full((len(examples), length), IGNORE, dtype=torch.long)
    for b, (x, y) in enumerate(examples):
        ids[b, :len(x)] = torch.tensor(x)
        labels[b, :len(y)] = torch.tensor(y)
    return ids.to(device), labels.to(device)


def pretrain(model, examples, pad: int, abstain: int, steps: int, batch_size: int = 64, lr: float = 3e-3,
             warmup: int = 40, seed: int = 0, log=print, log_every: int = 100, device=None) -> "list[dict]":
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rate = lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        picks = torch.randint(0, len(examples), (batch_size,), generator=generator).tolist()
        ids, labels = pad_batch([examples[i] for i in picks], pad, device)
        logits = model(input_ids=ids, attention_mask=(ids != pad).long(), use_cache=False).logits
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]).float(), labels.view(-1), ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % log_every == 0 or step == steps:
            at_calls = labels == abstain
            row = {"step": step, "loss": round(loss.item(), 4),
                   "abstain_at_call_sites": round(float((logits.argmax(-1)[at_calls] == abstain).float().mean()), 4)}
            history.append(row)
            log(row)
    model.eval()
    return history
