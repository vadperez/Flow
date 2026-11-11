"""Fitting the function layer to a frozen core (pipeline steps S3.2 and S6.1).

Supervision comes from the traces: where a call happens, which operation, and which earlier
values its arguments point at. The core's weights never change here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..data.align import RES, Aligned
from ..typesys.values import Kind
from .model import ControllerBlock, FunctionHeads


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    operations: "tuple[str, ...]"
    arg_kinds: "tuple[Kind, ...]"
    result_kind: Kind

    @staticmethod
    def of(body) -> "FunctionSpec":
        return FunctionSpec(body.function, tuple(body.operations), tuple(body.signature.args), body.signature.result)


def collate(batch: "list[Aligned]", bos: int, pad: int, device=None) -> "tuple[torch.Tensor, torch.Tensor]":
    """Token ids with <bos> in front, padded on the right. Returns (input_ids [B, T], pad_mask [B, T])."""
    length = max(len(a.ids) for a in batch) + 1
    ids = torch.full((len(batch), length), pad, dtype=torch.long)
    for b, a in enumerate(batch):
        ids[b, 0] = bos
        ids[b, 1:len(a.ids) + 1] = torch.tensor(a.ids)
    return ids.to(device), (ids == pad).to(device)


@torch.no_grad()
def core_hidden(core, input_ids: torch.Tensor, pad_mask: torch.Tensor):
    """All hidden states of the frozen core."""
    out = core.model(input_ids=input_ids, attention_mask=(~pad_mask).long(), output_hidden_states=True, use_cache=False)
    return out.hidden_states


def candidates(a: Aligned, position: int, kind: Kind) -> "list[int]":
    """Indices of the anchors a binder may point at: typed values at or before the call, of the wanted type."""
    return [i for i, anchor in enumerate(a.anchors) if anchor.token <= position and anchor.value.kind is kind]


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy in which the positive entries together weigh as much as the negative ones."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = target > 0, target == 0
    total = loss[negative].mean() if negative.any() else loss.new_zeros(())
    return total + (loss[positive].mean() if positive.any() else 0.0)


def losses(controller: ControllerBlock, heads: "dict[str, FunctionHeads]", specs: "dict[str, FunctionSpec]",
           batch: "list[Aligned]", hidden, pad_mask: torch.Tensor):
    """Trigger and binder losses of the functions in `heads` on one batch. Returns (loss, counts).

    Kinds of trigger examples, each with its own average so that none drowns the others:
      plain positions      every position where no call belongs: no call
      decisions            every call step at every call site, including the step after the last
                           call and the sites of other functions: call or stay silent (the silent
                           ones at call sites are the hard negatives)
      operation, final     at this module's own calls: which operation, and does the call
                           complete the chain?
    """
    device = pad_mask.device
    c = controller.features(hidden, pad_mask)  # shared features [B, T, width]
    z = {name: h.features(hidden, pad_mask) for name, h in heads.items()}  # private features per module
    counts = {"pointer_examples": 0, "pointer_correct": 0, "sites": 0, "sites_correct": 0, "false_calls": 0, "plain_positions": 0}
    trigger_loss = c.new_zeros(())
    pointer_loss = c.new_zeros(())

    # Every decision: (sequence, position, call step); earlier results are taken from the trace.
    plain = ~pad_mask
    rows = []
    for b, a in enumerate(batch):
        for s, site in enumerate(a.sites):
            p = site.position + 1  # position in the input, shifted by <bos>
            plain[b, p] = False
            for step in range(len(site.steps) + 1):  # the last one is the step after the final call
                res_kind = Kind.NONE if step == 0 else site.steps[step - 1].result.kind
                current = site.steps[step] if step < len(site.steps) else None
                rows.append((b, p, s, step, current, res_kind, site, a))
    counts["plain_positions"] = int(plain.sum())
    site_ok = {(r[0], r[2]): True for r in rows}

    ctrl0 = controller.ctrl(0, Kind.NONE, device)
    for name, h in heads.items():
        quiet = h.outputs(z[name], c, ctrl0)[0][plain]  # "call" at every plain position
        trigger_loss = trigger_loss + F.binary_cross_entropy_with_logits(quiet, torch.zeros_like(quiet))
        counts["false_calls"] += int((quiet > 0).sum())
    if rows:
        bi = torch.tensor([r[0] for r in rows], device=device)
        pi = torch.tensor([r[1] for r in rows], device=device)
        ctrl = torch.stack([controller.ctrl(r[3], r[5], device) for r in rows])
        for name, h in heads.items():
            call, ops, final = h.outputs(z[name][bi, pi], c[bi, pi], ctrl)  # [N], [N, n_ops], [N]
            mine = torch.zeros(len(rows), dtype=torch.bool, device=device)
            which = torch.zeros(len(rows), dtype=torch.long, device=device)
            last = torch.zeros(len(rows), device=device)
            for i, (_, _, _, step, current, _, site, _) in enumerate(rows):
                if current is not None and current.fn == name:
                    mine[i] = True
                    which[i] = specs[name].operations.index(current.op)
                    last[i] = float(step == len(site.steps) - 1)
            trigger_loss = trigger_loss + balanced_bce(call, mine.float())
            if mine.any():
                trigger_loss = trigger_loss + F.cross_entropy(ops[mine], which[mine])
                trigger_loss = trigger_loss + F.binary_cross_entropy_with_logits(final[mine], last[mine])
            right = ((call > 0) == mine) & (~mine | ((ops.argmax(-1) == which) & ((final > 0) == (last > 0))))
            for i, r in enumerate(rows):
                site_ok[(r[0], r[2])] &= bool(right[i])

    # Binders: one hard pointer per argument, among the typed values before the call and RES.
    for b, p, s, step, current, res_kind, site, a in rows:
        if current is None or current.fn not in heads:
            continue
        queries = heads[current.fn].queries(z[current.fn][b, p], c[b, p], controller.ctrl(step, res_kind, device))
        for k, (pointer, kind) in enumerate(zip(current.pointers, specs[current.fn].arg_kinds)):
            cands = candidates(a, site.position, kind)
            with_res = res_kind is kind
            tokens = torch.tensor([a.anchors[i].token + 1 for i in cands], dtype=torch.long, device=device)
            keys = controller.candidate_keys(c[b], tokens, with_res)
            gold = len(cands) if pointer == RES else cands.index(pointer)
            scores = FunctionHeads.scores(queries[k], keys)
            pointer_loss = pointer_loss + F.cross_entropy(scores[None], torch.tensor([gold], device=device))
            counts["pointer_examples"] += 1
            ok = int(scores.argmax()) == gold
            counts["pointer_correct"] += ok
            site_ok[(b, s)] &= ok
    counts["sites"] = len(site_ok)
    counts["sites_correct"] = sum(site_ok.values())
    loss = trigger_loss + pointer_loss / max(counts["pointer_examples"], 1)
    return loss, counts


def fit(core, controller: ControllerBlock, heads: "dict[str, FunctionHeads]", specs: "dict[str, FunctionSpec]",
        data: "list[Aligned]", bos: int, pad: int, steps: int = 600, batch_size: int = 48, lr: float = 2e-3,
        train_controller: bool = True, seed: int = 0, log=print, log_every: int = 100, device=None) -> "list[dict]":
    """Fit the heads (and, if asked, the controller block) with the core frozen."""
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)
    params = [p for h in heads.values() for p in h.parameters()]
    if train_controller:
        params += list(controller.parameters())
    else:
        for p in controller.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)
    generator = torch.Generator().manual_seed(seed)
    history, running = [], None
    for step in range(1, steps + 1):
        picks = torch.randint(0, len(data), (batch_size,), generator=generator).tolist()
        batch = [data[i] for i in picks]
        input_ids, pad_mask = collate(batch, bos, pad, device)
        hidden = core_hidden(core, input_ids, pad_mask)
        loss, counts = losses(controller, heads, specs, batch, hidden, pad_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        running = counts if running is None else {k: running[k] + v for k, v in counts.items()}
        if step % log_every == 0 or step == steps:
            row = {"step": step, "loss": round(loss.item(), 4),
                   "call_sites_right": round(running["sites_correct"] / max(running["sites"], 1), 4),
                   "pointers_right": round(running["pointer_correct"] / max(running["pointer_examples"], 1), 4),
                   "false_calls_per_1000_positions": round(1000 * running["false_calls"] / max(running["plain_positions"], 1), 3)}
            history.append(row)
            log(row)
            running = None
    for p in controller.parameters():
        p.requires_grad_(True)
    return history
