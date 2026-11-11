"""Scoring on records with traces: what is printed at each place where a result is needed."""
from __future__ import annotations

from collections import defaultdict

import torch

from ..data.align import Aligned
from ..typesys import Anchor, Kind
from .loop import FunctionLayer
from .train import collate, core_hidden


@torch.no_grad()
def answers(core, tokenizer, layer: "FunctionLayer | None", data: "list[Aligned]", batch_size: int = 64, device=None):
    """For every call site: the text that would be printed there.

    Returns a list of dicts: record, template, gold, got ("<abstain>" when the core abstains),
    calls (the trace). With layer=None the core answers alone.
    """
    rows = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        input_ids, pad_mask = collate(batch, tokenizer.bos_id, tokenizer.pad_id, device)
        out = core.model(input_ids=input_ids, attention_mask=(~pad_mask).long(), output_hidden_states=True, use_cache=False)
        feats = layer.features(out.hidden_states, pad_mask) if layer is not None and layer.modules else None
        for b, a in enumerate(batch):
            for site in a.sites:
                p = site.position + 1  # <bos> shifts every position by one
                value, calls = None, []
                if feats is not None:
                    anchors = [Anchor(x.token + 1, x.start, x.end, x.value) for x in a.anchors]
                    value, calls = layer.calls_at(feats, b, p, anchors)
                if value is not None:
                    got = value.text()
                else:
                    got = tokenizer.vocab[int(core.lm_head(out.last_hidden_state[b, p]).argmax())]
                s, e = a.record["result_spans"][a.sites.index(site)]
                rows.append({"id": a.record["id"], "template": a.record["template"], "gold": a.record["text"][s:e],
                             "got": got, "calls": calls, "gold_calls": [f"{st.fn}.{st.op}" for st in site.steps]})
    return rows


def summarize(rows: "list[dict]", by: str = "template") -> "dict[str, dict]":
    """Exact match, abstentions and wrong answers per group."""
    groups = defaultdict(lambda: {"n": 0, "exact": 0, "abstained": 0, "wrong": 0})
    for r in rows:
        keys = [r[by]] if by == "template" else sorted(set(c.split(".")[0] for c in r["gold_calls"]))
        for key in keys + ["all"]:
            g = groups[key]
            g["n"] += 1
            g["exact"] += r["got"] == r["gold"]
            g["abstained"] += r["got"] == "<abstain>"
            g["wrong"] += r["got"] not in (r["gold"], "<abstain>")
    return {k: dict(v, exact_rate=round(v["exact"] / v["n"], 4)) for k, v in sorted(groups.items())}


@torch.no_grad()
def false_calls(core, tokenizer, layer: FunctionLayer, data: "list[Aligned]", batch_size: int = 64, device=None) -> dict:
    """How often a trigger fires at a position where no call belongs (first call step only)."""
    fired = positions = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        input_ids, pad_mask = collate(batch, tokenizer.bos_id, tokenizer.pad_id, device)
        feats = layer.features(core_hidden(core, input_ids, pad_mask), pad_mask)
        ctrl = layer.controller.ctrl(0, Kind.NONE, pad_mask.device)
        any_fire = torch.zeros_like(pad_mask)
        for name, module in layer.modules.items():
            call = module.heads.outputs(feats["z"][name], feats["c"], ctrl)[0]
            any_fire |= torch.sigmoid(call) >= module.threshold
        plain = ~pad_mask
        for b, a in enumerate(batch):
            for site in a.sites:
                plain[b, site.position + 1] = False
        fired += int((any_fire & plain).sum())
        positions += int(plain.sum())
    return {"positions": positions, "false_calls": fired, "per_1000": round(1000 * fired / max(positions, 1), 3)}
