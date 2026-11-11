"""The call loop (pipeline section 2.6) and a small generation routine around it.

At every new position:
    up to max_calls times: the resident module with the highest "call" score above its
    threshold is called, with the operation its trigger ranks first; its binder picks the
    arguments with hard pointers (earlier typed values or RES); its body computes the result
    into RES.
    Then: if RES holds an unprinted value and the last call said "final", the emitter prints it,
    token by token, in place of the core's own choice. Otherwise the core's token is used; at a
    place where a result is needed that token is <abstain>. A chain that cannot be completed,
    because the module that should continue is not loaded, is therefore dropped, not printed.

Modules can be attached and detached between any two tokens: session state lives in the
registers, never in a module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..bodies.base import Body, BodyError
from ..typesys import NONE, Anchor, Emitter, PieceMap, Value, scan
from .model import ControllerBlock, FunctionHeads


@dataclass
class CallTrace:
    position: int
    step: int
    fn: str
    op: str
    score: float
    pointers: "list"  # token index of each argument, or "RES"
    args: "list[str]"
    result: "str | None"
    error: "str | None" = None
    final: bool = True  # the module said this call completes the chain


@dataclass
class Resident:
    """A module that is loaded: learned heads plus body."""

    heads: FunctionHeads
    body: Body
    threshold: float = 0.5
    calls: int = 0


class FunctionLayer:
    def __init__(self, controller: ControllerBlock) -> None:
        self.controller = controller
        self.modules: "dict[str, Resident]" = {}

    def attach(self, name: str, heads: FunctionHeads, body: Body, threshold: float = 0.5) -> None:
        self.modules[name] = Resident(heads.eval(), body, threshold)

    def detach(self, name: str) -> None:
        self.modules.pop(name, None)

    @torch.no_grad()
    def features(self, hidden_states, pad_mask=None) -> dict:
        """Shared features and the private features of every resident module, for a batch of sequences."""
        return {"c": self.controller.features(hidden_states, pad_mask),
                "z": {name: m.heads.features(hidden_states, pad_mask) for name, m in self.modules.items()}}

    @torch.no_grad()
    def calls_at(self, feats: dict, row: int, position: int, anchors: "list[Anchor]", start: Value = NONE):
        """Run the calls for one position.

        feats: result of features(); row: which sequence of the batch; position: index into it;
        anchors: typed values of the text, with token indices in the same indexing as the features.
        Returns (value to print or None, traces).
        """
        c_seq = feats["c"][row]
        res, traces, printed, complete = start, [], True, True
        for step in range(self.controller.max_calls):
            ctrl = self.controller.ctrl(step, res.kind, c_seq.device)
            best = None
            for name, module in self.modules.items():
                call, ops, final = module.heads.outputs(feats["z"][name][row, position], c_seq[position], ctrl)
                score = float(torch.sigmoid(call))
                if score >= module.threshold and (best is None or score > best[2]):
                    best = (name, int(ops.argmax()), score, bool(final > 0))
            if best is None:
                break
            name, k, score, complete = best
            module = self.modules[name]
            op = module.body.operations[k]
            queries = module.heads.queries(feats["z"][name][row, position], c_seq[position], ctrl)
            args, pointers, problem = [], [], None
            for j, kind in enumerate(module.body.signature.args):
                cands = [a for a in anchors if a.token <= position and a.value.kind is kind]
                with_res = res.kind is kind
                if not cands and not with_res:
                    problem = f"no {kind.value} value to point at"
                    break
                tokens = torch.tensor([a.token for a in cands], dtype=torch.long, device=c_seq.device)
                keys = self.controller.candidate_keys(c_seq, tokens, with_res)
                pick = int(FunctionHeads.scores(queries[j], keys).argmax())
                if pick == len(cands):
                    args.append(res)
                    pointers.append("RES")
                else:
                    args.append(cands[pick].value)
                    pointers.append(cands[pick].token)
            if problem is None:
                try:
                    res = module.body(op, args)
                    printed = False
                    module.calls += 1
                except BodyError as exc:
                    problem = str(exc)
            traces.append(CallTrace(position, step + 1, name, op, round(score, 4), pointers, [a.text() for a in args],
                                    None if problem else res.text(), problem, complete))
            if problem is not None:
                return None, traces  # nothing is printed: the core's own token (normally <abstain>) is used
        return (None if printed or not complete else res), traces


@dataclass
class Generation:
    ids: "list[int]"
    text: str
    traces: "list[CallTrace]" = field(default_factory=list)
    printed_by_emitter: "list[int]" = field(default_factory=list)  # positions (in ids) written by the emitter


@torch.no_grad()
def generate(core, tokenizer, layer: "FunctionLayer | None", prompt: str, piece_map: PieceMap, max_new: int = 32,
             stop_ids: "tuple[int, ...]" = (), table=None, device=None) -> Generation:
    """Greedy generation with the call loop. Reference version: the core re-reads the whole text at
    every step (no cache); a cached runtime is pipeline stage S8."""
    ids = list(tokenizer.encode(prompt).ids)
    emitter, traces, by_emitter = Emitter(), [], []
    for _ in range(max_new):
        if emitter.active:
            by_emitter.append(len(ids))
            ids.append(piece_map.token_id(emitter.next_piece()))
            continue
        inputs = torch.tensor([[tokenizer.bos_id] + ids], device=device)
        out = core.model(input_ids=inputs, output_hidden_states=True, use_cache=False)
        if layer is not None and layer.modules:
            feats = layer.features(out.hidden_states)
            text, offsets = tokenizer.render(ids)
            pieces = [tokenizer.vocab[i] for i in ids]
            anchors = [Anchor(a.token + 1, a.start, a.end, a.value) for a in scan(text, offsets, pieces, table)]
            value, new = layer.calls_at(feats, 0, len(ids), anchors)
            traces += new
            if value is not None:
                emitter.start(value)
                continue
        nxt = int(core.lm_head(out.last_hidden_state[0, -1]).argmax())
        ids.append(nxt)
        if nxt in stop_ids:
            break
    return Generation(ids, tokenizer.render(ids)[0], traces, by_emitter)
