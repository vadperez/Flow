"""Register bank: the typed session state that function modules read and write.

    ARG0..ARG2  arguments of the current call        written by the binder    not persistent
    RES         result of the latest call            written by the body      until printed or replaced
    STATE       16 key -> value slots                written by state bodies  persistent for the session
    CTRL        call counter, flags, last result     written by the runtime   persistent

Session state lives here, not in any module, so a module can be unloaded and reloaded
without losing state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .values import NONE, Kind, Value

N_ARGS = 3
STATE_SLOTS = 16


@dataclass
class Registers:
    arg: "list[Value]" = field(default_factory=lambda: [NONE] * N_ARGS)
    res: Value = NONE
    res_printed: bool = True
    state: "dict[str, Value]" = field(default_factory=dict)  # insertion-ordered: oldest first
    call_count: int = 0  # calls made at the current position
    miss: bool = False
    overflow: bool = False
    last_fn: "str | None" = None

    # --- per-position bookkeeping -------------------------------------------------------
    def new_position(self) -> None:
        self.arg = [NONE] * N_ARGS
        self.call_count = 0

    def set_result(self, value: Value, fn: str) -> None:
        self.res = value
        self.res_printed = value.kind is Kind.NONE
        self.last_fn = fn
        self.call_count += 1

    # --- STATE: slot memory with overwrite and oldest-first eviction -----------------------
    def bind(self, key: str, value: Value) -> None:
        if key in self.state:
            del self.state[key]
        elif len(self.state) >= STATE_SLOTS:
            del self.state[next(iter(self.state))]
        self.state[key] = value

    def query(self, key: str) -> Value:
        self.miss = key not in self.state
        return self.state.get(key, NONE)

    def clear_state(self) -> None:
        self.state.clear()
