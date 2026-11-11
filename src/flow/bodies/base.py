"""What every function body looks like from the outside.

A body reads typed values from the argument registers and writes one typed value to RES.
It depends only on the type system, never on a core, so it is portable across cores.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..typesys.values import Kind, Value


@dataclass(frozen=True)
class Signature:
    args: "tuple[Kind, ...]"
    result: Kind


class BodyError(Exception):
    """The body cannot produce a result (overflow, wrong types). The runtime then abstains."""


class Body:
    """Base class. Subclasses set `function`, `operations`, `signature` and implement `run`."""

    function: str = ""
    operations: "tuple[str, ...]" = ()
    signature: Signature = Signature((), Kind.NONE)

    def check(self, operation: str, args: "list[Value]") -> None:
        if operation not in self.operations:
            raise BodyError(f"{self.function} has no operation {operation!r}")
        kinds = tuple(a.kind for a in args)
        if kinds != self.signature.args:
            raise BodyError(f"{self.function} needs {[k.value for k in self.signature.args]}, got {[k.value for k in kinds]}")

    def __call__(self, operation: str, args: "list[Value]") -> Value:
        self.check(operation, args)
        return self.run(operation, args)

    def run(self, operation: str, args: "list[Value]") -> Value:  # pragma: no cover
        raise NotImplementedError
