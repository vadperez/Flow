"""Function bodies. Exact functions are written weights: constructed, tested exhaustively, never trained."""
from .add_exact import AddExact
from .base import Body, BodyError, Signature
from .compare import CompareNumeric

BODIES = {"add_exact": AddExact, "compare_numeric": CompareNumeric}

__all__ = ["AddExact", "Body", "BodyError", "CompareNumeric", "Signature", "BODIES"]
