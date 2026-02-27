"""aiai-safety: circuit breakers, loop detection, invariants, protected files."""

from aiai_safety.circuit_breaker import CircuitBreaker, CircuitBreakerTrippedError
from aiai_safety.invariants import Invariant, InvariantChecker, InvariantViolation
from aiai_safety.loop_detector import LoopDetection, LoopDetector
from aiai_safety.protected_files import ProtectedFileChecker

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerTrippedError",
    "Invariant",
    "InvariantChecker",
    "InvariantViolation",
    "LoopDetection",
    "LoopDetector",
    "ProtectedFileChecker",
]
