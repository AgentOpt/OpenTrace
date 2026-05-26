"""GEPA optimize_anything-compatible API for Trace."""

from opto.optimize_anything.api import (
    EngineConfig,
    EvaluationRecord,
    GEPAConfig,
    GEPAResult,
    MergeConfig,
    OptimizationState,
    ReflectionConfig,
    RefinerConfig,
    TrackingConfig,
    get_log_context,
    log,
    make_litellm_lm,
    optimize_anything,
    reset_log_context,
    set_log_context,
)
from opto.optimize_anything.trace_backend import TraceOptimizerBackend, resolve_optimizer_cls

__all__ = [
    "EngineConfig",
    "EvaluationRecord",
    "GEPAConfig",
    "GEPAResult",
    "MergeConfig",
    "OptimizationState",
    "ReflectionConfig",
    "RefinerConfig",
    "TrackingConfig",
    "TraceOptimizerBackend",
    "get_log_context",
    "log",
    "make_litellm_lm",
    "optimize_anything",
    "reset_log_context",
    "resolve_optimizer_cls",
    "set_log_context",
]
