"""
opto.features.mlflow
===================

Optional MLflow integration for Trace.

Importing this package should be safe even when MLflow is not installed.
Use ``opto.features.mlflow.autolog`` to enable tracing/metrics capture.
"""

from .autolog import autolog, disable_autolog, get_autolog_config, is_autolog_enabled

__all__ = [
    "autolog",
    "disable_autolog",
    "get_autolog_config",
    "is_autolog_enabled",
]
