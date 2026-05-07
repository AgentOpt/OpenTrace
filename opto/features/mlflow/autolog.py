"""
opto.features.mlflow.autolog
===========================

Best-effort MLflow autologging integration.

Design goals
------------
- Keep MLflow as an *optional* dependency.
- Defaults should be "off" so existing code paths are unchanged.
- When enabled, ``@trace.bundle`` operations may be wrapped by ``mlflow.trace``
  (see ``opto.trace.bundle.bundle``), and LiteLLM calls can be autologged
  when supported by the installed MLflow version.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from opto.trace import settings

logger = logging.getLogger(__name__)

mlflow_autologging = False
mlflow_config = {}

def autolog(
    *,
    log_models: bool = True,
    disable_default_op_logging: bool = True,
    extra_tags: Optional[Dict[str, Any]] = None,
    silent: bool = False,
) -> None:
    """Enable MLflow autologging for Trace.

    Parameters
    ----------
    log_models
        If True, enable tracing spans (via ``mlflow.trace`` wrapping).
    disable_default_op_logging
        If True, suppress spans for low-level "default ops" (heuristically detected).
    extra_tags
        Optional tag dict to be attached by downstream MLflow tooling.
    silent
        If True, suppress warnings when MLflow isn't installed.
    """
    global mlflow_autologging, mlflow_config
    mlflow_autologging = True
    mlflow_config = {
        "log_models": log_models,
        "disable_default_op_logging": disable_default_op_logging,
        "extra_tags": extra_tags or {},
    }

    try:
        import mlflow  # type: ignore
    except Exception:
        settings.mlflow_autologging = False
        if not silent:
            logger.warning("MLflow is not installed; MLflow autologging disabled.")
        return

    try:
        if hasattr(mlflow, "litellm") and hasattr(mlflow.litellm, "autolog"):
            mlflow.litellm.autolog()
    except Exception:
        pass


def disable_autolog() -> None:
    """Disable MLflow autologging."""
    settings.mlflow_autologging = False
    settings.mlflow_config = {}


def is_autolog_enabled() -> bool:
    return bool(settings.mlflow_autologging)


def get_autolog_config() -> Dict[str, Any]:
    return dict(settings.mlflow_config or {})
