"""
opto.trace.settings
===================

Lightweight global settings for optional integrations.

This module is intentionally minimal; defaults keep integrations **disabled**
so importing opto.trace does not introduce extra runtime dependencies.

Currently supported:
- MLflow autologging toggle and config (used by opto.features.mlflow.autolog)
"""

mlflow_autologging = False

mlflow_config = {}
