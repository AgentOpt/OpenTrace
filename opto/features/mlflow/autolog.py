"""MLflow autologging functionality for Trace library.

This module provides MLflow autologging functionality similar to mlflow.dspy.autolog().
Users can call trace.mlflow.autolog() at the beginning of their scripts to enable
automatic logging of Trace operations to MLflow.
"""

import logging
from typing import Optional, Dict, Any

from opto.trace import settings

logger = logging.getLogger(__name__)


def autolog(
    log_models: bool = True,
    log_datasets: bool = True,
    disable: bool = False,
    disable_default_op_logging: bool = True,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
    registered_model_name: Optional[str] = None,
    extra_tags: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Enable automatic logging of Trace operations to MLflow.
    
    This function enables MLflow autologging for Trace library operations, similar to
    how mlflow.dspy.autolog() works for DSPy. When enabled, Trace operations will
    automatically log relevant information to MLflow.
    
    Args:
        log_models (bool): If True, trained models are logged as MLflow model artifacts.
            Defaults to True.
        log_datasets (bool): If True, dataset information used in training is logged to MLflow
            Tracking. Defaults to True.
        disable (bool): If True, disables MLflow autologging. Defaults to False.
        disable_default_op_logging (bool): If True, disables logging of default Trace operations in trace.operators
        exclusive (bool): If True, autologged content is not logged to additional MLflow
            Tracking fluent APIs. Defaults to False.
        disable_for_unsupported_versions (bool): If True, disable MLflow autologging for
            versions of Trace that have not been tested against this version of the MLflow
            client or are not supported. Defaults to False.
        silent (bool): If True, suppress all event logs and warnings from MLflow during
            Trace training. Defaults to False.
        registered_model_name (str): If given, each time a model is trained, it is registered
            as a new model version of the registered model with this name. The registered model
            is created if it does not already exist. Defaults to None.
        extra_tags (Dict[str, Any]): A dictionary of extra tags to set on each managed MLflow Run.
            Defaults to None.
    
    Example:
        >>> import opto.trace as trace
        >>> trace.mlflow.autolog()  # Enable MLflow autologging
        >>> # Your Trace code here - operations will be automatically logged to MLflow
    """
    if disable:
        settings.mlflow_autologging = False
        if not silent:
            logger.info("MLflow autologging for Trace has been disabled.")
        return
    
    # Enable MLflow autologging
    settings.mlflow_autologging = True

    # Enable litellm logging (Trace uses litellm backend)
    import mlflow
    mlflow.litellm.autolog()

    # Store configuration in settings
    settings.mlflow_config = {
        "log_models": log_models,
        "log_datasets": log_datasets,
        "disable_default_op_logging": disable_default_op_logging,
        "exclusive": exclusive,
        "disable_for_unsupported_versions": disable_for_unsupported_versions,
        "silent": silent,
        "registered_model_name": registered_model_name,
        "extra_tags": extra_tags or {},
    }
    
    if not silent:
        logger.info("MLflow autologging for Trace has been enabled.")


def is_autolog_enabled() -> bool:
    """
    Check if MLflow autologging is currently enabled.
    
    Returns:
        bool: True if MLflow autologging is enabled, False otherwise.
    """
    return getattr(settings, 'mlflow_autologging', False)


def get_autolog_config() -> Dict[str, Any]:
    """
    Get the current MLflow autolog configuration.
    
    Returns:
        Dict[str, Any]: The current autolog configuration, or empty dict if not configured.
    """
    return getattr(settings, 'mlflow_config', {})


def disable_autolog() -> None:
    """
    Disable MLflow autologging for Trace operations.
    
    This is equivalent to calling autolog(disable=True).
    """
    autolog(disable=True, silent=True)
