"""
opto.trace.io.bindings
======================

Minimal get/set binding layer that maps OTEL/TGJ parameter keys
(e.g. ``param.planner_prompt``, ``param.__code_planner``) to concrete
getter/setter callables.  This decouples the optimizer's string-keyed
updates from the runtime location of the actual variable, function, or
graph knob.

Usage
-----
>>> b = Binding(get=lambda: my_template, set=lambda v: setattr(cfg, "template", v))
>>> apply_updates({"planner_prompt": "new prompt"}, {"planner_prompt": b})
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class Binding:
    """Minimal get/set binding for a single trainable target.

    Attributes
    ----------
    get : Callable[[], Any]
        Returns the current value (used for logging / optimizer init).
    set : Callable[[Any], None]
        Applies an updated value in-memory (prompts / code / graph knobs).
    kind : ``"prompt"`` | ``"code"`` | ``"graph"``
        Describes the binding type for validation and reporting.
    """

    get: Callable[[], Any]
    set: Callable[[Any], None]
    kind: Literal["prompt", "code", "graph"] = "prompt"


def apply_updates(
    updates: Dict[Any, Any],
    bindings: Dict[str, Binding],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Apply optimizer updates to bound targets.

    Parameters
    ----------
    updates : Dict[Any, Any]
        Keys are parameter names (strings) **or** ParameterNode objects.
        Values are the new values suggested by the optimizer.
    bindings : Dict[str, Binding]
        Mapping from parameter names to ``Binding`` objects.
    strict : bool
        If *True* (default), raise ``KeyError`` when an update key has
        no corresponding binding.  If *False*, unknown keys are silently
        skipped.

    Returns
    -------
    Dict[str, Any]
        The updates that were actually applied (string-keyed).

    Raises
    ------
    KeyError
        When *strict* is True and an update key is missing from *bindings*.
    """

    def _normalize_key(k: Any) -> str:
        """Coerce update keys into binding names used by the runtime."""
        if isinstance(k, str):
            s = k
        else:
            s = (
                getattr(k, "name", None)
                or getattr(k, "_name", None)
                or getattr(k, "py_name", None)
                or str(k)
            )
        s = str(s).strip()
        if s.startswith("param."):
            s = s[len("param."):]
        s = s.split(":")[0].split("/")[-1]
        if s not in bindings:
            s2 = re.sub(r"\d+$", "", s)
            if s2 in bindings:
                s = s2
        return s

    applied: Dict[str, Any] = {}
    for raw_key, value in updates.items():
        key = _normalize_key(raw_key)
        binding = bindings.get(key)
        if binding is None:
            if strict:
                raise KeyError(
                    f"apply_updates: no binding for key {key!r} (from {raw_key!r}). "
                    f"Available bindings: {sorted(bindings.keys())}"
                )
            logger.debug("apply_updates: skipping unknown key %r (from %r)", key, raw_key)
            continue
        try:
            binding.set(value)
            applied[key] = value
            logger.debug("apply_updates: set %r (kind=%s)", key, binding.kind)
        except Exception:
            logger.exception("apply_updates: failed to set %r", key)
            if strict:
                raise
    return applied


def make_dict_binding(store: Dict[str, Any], key: str, kind: str = "prompt") -> Binding:
    """Convenience helper: create a ``Binding`` backed by a plain dict entry.

    Parameters
    ----------
    store : dict
        The dictionary that holds the value.
    key : str
        The key within *store*.
    kind : str
        Binding kind (``"prompt"``, ``"code"``, ``"graph"``).
    """
    return Binding(
        get=lambda: store.get(key),
        set=lambda v: store.__setitem__(key, v),
        kind=kind,
    )
