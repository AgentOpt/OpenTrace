"""Helpers for rebuilding Trace nodes from TGJ and OTEL-derived documents."""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager

from opto.trace.nodes import Node, MessageNode, ParameterNode, ExceptionNode, NAME_SCOPES

OTEL_PROFILE_VERSION = "trace-json/1.0+otel"

@contextmanager
def _scoped(scope: str):
    """Temporarily push a Trace name scope while ingesting one document."""
    if scope:
        NAME_SCOPES.append(scope)
    try:
        yield
    finally:
        if scope and NAME_SCOPES:
            NAME_SCOPES.pop()

def _mk_value(name: str, value: Any, desc: str="[Node]") -> Node:
    """Create a plain ``Node`` using a TGJ-safe name."""
    safe = name.replace(":", "_")
    return Node(value, name=safe, description=desc)

def _as_node(ref: Union[str, Dict[str,Any]], local: Dict[str,Node], ports: Dict[str,Node], port_index: Optional[Dict[str,Node]] = None) -> Node:
    """Resolve a TGJ input/output reference into a concrete Trace node."""
    if isinstance(ref, str):
        ref = {"ref": ref}
    if "ref" in ref:
        key = ref["ref"]
        local.setdefault(key, _mk_value(key, None))
        return local[key]
    if "export" in ref:
        pid = ref["export"]
        if port_index and pid in port_index:
            return port_index[pid]
        ports.setdefault(pid, _mk_value(pid, None, "[Node] (import)"))
        return ports[pid]
    if "literal" in ref:
        val = ref["literal"]
        nm = ref.get("name", f"lit_{abs(hash(str(val)))%10_000}")
        n = _mk_value(nm, val)
        local[nm] = n
        return n
    if "hash" in ref:
        nm = ref.get("name", f"hash_{ref['hash'][7:15]}")
        n = _mk_value(nm, ref.get("preview", "<redacted>"), "[Node] (redacted)")
        local[nm] = n
        return n
    raise ValueError(f"Unsupported ref: {ref}")


def _kind_norm(k: str) -> str:
    """Normalize legacy TGJ kind aliases to canonical names."""
    k = (k or "").lower()
    if k in ("param", "parameter"):
        return "parameter"
    if k in ("const", "value"):
        return "value"
    if k in ("msg", "message"):
        return "message"
    if k == "exception":
        return "exception"
    return k


def _nodes_iter(nodes_field: Union[List[Dict[str,Any]], Dict[str,Dict[str,Any]]]) -> List[Dict[str,Any]]:
    """Accept either list- or dict-shaped TGJ node collections."""
    if isinstance(nodes_field, dict):
        out = []
        for nid, rec in nodes_field.items():
            rec = dict(rec)
            rec.setdefault("id", nid)
            out.append(rec)
        return out
    return list(nodes_field or [])


def _convert_otel_profile(doc: Dict[str,Any]) -> Dict[str,Any]:
    """Convert ``trace-json/1.0+otel`` payloads into canonical TGJ v1 records."""
    raw_nodes = _nodes_iter(doc.get("nodes", {}))
    known_ids = {
        rec.get("id") or rec.get("name")
        for rec in raw_nodes
        if (rec.get("id") or rec.get("name")) is not None
    }
    nodes_list = []
    for rec in raw_nodes:
        kind = _kind_norm(rec.get("kind"))
        nid = rec.get("id") or rec.get("name")
        name = rec.get("name", nid)
        if kind == "parameter":
            nodes_list.append({
                "id": nid,
                "kind": "parameter",
                "name": name,
                "value": rec.get("data"),
                "trainable": rec.get("trainable", True),
                "description": rec.get("description", "[Parameter]")
            })
        elif kind == "message":
            inputs = {}
            for k, v in (rec.get("inputs") or {}).items():
                if isinstance(v, str):
                    if v.startswith("lit:"):
                        inputs[k] = {"literal": v.split(":",1)[1]}
                    elif ":" in v:
                        # First prefer exact-match refs against known node ids.
                        # This preserves stable logical ids like "service:message.id"
                        # introduced by the OTEL -> TGJ adapter.
                        if v in known_ids:
                            inputs[k] = {"ref": v}
                        else:
                            # Backward-compatible fallback for older span-id-based refs
                            # and parameter refs that may not be listed yet.
                            _svc, _, rest = v.partition(":")
                            is_span_like = (
                                len(rest) == 16
                                and all(c in "0123456789abcdef" for c in rest.lower())
                            )
                            is_param_like = rest.startswith("param_")
                            inputs[k] = {"ref": v} if (is_span_like or is_param_like) else {"literal": v}
                    else:
                        inputs[k] = {"literal": v}
                else:
                    inputs[k] = v
            msg_rec = {
                "id": nid,
                "kind": "message",
                "name": name,
                "description": f"[{rec.get('op','op')}] {rec.get('description', name)}".strip(),
                "inputs": inputs,
                "output": {"name": f"{name}:out", "value": rec.get("data")}
            }
            # Propagate info dict (contains otel metadata like temporal_ignore)
            if rec.get("info"):
                msg_rec["info"] = rec["info"]
            nodes_list.append(msg_rec)
        elif kind == "value":
            nodes_list.append({
                "id": nid,
                "kind": "value",
                "name": name,
                "value": rec.get("data"),
                "description": rec.get("description", "[Node]")
            })
    agent = (doc.get("agent") or {}).get("id", "agent")
    return {
        "tgj": "1.0",
        "run_id": (doc.get("otel_meta") or {}).get("trace_id"),
        "agent_id": agent,
        "graph_id": doc.get("graph_id", ""),
        "scope": f"{agent}/0",
        "nodes": nodes_list,
    }

def ingest_tgj(
    doc: Dict[str,Any],
    port_index: Optional[Dict[str,Node]] = None,
    *,
    param_cache: Optional[Dict[str,"ParameterNode"]] = None,
) -> Dict[str,Node]:
    """Rebuild Trace nodes from a TGJ document and return them by id/name."""
    version = doc.get("tgj") or doc.get("version")
    if version == OTEL_PROFILE_VERSION:
        doc = _convert_otel_profile(doc)
        version = doc.get("tgj")
    assert version == "1.0", "Unsupported TGJ version"
    nodes: Dict[str,Node] = {}
    exports: Dict[str,Node] = {}
    ports: Dict[str,Node] = {}

    with _scoped(doc.get("scope", "")):
        # pass 1: parameters/values
        for rec in _nodes_iter(doc.get("nodes", [])):
            k = rec["kind"]
            nid = rec["id"]
            nm = rec.get("name", nid)
            if k == "parameter":
                n = param_cache.get(nid) if param_cache is not None else None
                if n is None:
                    n = ParameterNode(
                        rec.get("value"),
                        name=nm,
                        trainable=bool(rec.get("trainable", True)),
                        description=rec.get("description", "[Parameter]"),
                    )
                    if param_cache is not None:
                        param_cache[nid] = n
                else:
                    try:
                        n._data = rec.get("value")
                    except Exception:
                        pass
                    try:
                        n.trainable = bool(rec.get("trainable", True))
                    except Exception:
                        pass
                nodes[nid] = n
                nodes[nm] = n
            elif k == "value":
                n = _mk_value(nm, rec.get("value"), rec.get("description", "[Node]"))
                nodes[nid] = n
                nodes[nm] = n

        # pass 2: messages/exceptions
        for rec in _nodes_iter(doc.get("nodes", [])):
            k = rec["kind"]
            nid = rec["id"]
            nm = rec.get("name", nid)
            if k in ("message", "exception"):
                in_spec = rec.get("inputs", {}) or {}
                inputs = {key: _as_node(v, nodes, ports, port_index) for key, v in in_spec.items()}
                out_meta = rec.get("output", {}) or {}
                out_name = out_meta.get("name", f"{nm}:out")
                out_node = _as_node(out_meta, nodes, ports, port_index) if ("hash" in out_meta) else _mk_value(out_name, out_meta.get("value"))
                info = {"meta": rec.get("meta", {})}
                iinfo = rec.get("info", {}) or {}
                if "inputs" in iinfo:
                    args = [_as_node(x, nodes, ports, port_index) for x in iinfo["inputs"].get("args", [])]
                    kwargs = {k: _as_node(v, nodes, ports, port_index) for k, v in iinfo["inputs"].get("kwargs", {}).items()}
                    info["inputs"] = {"args": args, "kwargs": kwargs}
                if "output" in iinfo:
                    info["output"] = _as_node(iinfo["output"], nodes, ports, port_index)
                # Preserve OTEL metadata (e.g. temporal_ignore) for
                # downstream consumers like _select_output_node.
                if "otel" in iinfo:
                    info["otel"] = iinfo["otel"]

                desc = rec.get("description", "[Node]")
                if k == "exception":
                    err = rec.get("error", {}) or {}
                    msg = err.get("message", "Exception")
                    n = ExceptionNode(value=Exception(msg), inputs=inputs, description=desc, name=nm, info=info)
                else:
                    n = MessageNode(out_node, inputs=inputs, description=desc, name=nm, info=info)
                nodes[nid] = n
                nodes[nm] = n
                nodes[out_name] = out_node

        # exports
        for port_id, ref in (doc.get("exports") or {}).items():
            exports[port_id] = _as_node(ref, nodes, ports, port_index)
        # resolve ports bound within same doc
        for pid in list(ports.keys()):
            if pid in exports:
                ports[pid] = exports[pid]

    nodes["__TGJ_EXPORTS__"] = exports
    nodes["__TGJ_META__"] = {
        "run_id": doc.get("run_id"),
        "agent_id": doc.get("agent_id"),
        "graph_id": doc.get("graph_id"),
        "scope": doc.get("scope"),
    }
    nodes["__TGJ_PORTS__"] = ports
    return nodes

def merge_tgj(docs: List[Dict[str,Any]]) -> Dict[str,Dict[str,Node]]:
    """Ingest multiple TGJ documents while resolving cross-document exports."""
    merged: Dict[str,Dict[str,Node]] = {}
    port_index: Dict[str,Node] = {}
    for d in docs:
        key = f"{d.get('agent_id','')}/{d.get('graph_id','')}/{d.get('run_id','')}"
        merged[key] = ingest_tgj(d, port_index=port_index)
        for pid, n in (merged[key].get("__TGJ_EXPORTS__") or {}).items():
            port_index[pid] = n
    return merged


class TLSFIngestor:
    """Minimal TLSF ingestor supporting TGJ/trace-json documents."""

    def __init__(self, run_id: Optional[str] = None):
        """Initialize the ingestor and its accumulated node index."""
        self.run_id = run_id
        self._nodes: Dict[str, Node] = {}

    def ingest_tgj(self, doc: Dict[str, Any]) -> None:
        """Ingest a TGJ v1 or trace-json/1.0+otel document."""
        self._nodes.update(ingest_tgj(doc))

    def get(self, name_or_event_id: str) -> Optional[Node]:
        """Look up a previously ingested node by name or event id."""
        return self._nodes.get(name_or_event_id)
