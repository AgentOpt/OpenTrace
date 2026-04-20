from __future__ import annotations

"""
Utilities to export an already-built Trace graph (Node / MessageNode / ParameterNode)
to TGJ format.

Intended use:
- debugging and inspecting native Trace graphs
- tests comparing native Trace graphs with OTEL-recovered TGJ graphs
- exporting a subgraph from the in-memory Trace graph

Non-goals:
- this does NOT reconstruct a graph from telemetry; it only exports an existing Trace graph
"""

from typing import Dict, Any, Iterable, Set
from opto.trace.nodes import (
    Node,
    MessageNode,
    ParameterNode,
    ExceptionNode,
    GRAPH,
    get_op_name,
)


def _base_name(n: Node) -> str:
    return n.name.split(":")[0]


def export_subgraph_to_tgj(
    nodes: Iterable[Node],
    run_id: str,
    agent_id: str,
    graph_id: str,
    scope: str = "",
) -> Dict[str, Any]:
    seen: Set[Node] = set()
    q = list(nodes)
    tgj_nodes = []
    idmap: Dict[Node, str] = {}
    used_ids: Set[str] = set()

    def nid(n: Node) -> str:
        if n not in idmap:
            base = _base_name(n)
            candidate = base
            i = 2
            while candidate in used_ids:
                candidate = f"{base}__{i}"
                i += 1
            idmap[n] = candidate
            used_ids.add(candidate)
        return idmap[n]

    while q:
        n = q.pop()
        if n in seen:
            continue
        seen.add(n)

        if isinstance(n, ParameterNode):
            tgj_nodes.append(
                {
                    "id": nid(n),
                    "kind": "parameter",
                    "name": _base_name(n),
                    "value": n.data,
                    "trainable": bool(getattr(n, "trainable", True)),
                    "description": "[Parameter]",
                }
            )

        elif isinstance(n, MessageNode):
            for p in n.parents:
                q.append(p)

            inputs = {f"in_{i}": {"ref": nid(p)} for i, p in enumerate(n.parents)}
            for i, dep in enumerate(getattr(n, "hidden_dependencies", ()) or ()):
                q.append(dep)
                inputs[f"hidden_{i}"] = {"ref": nid(dep)}

            op = getattr(n, "op_name", None)
            if not op:
                try:
                    op = get_op_name(n.description or "[op]")
                except Exception:
                    op = "op"

            rec = {
                "id": nid(n),
                "kind": "message",
                "name": _base_name(n),
                "op": op,
                "description": f"[{op}] {n.description or ''}".strip(),
                "inputs": inputs,
                "output": {
                    "name": f"{_base_name(n)}:out",
                    "value": n.data,
                },
            }
            tgj_nodes.append(rec)

        elif isinstance(n, ExceptionNode):
            for p in n.parents:
                q.append(p)

            err_type = "Exception"
            try:
                if n.data is not None:
                    err_type = type(n.data).__name__
            except Exception:
                pass

            tgj_nodes.append(
                {
                    "id": nid(n),
                    "kind": "exception",
                    "name": _base_name(n),
                    "description": f"[Exception] {n.description or ''}".strip(),
                    "inputs": {f"in_{i}": {"ref": nid(p)} for i, p in enumerate(n.parents)},
                    "error": {
                        "type": err_type,
                        "message": str(n.data),
                    },
                }
            )

        else:
            for p in n.parents:
                q.append(p)

            tgj_nodes.append(
                {
                    "id": nid(n),
                    "kind": "value",
                    "name": _base_name(n),
                    "value": n.data,
                    "description": "[Node]",
                }
            )

    # best-effort dependency order
    tgj_nodes.reverse()

    return {
        "tgj": "1.0",
        "run_id": run_id,
        "agent_id": agent_id,
        "graph_id": graph_id,
        "scope": scope,
        "nodes": tgj_nodes,
    }


def export_full_graph_to_tgj(
    run_id: str,
    agent_id: str,
    graph_id: str,
    scope: str = "",
) -> Dict[str, Any]:
    all_nodes = [n for lst in GRAPH._nodes.values() for n in lst]
    return export_subgraph_to_tgj(all_nodes, run_id, agent_id, graph_id, scope)
