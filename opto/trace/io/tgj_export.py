from __future__ import annotations
from typing import Dict, Any, Iterable, Set
from opto.trace.nodes import Node, MessageNode, ParameterNode, ExceptionNode, GRAPH, get_op_name


def _base_name(n: Node) -> str:
    return n.name.split(":")[0]

def export_subgraph_to_tgj(nodes: Iterable[Node], run_id: str, agent_id: str, graph_id: str, scope: str="") -> Dict[str,Any]:
    seen: Set[Node] = set()
    q = list(nodes)
    tgj_nodes = []
    idmap: Dict[Node,str] = {}

    def nid(n: Node) -> str:
        if n not in idmap:
            idmap[n] = _base_name(n)
        return idmap[n]

    while q:
        n = q.pop()
        if n in seen:
            continue
        seen.add(n)

        if isinstance(n, ParameterNode):
            tgj_nodes.append({
                "id": nid(n),
                "kind": "parameter",
                "name": _base_name(n),
                "value": n.data,
                "trainable": True,
                "description": "[Parameter]"
            })
        elif isinstance(n, MessageNode):
            for p in n.parents:
                q.append(p)
            inputs = {f"in_{i}": {"ref": nid(p)} for i, p in enumerate(n.parents)}
            op = n.op_name if hasattr(n, "op_name") else get_op_name(n.description or "[op]")
            rec = {
                "id": nid(n),
                "kind": "message",
                "name": _base_name(n),
                "op": op,
                "description": f"[{op}] {n.description or ''}".strip(),
                "inputs": inputs,
                "output": {"name": f"{_base_name(n)}:out", "value": n.data}
            }
            tgj_nodes.append(rec)
        elif isinstance(n, ExceptionNode):
            for p in n.parents:
                q.append(p)
            tgj_nodes.append({
                "id": nid(n),
                "kind": "exception",
                "name": _base_name(n),
                "description": f"[Exception] {n.description or ''}".strip(),
                "inputs": {f"in_{i}": {"ref": nid(p)} for i, p in enumerate(n.parents)},
                "error": {"type": "Exception", "message": str(n.data)}
            })
        else:
            for p in n.parents:
                q.append(p)
            tgj_nodes.append({
                "id": nid(n),
                "kind": "value",
                "name": _base_name(n),
                "value": n.data,
                "description": "[Node]"
            })
    tgj_nodes.reverse()
    return {
        "tgj": "1.0",
        "run_id": run_id,
        "agent_id": agent_id,
        "graph_id": graph_id,
        "scope": scope,
        "nodes": tgj_nodes,
    }

def export_full_graph_to_tgj(run_id: str, agent_id: str, graph_id: str, scope: str="") -> Dict[str,Any]:
    all_nodes = [n for lst in GRAPH._nodes.values() for n in lst]
    return export_subgraph_to_tgj(all_nodes, run_id, agent_id, graph_id, scope)
