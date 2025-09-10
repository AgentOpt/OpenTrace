from __future__ import annotations
from typing import Dict, Any, List


PROFILE_VERSION = "trace-json/1.0+otel"


def _sanitize(name: str) -> str:
    return (name or "node").replace(":", "_")


def _op(attrs, span):
    if "gen_ai.operation" in attrs or "gen_ai.model" in attrs:
        return "llm_call"
    if "rpc.system" in attrs:
        return f"rpc:{attrs['rpc.system']}"
    if "http.method" in attrs:
        return f"http:{attrs['http.method']}".lower()
    if "db.system" in attrs:
        return f"db:{attrs['db.system']}"
    return (span.get("kind", "op") or "op").lower()


def _attrs(l):
    out = {}
    for a in l or []:
        k = a["key"]
        v = a.get("value", {})
        if isinstance(v, dict) and v:
            out[k] = next(iter(v.values()))
    return out


def _lift_inputs(attrs: Dict[str, Any]) -> Dict[str, str]:
    inputs = {}
    for k, v in list(attrs.items()):
        if k.startswith("inputs.") and isinstance(v, str):
            role = k.split(".", 1)[1]
            if v.startswith("span:"):
                inputs[role] = v.split(":", 1)[1]
            else:
                inputs[role] = v
    for k in ("gen_ai.prompt", "gen_ai.system", "gen_ai.temperature", "db.statement", "http.url"):
        if k in attrs and f"inputs.{k}" not in attrs:
            inputs[k] = f"lit:{k}"
    return inputs


def _params(attrs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, v in attrs.items():
        if k.startswith("param.") and not k.endswith(".trainable"):
            name = k.split(".", 1)[1]
            out[name] = {
                "value": v,
                "trainable": bool(attrs.get(f"param.{name}.trainable", False)),
            }
    return out


def otlp_traces_to_trace_json(otlp: Dict[str, Any], agent_id_hint: str = "") -> List[Dict[str, Any]]:
    docs = []
    for rs in otlp.get("resourceSpans", []):
        rattrs = _attrs(rs.get("resource", {}).get("attributes", []))
        svc = rattrs.get("service.name", agent_id_hint or "service")
        inst = rattrs.get("service.instance.id", "0")
        for ss in rs.get("scopeSpans", []):
            scope_nm = ss.get("scope", {}).get("name", "scope")
            nodes = {}
            trace_id = None
            for sp in ss.get("spans", []):
                trace_id = sp.get("traceId") or trace_id
                sid = sp.get("spanId")
                psid = sp.get("parentSpanId")
                attrs = _attrs(sp.get("attributes", []))
                op = _op(attrs, sp)
                name = _sanitize(sp.get("name") or sid)
                params = _params(attrs)
                for pname, spec in params.items():
                    p_id = f"{svc}:param_{pname}"
                    nodes.setdefault(
                        p_id,
                        {
                            "kind": "param",
                            "name": pname,
                            "data": spec["value"],
                            "trainable": bool(spec["trainable"]),
                            "info": {"otel": {"span_id": sid}},
                        },
                    )
                inputs = _lift_inputs(attrs)
                if psid and "parent" not in inputs:
                    inputs["parent"] = f"{svc}:{psid}"
                rec = {
                    "kind": "msg",
                    "name": name,
                    "op": op,
                    "inputs": {},
                    "data": {"message_id": attrs.get("message.id")},
                    "info": {
                        "otel": {
                            "trace_id": trace_id,
                            "span_id": sid,
                            "parent_span_id": psid,
                            "service": svc,
                        }
                    },
                }
                for role, ref in inputs.items():
                    if ref.startswith("lit:"):
                        rec["inputs"][role] = ref
                    else:
                        rec["inputs"][role] = ref if ":" in ref else f"{svc}:{ref}"
                node_id = f"{svc}:{sid}"
                nodes[node_id] = rec
            docs.append(
                {
                    "version": PROFILE_VERSION,
                    "agent": {"id": svc, "service": svc},
                    "otel_meta": {"trace_id": trace_id},
                    "nodes": nodes,
                    "context": {},
                }
            )
    return docs

