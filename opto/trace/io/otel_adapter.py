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
                "trainable": str(raw).strip().lower() in ("1", "true", "yes", "y", "on") if isinstance((raw := attrs.get(f"param.{name}.trainable", False)), str) else bool(raw),
            }
    return out


def otlp_traces_to_trace_json(otlp: Dict[str, Any], agent_id_hint: str = "", use_temporal_hierarchy: bool = False) -> List[Dict[str, Any]]:
    """Convert OTLP traces to Trace-Graph JSON format.
    
    Args:
        otlp: OTLP JSON payload
        agent_id_hint: Optional service name hint
        use_temporal_hierarchy: If True, create parent-child relationships based on temporal ordering
                               (earlier spans become parents of later spans) when no explicit parent exists.
                               This enables backward propagation across sequential agent calls.
    
    Returns:
        List of TGJ documents
    """
    docs = []
    for rs in otlp.get("resourceSpans", []):
        rattrs = _attrs(rs.get("resource", {}).get("attributes", []))
        svc = rattrs.get("service.name", agent_id_hint or "service")
        inst = rattrs.get("service.instance.id", "0")
        for ss in rs.get("scopeSpans", []):
            scope_nm = ss.get("scope", {}).get("name", "scope")
            nodes = {}
            trace_id = None
            
            # First pass: collect all spans with their timestamps for temporal ordering
            spans_with_time = []
            for sp in ss.get("spans", []):
                spans_with_time.append((sp.get("startTimeUnixNano", 0), sp))
            
            # Sort by start time to establish temporal order
            spans_with_time.sort(key=lambda x: x[0])
            
            # Track the most recent span for temporal parenting
            prev_span_id = None
            # Map span_id -> actual TGJ node_id (for stable parent references)
            span_to_node_id: Dict[str, str] = {}
            
            # Identify root invocation spans (e.g. "service.invoke") so we
            # can exclude them from temporal chaining — they are structural
            # parents, not data-flow nodes.
            root_span_ids: set = set()
            for _, sp in spans_with_time:
                sp_name = sp.get("name", "")
                if sp_name.endswith(".invoke"):
                    root_span_ids.add(sp.get("spanId"))

            for start_time, sp in spans_with_time:
                trace_id = sp.get("traceId") or trace_id
                sid = sp.get("spanId")
                psid = sp.get("parentSpanId")
                attrs = _attrs(sp.get("attributes", []))

                # D10: Use trace.temporal_ignore to decide temporal chain
                temporal_ignore = str(
                    attrs.get("trace.temporal_ignore", "false")
                ).strip().lower() in ("true", "1", "yes")

                # Skip root invocation spans — they are structural wrappers,
                # not data-flow nodes.
                if sid in root_span_ids:
                    continue

                op = _op(attrs, sp)
                name = _sanitize(sp.get("name") or sid)
                params = _params(attrs)
                
                for pname, spec in params.items():
                    p_id = f"{svc}:param_{pname}"
                    nodes.setdefault(
                        p_id,
                        {
                            "kind": "parameter",
                            "name": pname,
                            "data": spec["value"],
                            "trainable": bool(spec["trainable"]),
                            "info": {"otel": {"span_id": sid}},
                        },
                    )
                inputs = _lift_inputs(attrs)
                
                # Temporal hierarchy: connect to previous non-ignored span
                # when use_temporal_hierarchy is enabled.
                # With root invocation spans (D9), node spans have a
                # structural parent.  We still want temporal chaining
                # among sibling node spans, so we use prev_span_id
                # regardless of whether psid is set — the key gate is
                # temporal_ignore.
                effective_psid = psid
                if use_temporal_hierarchy and prev_span_id and not temporal_ignore:
                    # If the OTEL parent is the root invocation span,
                    # prefer temporal parent for data-flow graph.
                    if not psid or psid in root_span_ids:
                        effective_psid = prev_span_id

                # If our effective parent is a skipped root invocation span,
                # do not emit a parent edge that would dangle in TGJ.
                if effective_psid and effective_psid in root_span_ids:
                    effective_psid = None

                if effective_psid and "parent" not in inputs:
                    # Resolve via mapping so parent refs use stable node ids
                    inputs["parent"] = span_to_node_id.get(effective_psid, f"{svc}:{effective_psid}")
                
                # Connect parameters as inputs to the MessageNode
                for pname in params.keys():
                    inputs[f"param_{pname}"] = f"{svc}:param_{pname}"
                
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
                            "parent_span_id": effective_psid,
                            "service": svc,
                            "temporal_ignore": temporal_ignore,
                        }
                    },
                }
                for role, ref in inputs.items():
                    if ref.startswith("lit:"):
                        rec["inputs"][role] = ref
                    else:
                        rec["inputs"][role] = ref if ":" in ref else f"{svc}:{ref}"
                # Use message.id as stable logical node identity when
                # available; fall back to span id for backward compat.
                msg_id = attrs.get("message.id")
                node_id = f"{svc}:{msg_id}" if msg_id else f"{svc}:{sid}"
                nodes[node_id] = rec
                span_to_node_id[sid] = node_id

                # D10: Advance temporal chain only on spans NOT marked
                # with trace.temporal_ignore (child LLM spans are ignored;
                # node spans advance the chain).
                if not temporal_ignore:
                    prev_span_id = sid

            # Post-process: remap any input refs that still use raw span IDs
            # through span_to_node_id so they point to stable message.id-based keys.
            for _nid, rec in nodes.items():
                for role, ref in list(rec.get("inputs", {}).items()):
                    if ref.startswith("lit:"):
                        continue
                    # ref format is "service:span_id" — extract the span_id part
                    if ":" in ref:
                        prefix, suffix = ref.split(":", 1)
                        if suffix in span_to_node_id and ref != span_to_node_id[suffix]:
                            rec["inputs"][role] = span_to_node_id[suffix]

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

