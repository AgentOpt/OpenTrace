import math
from opto.trace.nodes import Node, MessageNode, ParameterNode
from opto.trace.io.tgj_ingest import ingest_tgj, merge_tgj, TLSFIngestor
from opto.trace.io.tgj_export import export_subgraph_to_tgj
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json, PROFILE_VERSION
from opto.trace.propagators.graph_propagator import GraphPropagator

# ---------- 1) MLflow-style single-agent training pipeline ----------
MLFLOW_TGJ = {
  "tgj":"1.0","run_id":"run-mlf-1","agent_id":"trainer","graph_id":"train","scope":"trainer/0",
  "nodes":[
    {"id":"lr","kind":"parameter","name":"learning_rate","value":0.01,"trainable":True},
    {"id":"epochs","kind":"value","name":"epochs","value":3},
    {"id":"data","kind":"value","name":"dataset","value":"s3://bucket/train.csv"},
    {"id":"model","kind":"message","name":"model","description":"[train] fit(X,y)",
     "inputs":{"lr":{"ref":"lr"},"epochs":{"ref":"epochs"},"Xy":{"ref":"data"}},
     "output":{"name":"weights","value":{"w":[0.1,0.2]}} },
    {"id":"eval","kind":"message","name":"accuracy","description":"[eval] accuracy(model, X_valid)",
     "inputs":{"model":{"ref":"model"}}, "output":{"name":"acc","value":0.72}}
  ]
}

def test_mlflow_like_graph_backward():
    mp = ingest_tgj(MLFLOW_TGJ)
    acc = mp["accuracy"]
    assert isinstance(acc, MessageNode)
    gp = GraphPropagator()
    acc.backward("higher is better", propagator=gp, retain_graph=True)
    seen, stack, params = set(), [acc], []
    while stack:
        node = stack.pop()
        for parent in node.parents:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
                if isinstance(parent, ParameterNode):
                    params.append(parent)
    assert any(p.py_name.split('/')[-1].startswith("learning_rate") for p in params)

# ---------- 2) OpenTelemetry “Astronomy Shop” multi-agent ----------
ASTRO_CHECKOUT = {
  "tgj":"1.0","run_id":"trace-astro","agent_id":"checkout","graph_id":"svc","scope":"checkout/1",
  "nodes":[
    {"id":"req","kind":"value","name":"http_req","value":{"path":"/checkout","method":"POST"}},
    {"id":"checkout","kind":"message","name":"checkout","description":"[http:post] /checkout",
     "inputs":{"req":{"ref":"req"}}, "output":{"name":"order_id","value":"OID-1"}}
  ],
  "exports":{"port://order":{"ref":"checkout"}}
}
ASTRO_PAYMENT = {
  "tgj":"1.0","run_id":"trace-astro","agent_id":"payment","graph_id":"svc","scope":"payment/3",
  "imports":{"port://order":{"from_agent":"checkout","from_graph":"svc"}},
  "nodes":[
    {"id":"charge","kind":"message","name":"charge","description":"[rpc:grpc] charge",
     "inputs":{"order":{"export":"port://order"}}, "output":{"name":"receipt","value":"OK"}}
  ]
}

def test_astronomy_shop_multiagent_merge():
    merged = merge_tgj([ASTRO_CHECKOUT, ASTRO_PAYMENT])
    # sanity: both graphs loaded, edge wired through export
    ck = "checkout/svc/trace-astro"; pk = "payment/svc/trace-astro"
    assert "checkout" in merged[ck]["__TGJ_META__"]["scope"]
    charge = merged[pk]["charge"]; order = merged[ck]["checkout"]
    assert order in charge.parents

# ---------- 3) Kubernetes control-plane mini trace (scheduler -> kubelet) ----------
K8S_TGJ = {
  "tgj":"1.0","run_id":"trace-k8s","agent_id":"scheduler","graph_id":"s1","scope":"scheduler/1",
  "nodes":[
    {"id":"pod","kind":"value","name":"pod_spec","value":{"pod":"demo","cpu":"250m"}},
    {"id":"bind","kind":"message","name":"bind","description":"[schedule] bind pod",
     "inputs":{"spec":{"ref":"pod"}}, "output":{"name":"nodeName","value":"node-1"}}
  ],
  "exports":{"port://bind":{"ref":"bind"}}
}
K8S_TGJ2 = {
  "tgj":"1.0","run_id":"trace-k8s","agent_id":"kubelet","graph_id":"k1","scope":"kubelet/node-1",
  "nodes":[
    {"id":"start","kind":"message","name":"start","description":"[container] run",
     "inputs":{"binding":{"export":"port://bind"}}, "output":{"name":"status","value":"Running"}}
  ]
}

def test_k8s_stitch_and_backward():
    merged = merge_tgj([K8S_TGJ, K8S_TGJ2])
    klet = merged["kubelet/k1/trace-k8s"]["start"]
    sched = merged["scheduler/s1/trace-k8s"]["bind"]
    assert sched in klet.parents
    gp = GraphPropagator()
    klet.backward("keep containers running", propagator=gp, retain_graph=True)
    seen, stack, found = set(), [klet], False
    while stack:
        node = stack.pop()
        if node is sched:
            found = True
        for parent in node.parents:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    assert found

# ---------- 4) OTel adapter round-trip (tiny) ----------
def test_otel_adapter_minimal():
    otlp = {
      "resourceSpans": [{
        "resource": {"attributes":[{"key":"service.name","value":{"stringValue":"svcA"}},
                                   {"key":"service.instance.id","value":{"stringValue":"i1"}}]},
        "scopeSpans": [{
          "scope": {"name":"scopeA"},
          "spans": [{
            "traceId":"t-1","spanId":"s-1","name":"GET /items","kind":"SERVER",
            "startTimeUnixNano":"1","endTimeUnixNano":"1000000",
            "attributes":[{"key":"http.method","value":{"stringValue":"GET"}},
                          {"key":"http.url","value":{"stringValue":"/items"}}]
          }]
        }]
      }]
    }
    docs = otlp_traces_to_trace_json(otlp)
    assert docs and docs[0]["version"] == PROFILE_VERSION
    mp = ingest_tgj(docs[0])
    node = mp["GET /items"]
    assert isinstance(node, MessageNode)

# ---------- 5) Export → Import round-trip ----------
def test_export_import_roundtrip():
    # Build a mini graph in-memory and export
    x = ParameterNode(-1.0, name="x", trainable=True, description="[Parameter]")
    b = Node(1.0, name="b", description="[Node]")
    a = MessageNode(Node(None, name="a_out"), inputs={"x":x}, description="[bar] -2*x", name="a")
    y = MessageNode(Node(None, name="y_out"), inputs={"a":a,"b":b}, description="[add] a+b", name="y")
    from opto.trace.io.tgj_export import export_subgraph_to_tgj
    tgj = export_subgraph_to_tgj([y], run_id="r", agent_id="A", graph_id="g", scope="A/0")
    assert any(rec.get("op") for rec in tgj["nodes"] if rec["kind"]=="message")
    mp = ingest_tgj(tgj)
    y2 = mp["y"]
    assert isinstance(y2, MessageNode)
    # parents should be present
    assert any(p.py_name.split('/')[-1].startswith("a") for p in y2.parents)


def test_tlsf_ingestor_with_trace_json():
    otlp = {
      "resourceSpans": [{
        "resource": {"attributes":[{"key":"service.name","value":{"stringValue":"svcA"}},
                                   {"key":"service.instance.id","value":{"stringValue":"i1"}}]},
        "scopeSpans": [{
          "scope": {"name":"scopeA"},
          "spans": [{
            "traceId":"t-2","spanId":"s-2","name":"POST /submit","kind":"SERVER",
            "startTimeUnixNano":"1","endTimeUnixNano":"1000",
            "attributes":[{"key":"http.method","value":{"stringValue":"POST"}}]
          }]
        }]
      }]
    }
    docs = otlp_traces_to_trace_json(otlp)
    ing = TLSFIngestor()
    ing.ingest_tgj(docs[0])
    node = ing.get("POST /submit")
    assert isinstance(node, MessageNode)

# ---------- 6) Log enrichment via TGJ merge ----------
LOG_TGJ = {
  "tgj":"1.0","run_id":"trace-k8s","agent_id":"logger","graph_id":"log","scope":"logger/0",
  "imports":{"port://bind":{"from_agent":"scheduler","from_graph":"s1"}},
  "nodes":[
    {"id":"audit","kind":"message","name":"audit","description":"[log] bind recorded",
     "inputs":{"binding":{"export":"port://bind"}}, "output":{"name":"logline","value":"bind logged"}}
  ]
}

def test_log_enrichment_from_tgj():
    merged = merge_tgj([K8S_TGJ, LOG_TGJ])
    audit = merged["logger/log/trace-k8s"]["audit"]
    bind = merged["scheduler/s1/trace-k8s"]["bind"]
    assert bind in audit.parents

# ---------- 7) Link JSON parameter to executable code ----------
TRAINABLE_TGJ = {
  "tgj":"1.0","run_id":"rt","agent_id":"agent","graph_id":"g","scope":"agent/0",
  "nodes":[
    {"id":"w","kind":"parameter","name":"weight","value":1.0,"trainable":True},
    {"id":"x","kind":"value","name":"input","value":2.0},
    {"id":"prod","kind":"message","name":"prod","description":"[mul] weight*input",
     "inputs":{"w":{"ref":"w"},"x":{"ref":"x"}}, "output":{"name":"p_out","value":2.0}}
  ]
}

def test_link_trainable_parameter_from_json():
    mp = ingest_tgj(TRAINABLE_TGJ)
    w = mp["weight"]
    assert isinstance(w, ParameterNode)
    loss = MessageNode(Node(w.data ** 2, name="loss_out"), inputs={"w": w}, description="[square] w^2", name="loss")
    gp = GraphPropagator()
    loss.backward("minimize", propagator=gp, retain_graph=True)
    seen, stack, params = set(), [loss], []
    while stack:
        node = stack.pop()
        for parent in node.parents:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
                if isinstance(parent, ParameterNode):
                    params.append(parent)
    assert w in params

# ---------- 8) Branch reconstruction and filtering ----------
BRANCH_TGJ = {
  "tgj":"1.0","run_id":"r-branch","agent_id":"agent","graph_id":"g","scope":"agent/0",
  "nodes":[
    {"id":"x","kind":"value","name":"x","value":1},
    {"id":"dup","kind":"message","name":"dup","description":"[dup] x",
     "inputs":{"x":{"ref":"x"}}, "output":{"name":"x2","value":1}},
    {"id":"left","kind":"message","name":"left","description":"[add] dup+1",
     "inputs":{"d":{"ref":"dup"}}, "output":{"name":"l","value":2}},
    {"id":"right","kind":"message","name":"right","description":"[sub] dup-1",
     "inputs":{"d":{"ref":"dup"}}, "output":{"name":"r","value":0}},
    {"id":"merge","kind":"message","name":"merge","description":"[add] left+right",
     "inputs":{"a":{"ref":"left"},"b":{"ref":"right"}}, "output":{"name":"m","value":2}}
  ]
}

def test_branch_reconstruction_and_filtering():
    mp = ingest_tgj(BRANCH_TGJ)
    merge = mp["merge"]
    visited, stack, msg_names, value_names = set(), [merge], [], []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        base = node.name.split('/')[-1].split(":")[0]
        if isinstance(node, MessageNode):
            msg_names.append(base)
        else:
            value_names.append(base)
        stack.extend(node.parents)
    assert set(["merge", "left", "right", "dup"]).issubset(set(msg_names))
    assert "x" in value_names

# ---------- 9) OTel parent-child reconstruction ----------
OTLP_BRANCH = {
  "resourceSpans": [{
    "resource": {"attributes":[{"key":"service.name","value":{"stringValue":"svc"}}]},
    "scopeSpans": [{
      "scope": {"name":"scope"},
      "spans": [
        {"traceId":"t","spanId":"p","name":"parent","kind":"SERVER"},
        {"traceId":"t","spanId":"c1","parentSpanId":"p","name":"child1","kind":"INTERNAL"},
        {"traceId":"t","spanId":"c2","parentSpanId":"p","name":"child2","kind":"INTERNAL"}
      ]
    }]
  }]
}

def test_otel_parent_child_hierarchy():
    docs = otlp_traces_to_trace_json(OTLP_BRANCH)
    mp = ingest_tgj(docs[0])
    child1 = mp["child1"]
    parent = mp["parent"]
    # parent id recovered automatically from parentSpanId
    assert child1.parents[0].name.split('/')[-1].split(":")[0] == "p"
    # manual relink to the full parent node
    child1.parents[0] = parent
    child2 = mp["child2"]
    child2.parents[0] = parent
    visited, stack, names = set(), [child2], []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        names.append(node.name.split('/')[-1].split(":")[0])
        stack.extend(node.parents)
    assert "parent" in names and "child1" not in names
    child_nodes = [n for n in visited if n.name.split('/')[-1].split(":")[0].startswith("child")]
    assert all(isinstance(n, MessageNode) for n in child_nodes)
