# Trace IO utilities

This directory hosts utilities for importing/serializing and
exporting/deserializing Trace graphs to and from other formats for richer
integration. It provides a compact Trace‑Graph JSON (TGJ) format and an
OpenTelemetry bridge so existing telemetry or JSON descriptions can be
replayed as Trace nodes.

## Modules

* `tgj_ingest.py` – load TGJ documents or the `trace-json/1.0+otel` profile.
  It resolves references, stitches multi‑agent graphs and yields real
  `Node`/`MessageNode`/`ParameterNode` objects that can participate in
  optimisation.
* `tgj_export.py` – serialise a Trace graph back to TGJ so it can be stored or
  exchanged.
* `otel_adapter.py` – convert OTLP traces to the OTel‑profile TGJ documents.

## Usage

### Importing JSON graphs
```python
from opto.trace.io.tgj_ingest import ingest_tgj
from opto.trace.nodes import MessageNode
from opto.trace.propagators.graph_propagator import GraphPropagator

doc = {...}  # TGJ document
nodes = ingest_tgj(doc)
loss = nodes["loss"]
GraphPropagator().init_feedback(loss, "minimise")
```
Trainable nodes defined in the JSON (`kind:"parameter"`) become
`ParameterNode` instances. They can be linked with code variables and updated
by running optimisation passes over the graph.

### Enriching graphs with logs
Logs encoded as TGJ documents can reference existing nodes via
`exports`/`imports`. Merging them adds observability information to the graph:
```python
from opto.trace.io.tgj_ingest import merge_tgj
merged = merge_tgj([base_graph_doc, log_doc])
```

### From OpenTelemetry
```python
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
for doc in otlp_traces_to_trace_json(otlp_payload):
    ingest_tgj(doc)
```

### Exploring parent/child branches
TGJ `inputs` link parents to children, and the ingestor rebuilds the hierarchy
automatically. The OTel adapter derives the same edges from `parentSpanId` so
span trees become Trace graphs. Once loaded you can explore or filter branches:
```python
from opto.trace.io.tgj_ingest import ingest_tgj
from opto.trace.nodes import MessageNode
from opto.trace.propagators.graph_propagator import GraphPropagator

nodes = ingest_tgj(branching_doc)
tg = nodes["merge"].backward("inspect", propagator=GraphPropagator(), retain_graph=True)
message_branch = [n.py_name for _, n in tg.graph if isinstance(n, MessageNode)]
```
`message_branch` now lists the message nodes on that path; any predicate over
`tg.graph` lets you filter or visualise specific branches.

## Tests
The integration tests in `tests/test_tgj_otel_integration.py` cover common
scenarios: ML training pipelines, multi‑agent stitching, log enrichment,
OTLP conversion and linking JSON parameters to executable code. Run them with:
```
pytest tests/test_tgj_otel_integration.py -q
```
