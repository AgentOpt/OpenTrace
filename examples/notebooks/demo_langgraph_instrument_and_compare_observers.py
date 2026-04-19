#!/usr/bin/env python3
"""
LangGraph / OTEL / sys.monitoring comparison demo.

Demonstrates using instrument_graph with different backends:
- trace, trace+otel, trace+sysmon, trace+otel+sysmon
- otel, otel+sysmon  
- sysmon
"""

import sys
import time
from langgraph.graph import StateGraph, START, END
from opto.trace.io import instrument_graph


HAS_SYSMON = hasattr(sys, "monitoring")


def build_graph():
    """Build a simple planner->synth graph."""
    def planner(state):
        return {"plan": f"plan::{state['query']}"}

    def synth(state):
        query = state.get("query", "")
        plan = state.get("plan", "")
        return {"final_answer": f"answer::{query}::{plan}"}

    g = StateGraph(dict)
    g.add_node("planner", planner)
    g.add_node("synth", synth)
    g.add_edge(START, "planner")
    g.add_edge("planner", "synth")
    g.add_edge("synth", END)
    return g


def run_test(name, instrument_kwargs):
    """Run a single instrumentation test."""
    print(f"\nTest: {name}")
    try:
        t0 = time.perf_counter()
        
        # Build and instrument graph  
        graph = build_graph()
        if "backend" in instrument_kwargs and instrument_kwargs["backend"] == "trace":
            # For trace backend, pass graph_factory and scope
            instrumented = instrument_graph(
                graph_factory=build_graph,
                scope=globals(),
                **instrument_kwargs
            )
        else:
            # For otel/sysmon, pass compiled graph
            instrumented = instrument_graph(
                graph=graph.compile(),
                **instrument_kwargs
            )
        
        # Invoke
        result = instrumented.invoke({"query": "What is CRISPR?"})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        # Extract answer
        answer = result.get("final_answer", result)
        
        print(f"  ✓ SUCCESS ({dt_ms:.1f}ms)")
        print(f"    Answer: {str(answer)[:80]}")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("LangGraph Instrumentation Backends Comparison")
    print("=" * 80)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"sys.monitoring available: {HAS_SYSMON}\n")
    
    results = {}
    
    # Test 1: trace backend
    results["trace"] = run_test(
        "backend='trace'",
        {"backend": "trace", "output_key": "final_answer"}
    )
    
    # Test 2: trace + otel
    results["trace+otel"] = run_test(
        "backend='trace', observe_with=('otel',)",
        {
            "backend": "trace",
            "observe_with": ("otel",),
            "output_key": "final_answer"
        }
    )
    
    # Test 3-4: trace + sysmon variants (if available)
    if HAS_SYSMON:
        results["trace+sysmon"] = run_test(
            "backend='trace', observe_with=('sysmon',)",
            {
                "backend": "trace",
                "observe_with": ("sysmon",),
                "output_key": "final_answer"
            }
        )
        
        results["trace+otel+sysmon"] = run_test(
            "backend='trace', observe_with=('otel', 'sysmon')",
            {
                "backend": "trace",
                "observe_with": ("otel", "sysmon"),
                "output_key": "final_answer"
            }
        )
    
    # Test 5: otel backend
    results["otel"] = run_test(
        "backend='otel'",
        {"backend": "otel", "output_key": "final_answer"}
    )
    
    # Test 6: otel + sysmon (if available)
    if HAS_SYSMON:
        results["otel+sysmon"] = run_test(
            "backend='otel', observe_with=('sysmon',)",
            {
                "backend": "otel",
                "observe_with": ("sysmon",),
                "output_key": "final_answer"
            }
        )
        
        # Test 7: sysmon backend
        results["sysmon"] = run_test(
            "backend='sysmon'",
            {
                "backend": "sysmon",
                "output_key": "final_answer"
            }
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    # Final assertions
    assert results.get("trace", False), "trace backend must pass"
    assert results.get("otel", False), "otel backend must pass"
    
    print("\n✓ All critical tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
