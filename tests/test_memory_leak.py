from memory_profiler import profile
import sys
from opto.trace import node, GRAPH, bundle
import numpy as np

# GRAPH.LEGACY_GRAPH_BEHAVIOR = True
# GRAPH.clear()

base = node(np.ones(10000000))

@bundle()
def add(x, y):
    return x + y

def fun(x):
    return x + np.ones(10000000)
    # return add(x, base)
    # return add(x, np.ones(10000000))

@profile
def test_multiple_backward():
    x = node(1, name="x", trainable=True)

    for i in range(100):
        y1 = fun(x)
        y2 = fun(x)
        x.zero_feedback()
        y1.backward("first backward")
        y2.backward("second backward")
        x.zero_feedback()

    print(len(x.feedback))  # should be 0
    # print(len(base.feedback))  # should be 0


if __name__ == "__main__":
    test_multiple_backward()