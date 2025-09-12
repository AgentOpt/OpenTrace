from opto import trace
import pickle

@trace.model
class Dummy:
    def forward(self, x):
        return x * 2


dummy = Dummy()
pickle.dumps(dummy)

try:
    dummy.export("dummy.py")
except Exception as e:
    print("Export failed:", e)