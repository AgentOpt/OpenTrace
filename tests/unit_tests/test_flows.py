from opto import trace, flows

def test_trace_llm():

    si = trace.node("You're a helpful assistant.", trainable=True)
    user_prompt = "Hi there"
    traced_llm = flows.TracedLLM(si)  # this is trace.Module
    response = traced_llm(user_prompt)
