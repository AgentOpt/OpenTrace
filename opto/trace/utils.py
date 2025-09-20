from graphviz import Digraph
import builtins
import re
import json

# Get a list of all names in the builtins module
builtins_list = dir(builtins)
# Filter for function names; this includes exceptions, so you might want to refine this
global_functions_list = [
    name for name in builtins_list if callable(getattr(builtins, name))
]


def sum_feedback(nodes):
    """Aggregate feedback from a list of nodes.
    
    Sums all feedback values across all feedback channels for the given nodes.
    
    Parameters
    ----------
    nodes : list
        List of nodes containing feedback to aggregate.
    
    Returns
    -------
    Any
        Aggregated feedback value, typically the sum of all feedback.
    
    Notes
    -----
    Each node may have multiple feedback channels (stored in feedback.values()).
    This function sums across all channels and all nodes.
    """
    return sum([sum(gg) for p in nodes for gg in p.feedback.values()])


def contain(container_of_nodes, node):
    """Check if a node is contained in a collection by identity.
    
    Parameters
    ----------
    container_of_nodes : iterable
        Collection of nodes to search in.
    node : Node
        Node to search for.
    
    Returns
    -------
    bool
        True if node is found in container (by identity, not value).
    
    Notes
    -----
    Uses identity comparison (is) rather than value comparison (==)
    to ensure exact object matching.
    """
    # check for identity instead of value
    return any([node is n for n in container_of_nodes])


def parse_eqs_to_dict(text):
    """Parse text containing variable assignments into a dictionary.

    Parameters
    ----------
    text : str
        Text containing variable assignments in the format 'key = value'.

    Returns
    -------
    dict[str, str]
        Dictionary mapping variable names to their values.

    Notes
    -----
    Handles multi-line values by concatenating lines without '=' to
    the previous key's value. Removes backticks from values.

    Examples
    --------
    Input:
        x0 = 1
        x1=2
        x2=`2`
        x3= def fun():\\n    print('hello')\\n
        abc_test1=test

    Output:
        {'x0': '1', 'x1': '2', 'x2': '2', 'x3': "def fun():\\nprint('hello')", 'abc_test1': 'test'}
    """
    lines = text.split("\n")
    result_dict = {}
    last_key = None
    for line in lines:
        if line == "":
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            last_key = key.strip()
            result_dict[last_key] = value.replace("`", "")
        elif last_key:
            result_dict[last_key] += "\n" + line.replace("`", "")
    return result_dict


def for_all_methods(decorator):
    """Apply a decorator to all methods of a class.
    
    Class decorator that applies the given decorator to all non-dunder
    methods of the decorated class.
    
    Parameters
    ----------
    decorator : callable
        Decorator function to apply to class methods.
    
    Returns
    -------
    callable
        Class decorator function.
    
    Examples
    --------
    >>> @for_all_methods(my_decorator)
    ... class MyClass:
    ...     def method1(self):
    ...         pass
    ...     def method2(self):
    ...         pass
    
    Notes
    -----
    Only applies to callable attributes that don't start with '__'.
    Useful for applying logging, timing, or validation to all methods.
    """

    def decorate(cls):
        for name, attr in cls.__dict__.items():
            if callable(attr) and not name.startswith("__"):
                setattr(cls, name, decorator(attr))
        return cls

    return decorate


def render_opt_step(step_idx, optimizer, no_trace_graph=False, no_improvement=False):
    """Render an optimization step as HTML for Jupyter notebook display.
    
    Creates a visual representation of an optimization step showing the
    trace graph, feedback, reasoning, and suggested improvements.
    
    Parameters
    ----------
    step_idx : int
        Index of the optimization step to render.
    optimizer : Optimizer
        Optimizer instance containing logs and summaries.
    no_trace_graph : bool, default=False
        If True, omits the trace graph from the display.
    no_improvement : bool, default=False
        If True, omits the improvement section from the display.
    
    Returns
    -------
    None
        Displays HTML output directly in Jupyter notebook.
    
    Notes
    -----
    Requires IPython display capabilities. Creates color-coded boxes for:
    - Gray: Trace graph showing computation flow
    - Red: Feedback indicating issues or goals
    - Green: Reasoning about improvements
    - Blue: Suggested parameter updates
    """
    from IPython.display import display, HTML

    idx = step_idx
    llm_response = json.loads(optimizer.log[idx]["response"])
    r1 = llm_response["reasoning"]

    if llm_response.get("suggestion"):
        a1 = "".join(
            [
                f"{var_name}:\n\n{var_body}\n\n"
                for var_name, var_body in llm_response["suggestion"].items()
            ]
        )
    elif llm_response.get("answer") is not None:
        a1 = llm_response["answer"]
    else:
        a1 = "<ERROR> NULL/INVALID RESPONSE"

    pi = optimizer.summary_log[idx]["problem_instance"]  # full
    f1 = pi.feedback

    masked = ["#Feedback", "#Others", "#Instruction"]
    pi = optimizer.problem_instance(optimizer.summary_log[idx]["summary"], mask=masked)

    # a hack to remove "#Feedback:" because it has a colon
    pi = str(pi)
    pi = pi.replace("#Feedback:", "#Feedback")

    for m in masked:
        pi = pi.replace(m + "\n", "")

    # a quick processing to reduce multiple empty lines to one
    pi = re.sub(r"\n\s*\n", "\n\n", pi)
    g1 = pi

    html_template = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin-bottom: 10px;">
            <!-- First set of blocks -->
    """

    if not no_trace_graph:
        html_template += f"""
            <div style="display: flex; align-items: stretch; margin-bottom: 10px;">
                <div style="flex-grow: 1; background-color: #E0E0E0; border: 2px solid #9E9E9E; padding: 10px; border-radius: 5px; width: 550px;">
                    <p><b>Trace Graph</b></p><pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{g1}</pre>
                </div>
                <div style="width: 40px; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #9E9E9E;">
                    g<sub>{idx}</sub>
                </div>
            </div>
        """
    html_template += f"""
            <div style="display: flex; align-items: stretch; margin-bottom: 10px;">
                <div style="flex-grow: 1; background-color: #FFB3BA; border: 2px solid #FF6B6B; padding: 10px; border-radius: 5px;">
                    <p style="margin: 0;"><b>Feedback: </b>{f1}</p>
                </div>
                <div style="width: 40px; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #FF6B6B;">
                    f<sub>{idx}</sub>
                </div>
            </div>

            <div style="display: flex; align-items: stretch; margin-bottom: 10px;">
                <div style="flex-grow: 1; background-color: #BAFFC9; border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; width: 550px;">
                    <p style="margin: 0;"><b>Reasoning: </b>{r1}</p>
                </div>
                <div style="width: 40px; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #4CAF50;">
                    r<sub>{idx + 1}</sub>
                </div>
            </div>
        """

    if not no_improvement:
        html_template += f"""
                <div style="display: flex; align-items: stretch; margin-bottom: 20px;">
                <div style="flex-grow: 1; background-color: 'white'; border: 2px solid #4D9DE0; padding: 10px; border-radius: 5px;">
                    <p><b>Improvement</b></p>
                    <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; background-color: 'white';">{a1}</pre>
                </div>
                <div style="width: 40px; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #4D9DE0;">
                    a<sub>{idx + 1}</sub>
                </div>
            </div>
        """

    html_template += "</div>"

    display(HTML(html_template))


def escape_json_nested_quotes(json_str):
    """
    Escapes double quotation marks inside JSON string values for a specific format:
    {"name": "string value", "value": "string value"}
    Does not escape quotes around keys or structural quotes.

    Warning:
        Here are what this function does not do:
        1. Cannot handle "\\n" or "\\t" type of strings
        2. Does not check if "\\n", "\\t", or other control characters are properly escaped.
        Please use json_str.replace("\\n", "\\n") to escape control characters outside of this function.

    Example usage can be found in optimizers/textgrad.py.

    Args:
        json_str (str): A string representation of JSON with exactly two keys: name and value.

    Returns:
        str: JSON string with properly escaped quotes in values.
    """
    result = []
    i = 0
    in_value = False
    while i < len(json_str):
        char = json_str[i]

        if char == '"':
            # Check if this quote is around "name" or "value"
            next_four = json_str[i + 1: i + 5]
            next_five = json_str[i + 1: i + 6]
            is_key = next_four == "name" or next_five == "value"

            # Check if this is a structural quote (after : or before })
            prev_char = json_str[i - 1] if i > 0 else ""
            next_char = json_str[i + 1] if i < len(json_str) - 1 else ""
            is_value_boundary = (
                    prev_char == ":"
                    or (prev_char == " " and json_str[i - 2] == ":")
                    or next_char == "}"
                    or next_char == ","
            )

            if is_key or is_value_boundary:
                result.append(char)
                if prev_char == ":" or (prev_char == " " and json_str[i - 2] == ":"):
                    in_value = True
                if next_char == "}" or next_char == ",":
                    in_value = False
            else:
                # if we double-escpaed like \\", we remove one
                if in_value and prev_char == "\\" and json_str[i - 2] == "\\":
                    result.pop(-1)
                    result.append(char)
                # If we're in a value and this is not a boundary quote, escape it
                elif in_value and prev_char != "\\":
                    result.append(r"\"")
                else:
                    result.append(char)
        else:
            # we need to remove markdown latex syntax
            # it's a simple procedure that removes all "\\alpha" or "\\(" type strings
            # JSON can't accept any \ with invalid characters, in here we took a short cut and only keep \ for

            # we didn't add \u to this list
            if json_str[i - 1] == "\\" and char not in [
                "\\",
                "\\/",
                "n",
                "b",
                "f",
                "r",
                "t",
            ]:
                result.pop(-1)

            result.append(char)

        # print(in_value, ''.join(result))
        i += 1

    return "".join(result)


def remove_non_ascii(json_txt):
    """Remove non-ASCII and non-printable characters from JSON text.
    
    Cleans JSON strings by removing control characters and non-printable
    characters while preserving valid escape sequences.
    
    Parameters
    ----------
    json_txt : str
        JSON text that may contain non-ASCII or control characters.
    
    Returns
    -------
    str
        Cleaned JSON text with only printable ASCII characters.
    
    Notes
    -----
    First applies escape_json_nested_quotes, then removes:
    - Newlines, tabs, backspaces, carriage returns, form feeds
    - Any other non-printable characters
    
    Example usage can be found in optimizers/textgrad.py.
    """
    cleaned = ""
    for c in escape_json_nested_quotes(json_txt):
        if c not in ["\n", "\t", "\b", "\r", "\f"] and not c.isprintable():
            continue
        cleaned += c
    return cleaned


def dedent(text: str):
    """Remove leading and trailing whitespace from each line.
    
    A simpler alternative to textwrap.dedent that strips whitespace
    from the beginning and end of each line individually, rather than
    removing common leading whitespace.
    
    Parameters
    ----------
    text : str
        Multi-line text to dedent.
    
    Returns
    -------
    str
        Text with each line stripped of leading/trailing whitespace.
    
    Examples
    --------
    >>> text = '''\n        Line 1 has leading space\n            Line 2 has more\n        '''
    >>> dedent(text)
    'Line 1 has leading space\nLine 2 has more'
    
    Notes
    -----
    Unlike textwrap.dedent, this function:
    - Strips each line independently
    - Removes ALL leading/trailing whitespace per line
    - Useful for cleaning up multi-line prompts in code
    """
    return "\n".join([line.strip() for line in text.split("\n")])


def test_json_quote_escaper():
    """Test suite for escape_json_nested_quotes function.
    
    Verifies that the JSON quote escaper correctly handles various
    edge cases including nested quotes, already-escaped quotes, and
    special characters.
    
    Raises
    ------
    AssertionError
        If any test case fails to produce expected output.
    
    Notes
    -----
    Tests cover:
    - Multiple quotes within string values
    - Quotes at various positions
    - Already escaped quotes
    - LaTeX-style escape sequences
    """
    test_cases = [
        (
            '{"name": "Multiple "quotes" in "one" string", "value": "Multiple "quotes" in "the second" string"}',
            r'{"name": "Multiple \"quotes\" in \"one\" string", "value": "Multiple \"quotes\" in \"the second\" string"}',
        ),
        (
            '{"name": "Simple "quote"", "value": "Another "quote""}',
            r'{"name": "Simple \"quote\"", "value": "Another \"quote\""}',
        ),
        (
            '{"name": "No quotes here", "value": "But "quotes" here"}',
            r'{"name": "No quotes here", "value": "But \"quotes\" here"}',
        ),
        (
            '{"name": "Quote at "end"", "value": "Another at "end""}',
            r'{"name": "Quote at \"end\"", "value": "Another at \"end\""}',
        ),
        (
            r'{"name": "Quote at "end"", "value": "Partial at \"end""}',
            r'{"name": "Quote at \"end\"", "value": "Partial at \"end\""}',
        ),
        (
            r'{"name": "Quote at \\"end\\"", "value": "Partial at \"end""}',
            r'{"name": "Quote at \"end\"", "value": "Partial at \"end\""}',
        ),
        (
            r'{"name": "Quote at \\"end\\"", "value": "\( \alpha_t \) \\n"}',
            r'{"name": "Quote at \"end\"", "value": "( alpha_t ) \\n"}',
        ),
    ]

    for i, (input_str, expected) in enumerate(test_cases, 1):
        result = escape_json_nested_quotes(input_str)
        assert (
                result == expected
        ), f"\nTest case {i} failed:\nInput:    {input_str}\nExpected: {expected}\nGot:      {result}"

    print("All tests passed!")
