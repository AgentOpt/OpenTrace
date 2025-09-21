from typing import Any, List, Dict, Union, Tuple
from dataclasses import dataclass, asdict
from textwrap import dedent, indent
import warnings
import json
import re
import copy
import pickle
from opto.trace.nodes import ParameterNode, Node, MessageNode
from opto.trace.propagators import TraceGraph, GraphPropagator
from opto.trace.propagators.propagators import Propagator
from opto.optimizers.optimizer import Optimizer
from opto.optimizers.buffers import FIFOBuffer
from opto.utils.llm import AbstractModel, LLM


def get_fun_name(node: MessageNode):
    """Extract the function name from a MessageNode.

    Parameters
    ----------
    node : MessageNode
        The node to extract the function name from.

    Returns
    -------
    str
        The function name, either from node.info['fun_name'] or
        extracted from the node name.
    """
    if isinstance(node.info, dict) and "fun_name" in node.info:
        return node.info["fun_name"]
    return node.name.split(":")[0]


def repr_function_call(child: MessageNode):
    """Generate a string representation of a function call from a MessageNode.

    Parameters
    ----------
    child : MessageNode
        The node representing a function call.

    Returns
    -------
    str
        String representation in format: 'output = function(arg1=val1, arg2=val2)'.
    """
    function_call = f"{child.py_name} = {get_fun_name(child)}("
    for k, v in child.inputs.items():
        function_call += f"{k}={v.py_name}, "
    function_call = function_call[:-2] + ")"
    return function_call


def node_to_function_feedback(node_feedback: TraceGraph):
    """Convert a TraceGraph to a FunctionFeedback structure.

    Parameters
    ----------
    node_feedback : TraceGraph
        The trace graph containing nodes and feedback to convert.

    Returns
    -------
    FunctionFeedback
        Structured feedback with separated roots, intermediates, and outputs.

    Notes
    -----
    The conversion process:
    1. Traverses the graph in topological order
    2. Classifies nodes as roots, intermediates, or outputs
    3. Extracts function documentation and call representations
    4. Preserves user feedback from the original graph

    Roots include both true root nodes and 'blanket' nodes whose
    parents haven't been visited yet.
    """
    depth = 0 if len(node_feedback.graph) == 0 else node_feedback.graph[-1][0]
    graph = []
    others = {}
    roots = {}
    output = {}
    documentation = {}

    visited = set()
    for level, node in node_feedback.graph:
        # the graph is already sorted
        visited.add(node)

        if node.is_root:  # Need an or condition here
            roots.update({node.py_name: (node.data, node.description)})
        else:
            # Some might be root (i.e. blanket nodes) and some might be intermediate nodes
            # Blanket nodes belong to roots
            if all([p in visited for p in node.parents]):
                # this is an intermediate node
                assert isinstance(node, MessageNode)
                documentation.update({get_fun_name(node): node.description})
                graph.append((level, repr_function_call(node)))
                if level == depth:
                    output.update({node.py_name: (node.data, node.description)})
                else:
                    others.update({node.py_name: (node.data, node.description)})
            else:
                # this is a blanket node (classified into roots)
                roots.update({node.py_name: (node.data, node.description)})

    return FunctionFeedback(
        graph=graph,
        others=others,
        roots=roots,
        output=output,
        user_feedback=node_feedback.user_feedback,
        documentation=documentation,
    )


@dataclass
class FunctionFeedback:
    """Container for structured feedback from function execution traces.

    Used by OptoPrime to organize execution traces into a format suitable
    for LLM-based optimization.

    Attributes
    ----------
    graph : list[tuple[int, str]]
        Topologically sorted function calls with (depth, representation) pairs.
    documentation : dict[str, str]
        Mapping of function names to their documentation strings.
    others : dict[str, Any]
        Intermediate variables with (data, description) tuples.
    roots : dict[str, Any]
        Input/root variables with (data, description) tuples.
    output : dict[str, Any]
        Output/leaf variables with (data, description) tuples.
    user_feedback : str
        User-provided feedback about the execution.

    Notes
    -----
    This structure separates the execution trace into logical components
    that can be formatted into prompts for LLM-based optimization.
    """

    graph: List[
        Tuple[int, str]
    ]  # Each item is is a representation of function call. The items are topologically sorted.
    documentation: Dict[str, str]  # Function name and its documentationstring
    others: Dict[str, Any]  # Intermediate variable names and their data
    roots: Dict[str, Any]  # Root variable name and its data
    output: Dict[str, Any]  # Leaf variable name and its data
    user_feedback: str  # User feedback at the leaf of the graph


@dataclass
class ProblemInstance:
    instruction: str
    code: str
    documentation: str
    variables: str
    inputs: str
    others: str
    outputs: str
    feedback: str
    constraints: str

    problem_template = dedent(
        """
        #Instruction
        {instruction}

        #Code
        {code}

        #Documentation
        {documentation}

        #Variables
        {variables}

        #Constraints
        {constraints}

        #Inputs
        {inputs}

        #Others
        {others}

        #Outputs
        {outputs}

        #Feedback
        {feedback}
        """
    )

    def __repr__(self) -> str:
        return self.problem_template.format(
            instruction=self.instruction,
            code=self.code,
            documentation=self.documentation,
            variables=self.variables,
            constraints=self.constraints,
            inputs=self.inputs,
            outputs=self.outputs,
            others=self.others,
            feedback=self.feedback,
        )


class OptoPrime(Optimizer):
    """Language model-based optimizer for text and code parameters.
    
    OptoPrime implements optimization through structured problem representation and 
    language model reasoning. It converts execution traces into problem instances 
    that language models can understand and improve.
    
    The optimizer operates by:
    1. Collecting execution traces and feedback from the computation graph
    2. Converting traces into structured problem representations
    3. Prompting language models to suggest parameter improvements
    4. Extracting and applying suggested updates to parameters
    
    Parameters
    ----------
    parameters : list[ParameterNode]
        List of trainable parameters to optimize.
    llm : AbstractModel, optional
        Language model for generating parameter updates, by default None (uses default LLM).
    propagator : Propagator, optional
        Custom propagator for trace graph processing, by default None.
    objective : str, optional
        Optimization objective description, by default uses default_objective.
    ignore_extraction_error : bool, default=True
        Whether to ignore type conversion errors when extracting LLM suggestions.
    include_example : bool, default=False
        Whether to include example problems in prompts.
    memory_size : int, default=0
        Size of feedback memory buffer for historical context.
    max_tokens : int, default=4096
        Maximum tokens for language model responses.
    log : bool, default=True
        Whether to log optimization steps and responses.
    prompt_symbols : dict, optional
        Custom symbols for prompt sections (e.g., "#Variables", "#Code").
    json_keys : dict, optional
        Keys for JSON response format (reasoning, answer, suggestion).
    use_json_object_format : bool, default=True
        Whether to request JSON object format from LLM.
    highlight_variables : bool, default=False
        Whether to highlight variables at the end of prompts.
    **kwargs
        Additional keyword arguments passed to parent class.
    
    Attributes
    ----------
    llm : AbstractModel
        The language model used for optimization.
    objective : str
        The optimization objective description.
    log : list or None
        Log of optimization steps if logging is enabled.
    summary_log : list or None
        Log of problem summaries if logging is enabled.
    memory : FIFOBuffer
        Buffer storing historical feedback.
    
    Methods
    -------
    summarize()
        Aggregate feedback into structured problem representation.
    problem_instance(summary, mask=None)
        Create a ProblemInstance from aggregated feedback.
    extract_llm_suggestion(response)
        Parse LLM response to extract parameter updates.
    
    Notes
    -----
    OptoPrime excels at optimizing:
    - Natural language prompts and instructions
    - Code implementations and algorithms
    - Mixed text-code parameters
    - Parameters with complex constraints
    
    The optimizer uses structured problem representations that separate:
    - Variables (trainable parameters)
    - Inputs (non-trainable values)
    - Code (execution trace)
    - Outputs (results)
    - Feedback (optimization signals)
    
    This structure enables language models to understand the optimization
    context and suggest targeted improvements.
    
    See Also
    --------
    Optimizer : Base optimizer class
    OptoPrimeV2 : Enhanced version with improved prompt engineering
    TextGrad : Alternative text-based optimizer
    
    Examples
    --------
    >>> from opto.optimizers import OptoPrime
    >>> from opto.trace import node
    >>> 
    >>> # Create trainable parameters
    >>> prompt = node("Explain quantum computing", trainable=True)
    >>> 
    >>> # Initialize optimizer
    >>> optimizer = OptoPrime([prompt], objective="Make explanation clearer")
    >>> 
    >>> # Run optimization loop
    >>> for _ in range(5):
    ...     output = model(prompt)
    ...     feedback = evaluate(output)
    ...     optimizer.backward(feedback)
    ...     optimizer.step()
    """
    # This is generic representation prompt, which just explains how to read the problem.
    representation_prompt = dedent(
        """
        You're tasked to solve a coding/algorithm problem. You will see the instruction, the code, the documentation of each function used in the code, and the feedback about the execution result.

        Specifically, a problem will be composed of the following parts:
        - #Instruction: the instruction which describes the things you need to do or the question you should answer.
        - #Code: the code defined in the problem.
        - #Documentation: the documentation of each function used in #Code. The explanation might be incomplete and just contain high-level description. You can use the values in #Others to help infer how those functions work.
        - #Variables: the input variables that you can change.
        - #Constraints: the constraints or descriptions of the variables in #Variables.
        - #Inputs: the values of other inputs to the code, which are not changeable.
        - #Others: the intermediate values created through the code execution.
        - #Outputs: the result of the code output.
        - #Feedback: the feedback about the code's execution result.

        In #Variables, #Inputs, #Outputs, and #Others, the format is:

        <data_type> <variable_name> = <value>

        If <type> is (code), it means <value> is the source code of a python code, which may include docstring and definitions.
        """
    )

    # Optimization
    default_objective = "You need to change the <value> of the variables in #Variables to improve the output in accordance to #Feedback."

    output_format_prompt_original = dedent(
        """
        Output_format: Your output should be in the following json format, satisfying the json syntax:

        {{
        "{reasoning}": <Your reasoning>,
        "{answer}": <Your answer>,
        "{suggestion}": {{
            <variable_1>: <suggested_value_1>,
            <variable_2>: <suggested_value_2>,
        }}
        }}

        In "{reasoning}", explain the problem: 1. what the #Instruction means 2. what the #Feedback on #Output means to #Variables considering how #Variables are used in #Code and other values in #Documentation, #Inputs, #Others. 3. Reasoning about the suggested changes in #Variables (if needed) and the expected result.

        If #Instruction asks for an answer, write it down in "{answer}".

        If you need to suggest a change in the values of #Variables, write down the suggested values in "{suggestion}". Remember you can change only the values in #Variables, not others. When <type> of a variable is (code), you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature.

        If no changes or answer are needed, just output TERMINATE.
        """
    )

    output_format_prompt_no_answer = dedent(
        """
        Output_format: Your output should be in the following json format, satisfying the json syntax:

        {{
        "{reasoning}": <Your reasoning>,
        "{suggestion}": {{
            <variable_1>: <suggested_value_1>,
            <variable_2>: <suggested_value_2>,
        }}
        }}

        In "{reasoning}", explain the problem: 1. what the #Instruction means 2. what the #Feedback on #Output means to #Variables considering how #Variables are used in #Code and other values in #Documentation, #Inputs, #Others. 3. Reasoning about the suggested changes in #Variables (if needed) and the expected result.

        If you need to suggest a change in the values of #Variables, write down the suggested values in "{suggestion}". Remember you can change only the values in #Variables, not others. When <type> of a variable is (code), you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature.

        If no changes are needed, just output TERMINATE.
        """
    )


    example_problem_template = dedent(
        """
        Here is an example of problem instance and response:

        ================================
        {example_problem}
        ================================

        Your response:
        {example_response}
        """
    )

    user_prompt_template = dedent(
        """
        Now you see problem instance:

        ================================
        {problem_instance}
        ================================

        """
    )

    example_prompt = dedent(
        """

        Here are some feasible but not optimal solutions for the current problem instance. Consider this as a hint to help you understand the problem better.

        ================================

        {examples}

        ================================
        """
    )

    final_prompt = dedent(
        """
        Your response:
        """
    )

    final_prompt_with_variables = dedent(
        """
        What are your suggestions on variables {names}?

        Your response:
        """
    )

    default_prompt_symbols = {
        "variables": "#Variables",
        "constraints": "#Constraints",
        "inputs": "#Inputs",
        "outputs": "#Outputs",
        "others": "#Others",
        "feedback": "#Feedback",
        "instruction": "#Instruction",
        "code": "#Code",
        "documentation": "#Documentation",
    }

    default_json_keys = {
        "reasoning": "reasoning",
        "answer": "answer",
        "suggestion": "suggestion",
    }

    def __init__(
        self,
        parameters: List[ParameterNode],
        llm: AbstractModel = None,
        *args,
        propagator: Propagator = None,
        objective: Union[None, str] = None,
        ignore_extraction_error: bool = True,  # ignore the type conversion error when extracting updated values from LLM's suggestion
        include_example=False,  # TODO # include example problem and response in the prompt
        memory_size=0,  # Memory size to store the past feedback
        max_tokens=4096,
        log=True,
        prompt_symbols=None,
        json_keys=None,  # keys to use in the json object format (can remove "answer" if not needed)
        use_json_object_format=True,  # whether to use json object format for the response when calling LLM
        highlight_variables=False,  # whether to highlight the variables at the end in the prompt
        **kwargs,
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)
        self.ignore_extraction_error = ignore_extraction_error
        self.llm = llm or LLM()
        self.objective = objective or self.default_objective
        self.example_problem = ProblemInstance.problem_template.format(
            instruction=self.default_objective,
            code="y = add(x=a,y=b)\nz = subtract(x=y, y=c)",
            documentation="add: add x and y \nsubtract: subtract y from x",
            variables="(int) a = 5",
            constraints="a: a > 0",
            outputs="(int) z = 1",
            others="(int) y = 6",
            inputs="(int) b = 1\n(int) c = 5",
            feedback="The result of the code is not as expected. The result should be 10, but the code returns 1",
            stepsize=1,
        )
        self.example_response = dedent(
            """
            {"reasoning": 'In this case, the desired response would be to change the value of input a to 14, as that would make the code return 10.',
             "answer", {},
             "suggestion": {"a": 10}
            }
            """
        )

        self.include_example = include_example
        self.max_tokens = max_tokens
        self.log = [] if log else None
        self.summary_log = [] if log else None
        self.memory = FIFOBuffer(memory_size)
        self.prompt_symbols = copy.deepcopy(self.default_prompt_symbols)
        if prompt_symbols is not None:
            self.prompt_symbols.update(prompt_symbols)
        if json_keys is not None:
            self.default_json_keys.update(json_keys)
        # if self.default_json_keys['answer'] is None:
        #     del self.default_json_keys['answer']
        # NOTE del cause KeyError if the key is not in the dict due to changing class attribute
        if 'answer' not in self.default_json_keys or self.default_json_keys['answer'] is None:  # answer field is not needed
            # If 'answer' is not in the json keys, we use the no-answer format
            self.output_format_prompt = self.output_format_prompt_no_answer.format(**self.default_json_keys)
        else:  # If 'answer' is in the json keys, we use the original format of OptoPrime
            self.output_format_prompt = self.output_format_prompt_original.format(**self.default_json_keys)
        self.use_json_object_format = use_json_object_format
        self.highlight_variables = highlight_variables

    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        return GraphPropagator()

    def summarize(self):
        """Aggregate feedback from parameters into a structured summary.
        
        Collects and organizes feedback from all trainable parameters into
        a FunctionFeedback structure suitable for problem representation.
        
        Returns
        -------
        FunctionFeedback
            Structured feedback containing:
            - variables: Trainable parameters with values and descriptions
            - inputs: Non-trainable root nodes
            - graph: Topologically sorted function calls
            - others: Intermediate computation values
            - output: Final output values
            - documentation: Function documentation strings
            - user_feedback: Aggregated user feedback
        
        Notes
        -----
        The method performs several transformations:
        1. Aggregates feedback from all trainable parameters
        2. Converts the trace graph to FunctionFeedback structure
        3. Separates root nodes into variables (trainable) and inputs (non-trainable)
        4. Preserves the computation graph and intermediate values
        
        Parameters without feedback (disconnected from output) are still
        included in the summary but may not receive updates.
        """
        # Aggregate feedback from all the parameters
        feedbacks = [
            self.propagator.aggregate(node.feedback)
            for node in self.parameters
            if node.trainable
        ]
        summary = sum(feedbacks)  # TraceGraph
        # Construct variables and update others
        # Some trainable nodes might not receive feedback, because they might not be connected to the output
        summary = node_to_function_feedback(summary)
        # Classify the root nodes into variables and others
        # summary.variables = {p.py_name: p.data for p in self.parameters if p.trainable and p.py_name in summary.roots}

        trainable_param_dict = {p.py_name: p for p in self.parameters if p.trainable}
        summary.variables = {
            py_name: data
            for py_name, data in summary.roots.items()
            if py_name in trainable_param_dict
        }
        summary.inputs = {
            py_name: data
            for py_name, data in summary.roots.items()
            if py_name not in trainable_param_dict
        }  # non-variable roots

        return summary

    @staticmethod
    def repr_node_value(node_dict):
        """Format node values for display.

        Parameters
        ----------
        node_dict : dict
            Dictionary of node names to (value, description) tuples.

        Returns
        -------
        str
            Formatted string with type and value for each node.
        """
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                temp_list.append(f"({type(v[0]).__name__}) {k}={v[0]}")
            else:
                temp_list.append(f"(code) {k}:{v[0]}")
        return "\n".join(temp_list)

    @staticmethod
    def repr_node_constraint(node_dict):
        """Format node constraints for display.
        
        Parameters
        ----------
        node_dict : dict
            Dictionary of node names to (value, description) tuples.
        
        Returns
        -------
        str
            Formatted string with type and constraint for each node.
            Only includes nodes with non-None descriptions.
        """
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                if v[1] is not None:
                    temp_list.append(f"({type(v[0]).__name__}) {k}: {v[1]}")
            else:
                if v[1] is not None:
                    temp_list.append(f"(code) {k}: {v[1]}")
        return "\n".join(temp_list)

    def problem_instance(self, summary, mask=None):
        """Create a ProblemInstance from aggregated feedback.
        
        Converts a FunctionFeedback summary into a formatted problem
        representation for the language model.
        
        Parameters
        ----------
        summary : FunctionFeedback
            Aggregated feedback from summarize() method.
        mask : list[str], optional
            List of sections to exclude from the problem instance.
            Can include: "#Instruction", "#Code", "#Variables", etc.
        
        Returns
        -------
        ProblemInstance
            Structured problem representation with all sections
            formatted for language model consumption.
        
        Notes
        -----
        The mask parameter allows selective inclusion of problem
        components, useful for ablation studies or focused optimization.
        """
        mask = mask or []
        return ProblemInstance(
            instruction=self.objective if "#Instruction" not in mask else "",
            code=(
                "\n".join([v for k, v in sorted(summary.graph)])
                if "#Code" not in mask
                else ""
            ),
            documentation=(
                "\n".join([f"[{k}] {v}" for k, v in summary.documentation.items()])
                if "#Documentation" not in mask
                else ""
            ),
            variables=(
                self.repr_node_value(summary.variables)
                if "#Variables" not in mask
                else ""
            ),
            constraints=(
                self.repr_node_constraint(summary.variables)
                if "#Constraints" not in mask
                else ""
            ),
            inputs=(
                self.repr_node_value(summary.inputs) if "#Inputs" not in mask else ""
            ),
            outputs=(
                self.repr_node_value(summary.output) if "#Outputs" not in mask else ""
            ),
            others=(
                self.repr_node_value(summary.others) if "#Others" not in mask else ""
            ),
            feedback=summary.user_feedback if "#Feedback" not in mask else "",
        )

    def construct_prompt(self, summary, mask=None, *args, **kwargs):
        """Construct the system and user prompt."""
        system_prompt = (
            self.representation_prompt + self.output_format_prompt
        )  # generic representation + output rule
        user_prompt = self.user_prompt_template.format(
            problem_instance=str(self.problem_instance(summary, mask=mask))
        )  # problem instance
        if self.include_example:
            user_prompt = (
                self.example_problem_template.format(
                    example_problem=self.example_problem,
                    example_response=self.example_response,
                )
                + user_prompt
            )


        if self.highlight_variables:
            var_names = []
            for k, v in summary.variables.items():
                var_names.append(f"{k}")  # ({type(v[0]).__name__})
            var_names = ", ".join(var_names)

            user_prompt += self.final_prompt_with_variables.format(names=var_names)
        else:  # This is the original OptoPrime prompt
            user_prompt += self.final_prompt

        # Add examples
        if len(self.memory) > 0:
            prefix = user_prompt.split(self.final_prompt)[0]
            examples = []
            for variables, feedback in self.memory:
                examples.append(
                    json.dumps(
                        {
                            "variables": {k: v[0] for k, v in variables.items()},
                            "feedback": feedback,
                        },
                        indent=4,
                    )
                )
            examples = "\n".join(examples)
            user_prompt = (
                prefix
                + f"\nBelow are some variables and their feedbacks you received in the past.\n\n{examples}\n\n"
                + self.final_prompt
            )
        self.memory.add((summary.variables, summary.user_feedback))

        return system_prompt, user_prompt

    def replace_symbols(self, text: str, symbols: Dict[str, str]) -> str:
        for k, v in symbols.items():
            text = text.replace(self.default_prompt_symbols[k], v)
        return text

    def _step(
        self, verbose=False, mask=None, *args, **kwargs
    ) -> Dict[ParameterNode, Any]:
        assert isinstance(self.propagator, GraphPropagator)
        summary = self.summarize()
        system_prompt, user_prompt = self.construct_prompt(summary, mask=mask)

        system_prompt = self.replace_symbols(system_prompt, self.prompt_symbols)
        user_prompt = self.replace_symbols(user_prompt, self.prompt_symbols)

        response = self.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                verbose=verbose,
                max_tokens=self.max_tokens,
            )
            
        if "TERMINATE" in response:
            return {}

        suggestion = self.extract_llm_suggestion(response)
        update_dict = self.construct_update_dict(suggestion)

        if self.log is not None:
            self.log.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response,
                }
            )
            self.summary_log.append(
                {"problem_instance": self.problem_instance(summary), "summary": summary}
            )

        return update_dict

    def construct_update_dict(
        self, suggestion: Dict[str, Any]
    ) -> Dict[ParameterNode, Any]:
        """Convert the suggestion in text into the right data type."""
        # TODO: might need some automatic type conversion
        update_dict = {}
        for node in self.parameters:
            if node.trainable and node.py_name in suggestion:
                try:
                    formatted_suggestion = suggestion[node.py_name]
                    update_dict[node] = type(node.data)(formatted_suggestion)
                except (ValueError, KeyError) as e:
                    # catch error due to suggestion missing the key or wrong data type
                    if self.ignore_extraction_error:
                        warnings.warn(
                            f"Cannot convert the suggestion '{suggestion[node.py_name]}' for {node.py_name} to the right data type"
                        )
                    else:
                        raise e
        return update_dict

    def extract_llm_suggestion(self, response: str):
        """Extract the suggestion from the response."""
        suggestion_tag = self.default_json_keys["suggestion"]

        suggestion = {}
        attempt_n = 0
        while attempt_n < 2:
            try:
                suggestion = json.loads(response)[suggestion_tag]
                break
            except json.JSONDecodeError:
                # Remove things outside the brackets
                response = re.findall(r"{.*}", response, re.DOTALL)
                if len(response) > 0:
                    response = response[0]
                attempt_n += 1
            except Exception:
                attempt_n += 1

        if not isinstance(suggestion, dict):
            suggestion = {}

        if len(suggestion) == 0:
            # we try to extract key/value separately and return it as a dictionary
            pattern = rf'"{suggestion_tag}"\s*:\s*\{{(.*?)\}}'
            suggestion_match = re.search(pattern, str(response), re.DOTALL)
            if suggestion_match:
                suggestion = {}
                # Extract the entire content of the suggestion dictionary
                suggestion_content = suggestion_match.group(1)
                # Regex to extract each key-value pair;
                # This scheme assumes double quotes but is robust to missing commas at the end of the line
                pair_pattern = r'"([a-zA-Z0-9_]+)"\s*:\s*"(.*)"'
                # Find all matches of key-value pairs
                pairs = re.findall(pair_pattern, suggestion_content, re.DOTALL)
                for key, value in pairs:
                    suggestion[key] = value

        if len(suggestion) == 0:
            if not self.ignore_extraction_error:
                print(f"Cannot extract {self.default_json_keys['suggestion']} from LLM's response:")
                print(response)

        # if the suggested value is a code, and the entire code body is empty (i.e., not even function signature is present)
        # then we remove such suggestion
        keys_to_remove = []
        for key, value in suggestion.items():
            if "__code" in key and value.strip() == "":
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del suggestion[key]

        return suggestion

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        verbose: Union[bool, str] = False,
        max_tokens: int = 4096,
    ):
        """Call the LLM with a prompt and return the response."""
        if verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format =  {"type": "json_object"} if self.use_json_object_format else None
        try:  # Try tp force it to be a json object
            response = self.llm(messages=messages, max_tokens=max_tokens, response_format=response_format)
        except Exception:
            response = self.llm(messages=messages, max_tokens=max_tokens)

        response = response.choices[0].message.content

        if verbose:
            print("LLM response:\n", response)
        return response