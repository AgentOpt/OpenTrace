import json
from textwrap import dedent
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from opto.optimizers.optoprime_v2 import OptoPrimeV2, OptimizerPromptSymbolSet

# Not inheriting from optoprime_v2 because this should have a smaller set
class OPROPromptSymbolSet(OptimizerPromptSymbolSet):
    """Prompt symbol set for OPRO optimizer.

    This class defines the tags and symbols used in the OPRO optimizer's prompts
    and output parsing. It provides a structured way to format problems and parse
    responses from the language model.

    Attributes
    ----------
    instruction_section_title : str
        Title for the instruction section in prompts.
    variable_section_title : str
        Title for the variable/solution section in prompts.
    feedback_section_title : str
        Title for the feedback section in prompts.
    node_tag : str
        Tag used to identify constant nodes in the computation graph.
    variable_tag : str
        Tag used to identify variable nodes that can be optimized.
    value_tag : str
        Tag used to wrap the value of a node.
    constraint_tag : str
        Tag used to wrap constraint expressions for nodes.
    reasoning_tag : str
        Tag used to wrap reasoning in the output.
    improved_variable_tag : str
        Tag used to wrap improved variable values in the output.
    name_tag : str
        Tag used to wrap variable names.
    expect_json : bool
        Whether to expect JSON output format (default: False).

    Methods
    -------
    default_prompt_symbols
        Returns default prompt symbols dictionary.

    Notes
    -----
    This class inherits from OptimizerPromptSymbolSet but defines a smaller,
    more focused set of symbols specifically for OPRO optimization.
    """

    instruction_section_title = "# Instruction"
    variable_section_title = "# Solution"
    feedback_section_title = "# Feedback"
    context_section_title = "# Context"

    node_tag = "node"  # nodes that are constants in the graph
    variable_tag = "solution"  # nodes that can be changed
    value_tag = "value"  # inside node, we have value tag
    constraint_tag = "constraint"  # inside node, we have constraint tag

    # output format
    # Note: we currently don't support extracting format's like "```code```" because we assume supplied tag is name-only, i.e., <tag_name></tag_name>
    reasoning_tag = "reasoning"
    improved_variable_tag = "variable"
    name_tag = "name"

    expect_json = False  # this will stop `enforce_json` arguments passed to LLM calls

    @property
    def default_prompt_symbols(self) -> Dict[str, str]:
        return {
            "variables": self.variables_section_title,
            "feedback": self.feedback_section_title,
            "instruction": self.instruction_section_title,
            "context": self.context_section_title
        }

@dataclass
class ProblemInstance:
    """Represents a problem instance for OPRO optimization.

    This dataclass encapsulates a complete problem instance including the
    instruction, current variables/solution, and feedback received.

    Attributes
    ----------
    instruction : str
        The instruction describing what needs to be done or the question to answer.
    variables : str
        The current proposed solution that can be modified.
    feedback : str
        Feedback about the current solution.
    context: str
        Optional context information that might be useful to solve the problem.

    optimizer_prompt_symbol_set : OPROPromptSymbolSet
        The symbol set used for formatting the problem.
    problem_template : str
        Template for formatting the problem instance as a string.

    Methods
    -------
    __repr__()
        Returns a formatted string representation of the problem instance.

    Notes
    -----
    The problem instance is formatted using the problem_template which
    organizes the instruction, variables, and feedback into a structured format.
    """
    instruction: str
    variables: str
    feedback: str
    context: Optional[str]

    optimizer_prompt_symbol_set: OPROPromptSymbolSet

    problem_template = dedent(
        """
        # Instruction
        {instruction}

        # Solution
        {variables}

        # Feedback
        {feedback}
        """
    )

    def __repr__(self) -> str:
        optimization_query = self.problem_template.format(
            instruction=self.instruction,
            variables=self.variables,
            feedback=self.feedback,
        )

        context_section = dedent("""

               # Context
               {context}
               """)

        if self.context is not None and self.context.strip() != "":
            context_section.format(context=self.context)
            optimization_query += context_section

        return optimization_query

class OPROv2(OptoPrimeV2):
    """OPRO (Optimization by PROmpting) optimizer version 2.

    OPRO is an optimization algorithm that leverages large language models to
    iteratively improve solutions based on feedback. It treats optimization as
    a natural language problem where the LLM proposes improvements to variables
    based on instruction and feedback.

    Parameters
    ----------
    *args
        Variable length argument list passed to parent class.
    optimizer_prompt_symbol_set : OptimizerPromptSymbolSet, optional
        The symbol set for formatting prompts and parsing outputs.
        Defaults to OPROPromptSymbolSet().
    include_example : bool, optional
        Whether to include examples in the prompt. Default is False as
        the default example in OptoPrimeV2 does not work well with OPRO.
    memory_size : int, optional
        Number of past optimization steps to remember. Default is 5.
    **kwargs
        Additional keyword arguments passed to parent class.

    Attributes
    ----------
    representation_prompt : str
        Template for explaining the problem representation to the LLM.
    output_format_prompt_template : str
        Template for specifying the expected output format.
    user_prompt_template : str
        Template for presenting the problem instance to the LLM.
    final_prompt : str
        Template for requesting the final revised solutions.
    default_objective : str
        Default objective when none is specified.

    Methods
    -------
    problem_instance(summary, mask=None)
        Creates a ProblemInstance from an optimization summary.
    initialize_prompt()
        Initializes and formats the prompt templates.

    Notes
    -----
    OPRO differs from OptoPrime by focusing on simpler problem representations
    and clearer feedback incorporation. It is particularly effective for
    problems where the optimization can be expressed in natural language.

    See Also
    --------
    OptoPrimeV2 : Parent class providing core optimization functionality.
    OPROPromptSymbolSet : Symbol set used for formatting.

    Examples
    --------
    >>> optimizer = OPROv2(memory_size=10)
    >>> # Use optimizer to improve solutions based on feedback
    """
    representation_prompt = dedent(
        """
        You're tasked to change the proposed solution according to feedback.

        Specifically, a problem will be composed of the following parts:
        - {instruction_section_title}: the instruction which describes the things you need to do or the question you should answer.
        - {variables_section_title}: the proposed solution that you can change/tweak (trainable).
        - {feedback_section_title}: the feedback about the solution.
        - {context_section_title}: the context information that might be useful to solve the problem.

        If `data_type` is `code`, it means `{value_tag}` is the source code of a python code, which may include docstring and definitions.
        """
    )

    output_format_prompt_template = dedent(
        """
        Output_format: Your output should be in the following XML/HTML format:

        ```
        {output_format}
        ```

        In <{reasoning_tag}>, explain the problem: 1. what the {instruction_section_title} means 2. what the {feedback_section_title} means to {variables_section_title} considering how {variables_section_title} follow {instruction_section_title}. 3. Reasoning about the suggested changes in {variables_section_title} (if needed) and the expected result.

        If you need to suggest a change in the values of {variables_section_title}, write down the suggested values in <{improved_variable_tag}>. Remember you can change only the values in {variables_section_title}, not others. When `type` of a variable is `code`, you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature.

        If no changes are needed, just output TERMINATE.
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

    context_prompt = dedent(
        """
        Here is some additional **context** to solving this problem:

        {context}
        """
    )

    final_prompt = dedent(
        """
        What are your revised solutions on {names}?

        Your response:
        """
    )

    # Default Objective becomes instruction for the next block
    default_objective = "Propose a new solution that will incorporate the feedback."

    def __init__(self, *args,
                 optimizer_prompt_symbol_set: OptimizerPromptSymbolSet = None,
                 include_example=False, # default example in OptoPrimeV2 does not work in OPRO
                 memory_size=5,
                 problem_context: Optional[str] = None,
                 **kwargs):
        """Initialize the OPROv2 optimizer.

        Parameters
        ----------
        *args
            Variable length argument list passed to parent class.
        optimizer_prompt_symbol_set : OptimizerPromptSymbolSet, optional
            The symbol set for formatting prompts and parsing outputs.
            If None, uses OPROPromptSymbolSet().
        include_example : bool, optional
            Whether to include examples in the prompt. Default is False.
        memory_size : int, optional
            Number of past optimization steps to remember. Default is 5.
        **kwargs
            Additional keyword arguments passed to parent class.
        """
        optimizer_prompt_symbol_set = optimizer_prompt_symbol_set or OPROPromptSymbolSet()
        super().__init__(*args, optimizer_prompt_symbol_set=optimizer_prompt_symbol_set,
                         include_example=include_example, memory_size=memory_size,
                         problem_context=problem_context,
                         **kwargs)

    def problem_instance(self, summary, mask=None):
        """Create a ProblemInstance from an optimization summary.

        Parameters
        ----------
        summary : object
            The optimization summary containing variables and feedback.
        mask : list, optional
            List of sections to mask/hide in the problem instance.
            Can include "#Instruction", variable section title, or feedback section title.

        Returns
        -------
        ProblemInstance
            A formatted problem instance ready for presentation to the LLM.

        Notes
        -----
        The mask parameter allows selective hiding of problem components,
        useful for ablation studies or specific optimization strategies.
        """
        mask = mask or []
        return ProblemInstance(
            instruction=self.objective if "#Instruction" not in mask else "",
            variables=(
                self.repr_node_value_compact(summary.variables, node_tag=self.optimizer_prompt_symbol_set.variable_tag,
                                             value_tag=self.optimizer_prompt_symbol_set.value_tag,
                                             constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag)
                if self.optimizer_prompt_symbol_set.variables_section_title not in mask
                else ""
            ),
            feedback=summary.user_feedback if self.optimizer_prompt_symbol_set.feedback_section_title not in mask else "",
            optimizer_prompt_symbol_set=self.optimizer_prompt_symbol_set
        )

    def initialize_prompt(self):
        """Initialize and format the prompt templates.

        This method formats the representation_prompt and output_format_prompt
        templates with the appropriate symbols from the optimizer_prompt_symbol_set.
        It prepares the prompts for use in optimization.

        Notes
        -----
        This method should be called during initialization to ensure all
        prompt templates are properly formatted with the correct tags and symbols.
        """
        self.representation_prompt = self.representation_prompt.format(
            variable_expression_format=dedent(f"""
            <{self.optimizer_prompt_symbol_set.variable_tag} name="variable_name" type="data_type">
            <{self.optimizer_prompt_symbol_set.value_tag}>
            value
            </{self.optimizer_prompt_symbol_set.value_tag}>
            <{self.optimizer_prompt_symbol_set.constraint_tag}>
            constraint_expression
            </{self.optimizer_prompt_symbol_set.constraint_tag}>
            </{self.optimizer_prompt_symbol_set.variable_tag}>
        """),
            value_tag=self.optimizer_prompt_symbol_set.value_tag,
            variables_section_title=self.optimizer_prompt_symbol_set.variables_section_title.replace(" ", ""),
            feedback_section_title=self.optimizer_prompt_symbol_set.feedback_section_title.replace(" ", ""),
            instruction_section_title=self.optimizer_prompt_symbol_set.instruction_section_title.replace(" ", ""),
            context_section_title=self.optimizer_prompt_symbol_set.context_section_title.replace(" ", "")
        )
        self.output_format_prompt = self.output_format_prompt_template.format(
            output_format=self.optimizer_prompt_symbol_set.output_format,
            reasoning_tag=self.optimizer_prompt_symbol_set.reasoning_tag,
            improved_variable_tag=self.optimizer_prompt_symbol_set.improved_variable_tag,
            instruction_section_title=self.optimizer_prompt_symbol_set.instruction_section_title.replace(" ", ""),
            feedback_section_title=self.optimizer_prompt_symbol_set.feedback_section_title.replace(" ", ""),
            variables_section_title=self.optimizer_prompt_symbol_set.variables_section_title.replace(" ", ""),
            context_section_title=self.optimizer_prompt_symbol_set.context_section_title.replace(" ", "")
        )
