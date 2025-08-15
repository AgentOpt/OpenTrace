import json
from textwrap import dedent
from dataclasses import dataclass, asdict
from typing import Dict

from opto.optimizers.optoprime_v2 import OptoPrimeV2, OptimizerPromptSymbolSet

"""
OPRO is a single parameter / solution optimizer that conditions on feedback.
(context, solution, feedback) -> new_solution

It does not contain execution graph and is more streamlined/faster in inference.
"""


# Not inheriting from optoprime_v2 because this should have a smaller set
class OPROPromptSymbolSet(OptimizerPromptSymbolSet):

    problem_context_section_title = "# Problem Context"
    variable_section_title = "# Solution"
    feedback_section_title = "# Feedback"

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
        }

@dataclass
class ProblemInstance:
    instruction: str
    variables: str
    feedback: str

    optimizer_prompt_symbol_set: OPROPromptSymbolSet

    problem_template = dedent(
        """
        # Problem Context
        {instruction}

        # Solution
        {variables}

        # Feedback
        {feedback}
        """
    )

    def __repr__(self) -> str:
        return self.replace_symbols(self.problem_template.format(
            instruction=self.instruction,
            variables=self.variables,
            feedback=self.feedback,
        ), self.optimizer_prompt_symbol_set.default_prompt_symbols)

    def replace_symbols(self, text: str, symbols: Dict[str, str]) -> str:
        default_prompt_symbols = {
            "variables": "# Variables",
            "feedback": "# Feedback",
            "instruction": "# Problem Context",
        }

        for k, v in symbols.items():
            text = text.replace(default_prompt_symbols[k], v)
        return text

"""
TODO:
1. think about how initial solution was generated...
"""

class OPROv2(OptoPrimeV2):
    representation_prompt = dedent(
        """
        You're tasked to change the proposed solution according to feedback.

        Specifically, a problem will be composed of the following parts:
        - {instruction_section_title}: the instruction which describes the things you need to do or the question you should answer.
        - {variables_section_title}: the proposed solution that you can change/tweak (trainable).
        - {feedback_section_title}: the feedback about the solution.

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
                 **kwargs):
        optimizer_prompt_symbol_set = optimizer_prompt_symbol_set or OPROPromptSymbolSet()
        super().__init__(*args, optimizer_prompt_symbol_set=optimizer_prompt_symbol_set, **kwargs)
        self.include_example = False # default example in OptoPrimeV2 does not work in OPRO
        self.memory_size = 5

    def problem_instance(self, summary, mask=None):
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
        )
        self.output_format_prompt = self.output_format_prompt_template.format(
            output_format=self.optimizer_prompt_symbol_set.output_format,
            reasoning_tag=self.optimizer_prompt_symbol_set.reasoning_tag,
            improved_variable_tag=self.optimizer_prompt_symbol_set.improved_variable_tag,
            instruction_section_title=self.optimizer_prompt_symbol_set.instruction_section_title.replace(" ", ""),
            feedback_section_title=self.optimizer_prompt_symbol_set.feedback_section_title.replace(" ", ""),
            variables_section_title=self.optimizer_prompt_symbol_set.variables_section_title.replace(" ", ""),
        )
