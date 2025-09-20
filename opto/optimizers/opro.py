import json
from textwrap import dedent

from opto.optimizers.optoprime import OptoPrime


class OPRO(OptoPrime):
    """Optimization by PROmpting (OPRO) optimizer implementing feedback-driven parameter updates.
    
    OPRO is a simplified version of OptoPrime that uses accumulated examples of variables
    and their feedback to guide parameter optimization. It maintains a buffer of historical
    examples to provide context for generating improved parameter suggestions.

    Parameters
    ----------
    *args
        Positional arguments passed to the OptoPrime parent class.
    **kwargs
        Keyword arguments passed to the OptoPrime parent class.

    Attributes
    ----------
    buffer : list[tuple]
        Buffer storing (variables, feedback) pairs for historical context.
    user_prompt_template : str
        Template for constructing user prompts with examples and instructions.
    output_format_prompt : str
        Template specifying the expected JSON output format for suggestions.
    default_objective : str
        Default optimization objective when none is specified.

    Notes
    -----
    OPRO differs from OptoPrime by using a simpler prompt structure focused on
    historical examples rather than detailed meta-information and reasoning chains.
    The optimizer accumulates variable-feedback pairs over time to build context
    for future optimization steps.
    """
    user_prompt_template = dedent(
        """
        Below are some example variables and their feedbacks.

        {examples}

        ================================

        {instruction}
        """
    )

    output_format_prompt = dedent(
        """
        Output_format: Your output should be in the following json format, satisfying
        the json syntax:

        {{
        "suggestion": {{
            <variable_1>: <suggested_value_1>,
            <variable_2>: <suggested_value_2>,
        }}
        }}

        When suggestion variables, write down the suggested values in "suggestion".
        When <type> of a variable is (code), you should write the new definition in the
        format of python code without syntax errors, and you should not change the
        function name or the function signature.

        If no changes or answer are needed, just output TERMINATE.
        """
    )

    default_objective = "Come up with a new variable in accordance to feedback."

    def __init__(self, *args, **kwargs):
        """Initialize OPRO optimizer with empty example buffer.

        Parameters
        ----------
        *args
            Positional arguments passed to OptoPrime parent class.
        **kwargs
            Keyword arguments passed to OptoPrime parent class.
        """
        super().__init__(*args, **kwargs)
        self.buffer = []

    def construct_prompt(self, summary, mask=None, *args, **kwargs):
        """Construct system and user prompts using historical examples.

        This method builds prompts by accumulating variable-feedback pairs in a buffer
        and formatting them as examples for the language model to learn from.

        Parameters
        ----------
        summary : Summary
            Summary object containing current variables and user feedback.
        mask : Any, optional
            Mask parameter (unused in OPRO), by default None.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tuple[str, str]
            System prompt (output format) and user prompt with examples.

        Notes
        -----
        The method adds the current summary to the buffer and formats all buffered
        examples into a structured prompt. Each example includes the variables
        dictionary and associated feedback.
        """
        self.buffer.append((summary.variables, summary.user_feedback))

        examples = []
        for variables, feedback in self.buffer:
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

        user_prompt = self.user_prompt_template.format(
            examples=examples, instruction=self.objective
        )
        return self.output_format_prompt, user_prompt
