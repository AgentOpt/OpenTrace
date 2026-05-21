from opto.optimizers.utils import print_color
from opto.utils.llm import LLM # For the selector LLM
import random
import re


def get_trajectory_of_one_rollout(rollout):
    """
    Convert a rollout into a structured markdown trajectory for optimization.

    This function extracts the trainable parameters and formats the trajectory 
    to guide the optimizer in improving the module's performance.

    Parameters
    ----------
    rollout : dict
        A rollout dictionary containing:
        - 'module': trace.Module - the agent module with trainable parameters
        - 'x': Any - the input data
        - 'info': Any - additional information about the input
        - 'target': Any - the generated output
        - 'score': float - evaluation score (0 = failed, 1 = success)
        - 'feedback': Any - detailed feedback from the evaluation

    Returns
    -------
    str
        A markdown-formatted trajectory string for optimizer guidance.
    """
    assert rollout['module'] is not None, "rollout['module'] is None."
    assert rollout['x'] is not None, "rollout['x'] is None."
    assert rollout['target'] is not None, "rollout['target'] is None."
    assert rollout['score'] is not None, "rollout['score'] is None."
    assert rollout['feedback'] is not None, "rollout['feedback'] is None."
    
    # Extract trainable parameters
    parameters = rollout['module'].parameters()
    parameters_dict = {p.py_name: p.data for p in parameters}
    
    # In multi-objective mode, rollouts carry a 'score_dict' with per-metric scores
    # (e.g. {"accuracy": 0.9, "latency": 0.2}) populated by validate() in PrioritySearch.
    # We render the full breakdown so the summarizer LLM can analyze trade-offs across
    # objectives, rather than seeing only the aggregate scalar score.
    # When score_dict is absent (single-objective mode), we fall back to scalar-only display.
    score_dict = rollout.get('score_dict')
    if isinstance(score_dict, dict) and score_dict:
        breakdown = "\n".join(f"  - {k}: {v}" for k, v in score_dict.items())
        result_section = (
            f"- **Overall Score:** {rollout['score']}\n"
            f"- **Score Breakdown:**\n{breakdown}\n"
            f"- **Feedback:** {rollout['feedback']}"
        )
    else:
        result_section = (
            f"- **Score:** {rollout['score']}\n"
            f"- **Feedback:** {rollout['feedback']}"
        )

    trajectory = f"""## Task Trajectory

## Module Parameters
{parameters_dict}

## Input 
{rollout['x']}

## Output
{rollout['target']}

## Result
{result_section}"""
    return trajectory

class Summarizer:
    """A class which use LLM to summarize the trajectories of the memory. It should be able to learn the patterns of the trajectories. Generate a summary to guide the optimizer to generate better candidates.
    """
    DEFAULT_SYSTEM_PROMPT = "You are an expert at analyzing agent behavior patterns and providing actionable guidance for parameter optimization."

    DEFAULT_USER_PROMPT_TEMPLATE = """Analyze the following agent conversation trajectories and extract insights for optimization.

        Current Summary (from previous analysis):
        {current_summary}

        New Trajectories to Analyze:
        {history_trajectories}

        Instructions:
        - Review both the Current Summary and the New Trajectories
        - Synthesize ALL insights into a single, cohesive summary
        - Integrate new patterns with existing knowledge
        - Reorganize and consolidate information as needed for clarity
        - DO NOT use incremental language like "[Previous points remain valid, plus:]"
        - Generate a complete, standalone summary that incorporates everything

        Provide your analysis in XML format:
        <reasoning>
        Analyze the key patterns and strategies that led to success or failure in these trajectories. Consider both the current summary and new trajectories.
        </reasoning>
        <summary>
        A complete, consolidated summary with concrete recommendations for generating better results. This should be a standalone summary that integrates insights from both the current summary and new trajectories, without using incremental modification language.
        </summary>"""

    def __init__(self, verbose: bool = False, success_threshold: float = 0,
                 max_candidates_in_prompt: int = 5,
                 current_summary: str = "Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories: ",
                 system_prompt: str = None,
                 user_prompt_template: str = None):
        self.llm = LLM() # use the default model
        self.max_candidates_in_prompt = max_candidates_in_prompt
        self.current_summary = current_summary
        self.used_candidates = set()  # Track candidates that have been summarized
        self.verbose = verbose
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
        # Configurable threshold for classifying rollouts as successful (score > threshold)
        # or failed (score <= threshold). Defaults to 0 for backward compatibility.
        # Previously hardcoded as 0, which also caused rollouts with negative scores to be
        # missed by both the success and failure lists.
        self.success_threshold = success_threshold

    def _get_trajectories_for_memory(self, memory):
        """
        Get trajectories for the memory. Memory is a list of (neg_score, candidate) tuples.
        We first collect rollouts from the each candidate, and then get the trajectories for each rollout.

        Return one single string of all trajectories.
        """
        trajectories = []
        if self.verbose:
            print_color(f"Getting trajectories from {len(memory)} candidates.", "blue")
        # Filter out candidates that have already been used and have rollouts
        memory_with_rollouts = [(neg_score, candidate) for neg_score, candidate in memory
                                if len([rollout for rollout in candidate.rollouts if rollout['score'] is not None]) > 0
                                and id(candidate) not in self.used_candidates]
        if self.verbose:
            print_color(f"Memory (unseen candidates) with rollouts: {len(memory_with_rollouts)}", "blue")
        # Sample 5 candidates (or fewer if not enough available)
        num_to_sample = min(5, len(memory_with_rollouts))
        temporary_memory = random.sample(memory_with_rollouts, k=num_to_sample)
        # Mark sampled candidates as used
        for _, candidate in temporary_memory:
            self.used_candidates.add(id(candidate))
        for _, candidate in temporary_memory:
            rollouts = [rollout for rollout in candidate.rollouts if rollout['score'] is not None]
            if len(rollouts) == 0:
                continue
            # For each candidate, add one (if exists) successful_rollout and one (if exists) failed_rollout.
            candidate_update_dict = candidate.update_dict.values()
            prompt = f"Candidate pamameters: {candidate_update_dict}."
            successful_rollouts = [rollout for rollout in rollouts if rollout['score'] > self.success_threshold]
            failed_rollouts = [rollout for rollout in rollouts if rollout['score'] <= self.success_threshold]
            if len(successful_rollouts) > 0: 
                random_successful_rollout = random.choice(successful_rollouts)
                prompt += f"\nSuccessful trajectory: {get_trajectory_of_one_rollout(random_successful_rollout)}."
            if len(failed_rollouts) > 0:
                random_failed_rollout = random.choice(failed_rollouts)
                prompt += f"\nFailed trajectory: {get_trajectory_of_one_rollout(random_failed_rollout)}."
            
            trajectories.append(prompt)
        if self.verbose:
            print_color(f"Generated trajectories from {len(trajectories)} candidates.", "green")
        
        return '\n'.join(trajectories)

    def summarize(self, memory) -> str:
        """Summarize the trajectories using the LLM.
        Args:
            memory: The memory containing trajectories to summarize.
        Returns:
            str: The summary.
        """

        history_trajectories = self._get_trajectories_for_memory(memory)
        if len(history_trajectories) == 0:
            return "No trajectories found for the memory."
        
        user_prompt = self.user_prompt_template.format(
            current_summary=self.current_summary,
            history_trajectories=history_trajectories,
        )

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        
        response = self.llm(messages=prompt_messages)
        response = response.choices[0].message.content
        # Extract summary using XML regex
        summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)

        self.current_summary = summary_match.group(1).strip()

        return self.current_summary
