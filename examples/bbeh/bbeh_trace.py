# Compare Minibatch vs. Non-MiniBatch vs. DSPy vs. TextGrad
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["TRACE_CUSTOMLLM_MODEL"] = os.getenv("TRACE_CUSTOMLLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
os.environ["TRACE_CUSTOMLLM_URL"] = os.getenv("TRACE_CUSTOMLLM_URL")
os.environ["TRACE_CUSTOMLLM_API_KEY"] = os.getenv("TRACE_CUSTOMLLM_API_KEY")
os.environ["TRACE_DEFAULT_LLM_BACKEND"] = os.getenv("TRACE_DEFAULT_LLM_BACKEND", "CustomLLM")

from opto.trace.nodes import node, GRAPH, ParameterNode
from textwrap import dedent
from opto.optimizers import OptoPrimeV2
from datasets import load_dataset
from opto.trace import model, bundle
import opto.trace.operators as trace_ops
import numpy as np
from tqdm import tqdm
from opto.trainer.examples.basic_algorithms import MinibatchAlgorithm, evaluate
from opto.trainer.guide import Guide


def eval_metric(true, prediction):
    # two types of answers:
    # (A)/(B) or "syndrome therefrom"/8/No/invalid
    matches = re.findall(r"\([A-Z]\)", true)
    if matches:
        pred = prediction
        matches = re.findall(r"\([A-Z]\)", pred)
        parsed_answer = matches[-1] if matches else ""
        return parsed_answer == true
    else:
        # substring match
        return prediction == true


class BigBenchGuide(Guide):
    """
    Custom guide that uses the eval_metric function to evaluate responses
    and provide feedback for the BigBench tasks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, task, response, info, **kwargs):
        """
        Evaluate the response using the eval_metric function.

        Args:
            task: The question
            response: The model's answer
            info: The correct answer

        Returns:
            score: 1.0 if correct, 0.0 if incorrect
            feedback: Feedback message
        """
        try:
            correctness = eval_metric(info, response)
            score = 1.0 if correctness else 0.0

            if correctness:
                feedback = "The answer is correct! No need to change anything."
            else:
                feedback = f"The answer is wrong. We expect the output of your answer to be \"{info}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."

            return score, feedback
        except Exception as e:
            return 0.0, f"Error occurred: {str(e)}. Please fix the error and try again."

    def metric(self, task, response, info, **kwargs):
        """
        Evaluate the response and return just the score.

        Args:
            task: The question
            response: The model's answer
            info: The correct answer

        Returns:
            score: 1.0 if correct, 0.0 if incorrect
        """
        score, _ = self.forward(task, response, info, **kwargs)
        return score


@model
class Predict:
    def __init__(self):
        super().__init__()

        self.demos = []
        self.prompt_template = dedent(
            """
            Given the fields `question`, produce the fields `answer`.

            ---

            Follow the following format.

            Question: 
            Answer: 

            ---
            Question:
            """
        )

        self.prompt_template = ParameterNode(self.prompt_template, trainable=True,
                                             description="[ParameterNode] This is the Prompt Template to the LLM. " + \
                                                         "Need to include information about what the format of answers LLM should output. " + \
                                                         "They can be (A)/(B), a number like 8, or a string, or Yes/No/Ambiguous.")

    @bundle(trainable=True, allow_external_dependencies=True)
    def extract_answer(self, prompt_template, question, response):
        """
        Need to read in the response, which can contain additional thought, delibration and an answer.
        Use code to process the response and find where the answer is.
        Can use self.call_llm("Return the answer from this text: " + response) again to refine the answer if necessary.

        Args:
            prompt_template: The prompt that was used to query LLM to get the response
            question: Question has a text describing the question but also "Options"
            response: LLM returned a string response
                      Process it and return the answer in the exact format that the evaluator wants to see.
                      Be mindful of the type of answer you need to produce.
                      It can be (A)/(B), a number like 8, or a string, or Yes/No.
        """
        answer = response.split("Answer:")[1].strip()
        return answer

    def forward(self, question):
        """
        question: text

        We read in a question and produces a response
        """
        user_prompt = self.prompt_template + "\n" + question  # we force concatenate here
        response = trace_ops.call_llm(user_prompt)
        answer = self.extract_answer(self.prompt_template, question, response)
        return answer


def run_minibatch_training(model, train, val,
                           batch_size=4,
                           num_epochs=5,
                           save_dir='./bbeh_minibatch_result/', task_name=None):
    # Epoch and batch_size dynamically decide the number of iterations
    # we actually need to fix the iteration count

    examples = train
    val_examples = val

    # Create task-specific directory if task_name is provided
    if task_name:
        task_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
    else:
        task_dir = save_dir
        os.makedirs(task_dir, exist_ok=True)

    optimizer = OptoPrimeV2(model.parameters())

    # Create the guide
    guide = BigBenchGuide()

    # Prepare the training dataset
    train_dataset = {
        'inputs': [ex['question'] for ex in examples],
        'infos': [ex['answer'] for ex in examples]
    }

    # Prepare the validation dataset
    val_dataset = {
        'inputs': [ex['question'] for ex in val_examples],
        'infos': [ex['answer'] for ex in val_examples]
    }

    # Create the MinibatchUpdate algorithm
    algorithm = MinibatchAlgorithm(
        agent=model,
        optimizer=optimizer,
        num_threads=args.num_threads  # Adjust as needed
    )

    # Train the model
    train_scores, val_score = algorithm.train(
        guide=guide,
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,  # Process multiple examples at a time
        eval_frequency=1,  # Evaluate every 5 steps
        save_frequency=1,  # Save every 5 steps
        save_path=os.path.join(task_dir, "checkpoints", "agent.pkl"),
        num_threads=args.num_threads,
        verbose=False,
        min_score=None  # No minimum score required
    )

    # Save model and scores
    model_path = os.path.join(task_dir, "model")
    model.save(model_path)

    # Save scores to JSON
    scores_data = {
        "train_scores": train_scores,
        "val_score": val_score
    }

    with open(os.path.join(task_dir, "scores.json"), "w") as f:
        json.dump(scores_data, f, indent=4)

    return train_scores, val_score


def evaluate_dp(model, examples, save_dir='./bbeh_minibatch_result/', task_name=None):
    """
    Evaluate the model on a set of examples using MinibatchAlgorithm's evaluate method.

    Args:
        model: The model to evaluate
        examples: The examples to evaluate on
        save_dir: Directory to save results
        task_name: Name of the task for organizing results

    Returns:
        accuracy: The accuracy of the model
        responses: The responses of the model
    """
    # Create task-specific directory if task_name is provided
    if task_name:
        task_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
    else:
        task_dir = save_dir
        os.makedirs(task_dir, exist_ok=True)

    # Create the guide
    guide = BigBenchGuide()

    # Prepare the evaluation dataset
    inputs = [ex['question'] for ex in examples]
    infos = [ex['answer'] for ex in examples]

    # Use the evaluate function from basic_algorithm.py
    scores = evaluate(
        agent=model,
        guide=guide,
        inputs=inputs,
        infos=infos,
        min_score=0.0,  # Use 0.0 as the minimum score when an exception occurs
        num_threads=args.num_threads,  # Adjust as needed
        description=f"Evaluating on {len(examples)} examples"  # Add descriptive message for the progress bar
    )

    # Calculate accuracy
    accuracy = np.mean(scores) if scores else 0.0

    # Save evaluation results
    eval_data = {
        "test_score": accuracy,
        "individual_scores": scores.tolist() if isinstance(scores, np.ndarray) else scores
    }

    with open(os.path.join(task_dir, "evaluation_results.json"), "w") as f:
        json.dump(eval_data, f, indent=4)

    return accuracy


def train_baseline(model, train, val, test_set, save_dir='./bbeh_minibatch_result/', task_name=None):
    """
    Run baseline evaluation without optimization.

    Args:
        model: The model to evaluate
        train: Training examples (not used in baseline)
        val: Validation examples
        test_set: Test examples
        save_dir: Directory to save results
        task_name: Name of the task for organizing results

    Returns:
        val_score: Validation accuracy
        test_score: Test accuracy
    """
    # Create task-specific directory if task_name is provided
    if task_name:
        task_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
    else:
        task_dir = save_dir
        os.makedirs(task_dir, exist_ok=True)

    # Evaluate on validation set - pass None as task_name to avoid path duplication
    val_score = evaluate_dp(model, val, save_dir=task_dir, task_name=None)
    print(f"Baseline validation score: {val_score}")

    # Evaluate on test set - pass None as task_name to avoid path duplication
    test_score = evaluate_dp(model, test_set, save_dir=task_dir, task_name=None)
    print(f"Baseline test score: {test_score}")

    # Save scores to JSON with empty train_scores
    scores_data = {
        "train_scores": [],  # Empty list for baseline
        "val_score": val_score
    }

    with open(os.path.join(task_dir, "scores.json"), "w") as f:
        json.dump(scores_data, f, indent=4)

    return val_score, test_score


if __name__ == '__main__':
    import argparse
    import os
    import json
    from datetime import datetime

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run minibatch training on BigBenchExtraHard tasks')
    parser.add_argument('--trial_idx', type=int, default=0, help='Trial index for experiment tracking')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for minibatch training. Try 1|3|5')
    parser.add_argument('--num_threads', type=int, default=5,
                        help='Number of threads to use for evaluation')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline evaluation without optimization')

    # Parse arguments
    args = parser.parse_args()

    # Calculate number of epochs based on batch size to maintain 15 total updates
    # For batch_size=1, we do 1 epoch (15 updates)
    # For batch_size=3, we do 3 epochs (15 updates)
    # For batch_size=5, we do 5 epochs (15 updates)
    if args.batch_size == 1:
        num_epochs = 1
    elif args.batch_size == 3:
        num_epochs = 3
    elif args.batch_size == 5:
        num_epochs = 5
    else:
        # For other batch sizes, calculate epochs to maintain 15 total updates
        updates_per_epoch = 15 / args.batch_size
        num_epochs = int(15 / updates_per_epoch)
        print(f"Warning: Using calculated epochs for batch_size={args.batch_size}. num_epochs={num_epochs}")

    print(f"Training with batch_size={args.batch_size}, num_epochs={num_epochs} (15 total updates)")

    # Update timestamp with trial index for better tracking
    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.trial_idx}"

    # Create a master folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.baseline:
        master_dir = f"./bbeh_minibatch_result_baseline_trial{args.trial_idx}_{timestamp}"
    else:
        master_dir = f"./bbeh_minibatch_result_batch{args.batch_size}_trial{args.trial_idx}_{timestamp}"
    os.makedirs(master_dir, exist_ok=True)

    dataset = load_dataset("hubert233/BigBenchExtraHard")

    tasks = ['boardgame_qa', 'boolean_expressions', 'causal_understanding', 'disambiguation_qa'] # first batch
    tasks += ['dyck_languages', 'geometric_shapes', 'hyperbaton', 'linguini', 'movie_recommendation']  # second batch

    for task in tasks:
        print(f"Processing task: {task}")
        train = dataset[task]
        examples = [{"question": r["input"], "answer": r["target"]} for r in train]

        # for boardgame QA, we might need to shuffle...
        trainset = examples[:15]
        valset = examples[15:25]  # last 10 to validate the performance
        test_set = examples[25:]

        dp = Predict()

        if args.baseline:
            val_score, test_score = train_baseline(dp, trainset, valset, test_set,
                                                   save_dir=master_dir, task_name=task)
            print(f"Completed task: {task}, baseline validation score: {val_score}, test score: {test_score}")
        else:
            train_scores, val_score = run_minibatch_training(dp, trainset, valset,
                                                             batch_size=args.batch_size,
                                                             num_epochs=num_epochs,
                                                             save_dir=master_dir, task_name=task)
            print(f"Completed task: {task}, validation score: {val_score}")

            # Create task-specific directory to avoid path duplication
            task_dir = os.path.join(master_dir, task)
            accuracy = evaluate_dp(dp, test_set, save_dir=task_dir, task_name=None)
            print(f"Completed task: {task}, accuracy: {accuracy}")
