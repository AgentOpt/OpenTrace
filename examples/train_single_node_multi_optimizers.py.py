from opto import trace, trainer
from opto.optimizers.optoprime_v2 import OptimizerPromptSymbolSet

def main():
    true_number = 3
    train_dataset = dict(inputs=[None], infos=[f'Correct answer is: {true_number}'])
    param = trace.node(0, description='An interger to guess', trainable=True)


    # In this toy example, we run PrioritySearch with 2 optimizers to optimize the same parameter with different objectives.
    symbols = OptimizerPromptSymbolSet()
    base_objective = f"You need to change the `{symbols.value_tag}` of the variables in {symbols.variables_section_title} to improve the output in accordance to {symbols.feedback_section_title}"
    optimizer_kwargs_list = [
        dict(objective=base_objective + ". The answer should be an integer between 0 and 5."),
        dict(objective=base_objective + ". The answer should be an integer between -5 and 0"),
    ]

    trainer.train(
        algorithm='PrioritySearch',
        model=param,
        train_dataset=train_dataset,
        # trainer kwargs
        num_epochs=3,
        batch_size=1,
        verbose='output',
        optimizer_kwargs=optimizer_kwargs_list, # use 2 optimizers
        num_candidates=2, # keep exploring the top 2 candidates
    )


if __name__ == "__main__":
    main()
