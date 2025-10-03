from opto import trace, trainer

def main():
    true_number = 3
    train_dataset = dict(inputs=[None], infos=[f'Correct answer is: {true_number}'])
    param = trace.node(0, description='An interger to guess', trainable=True)

    trainer.train(
        model=param,
        # optimizer='OptoPrimeV2',  # by default, OPROv2 is used for single-node optimization
        train_dataset=train_dataset,
        # trainer kwargs
        num_epochs=3,
        batch_size=1,
        verbose='output',
    )


if __name__ == "__main__":
    main()
