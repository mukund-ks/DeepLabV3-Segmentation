import click
from train import trainer
from data_processing import processData
from evaluation import evaluator
import traceback


@click.command()
@click.option(
    "--data-dir",
    prompt="Data Directory",
    type=str,
    required=True,
    help="Path for Data Directory.",
)
@click.option(
    "--eval-dir",
    prompt="Evaluation Directory",
    type=str,
    required=True,
    help="Path for Evaluation Directory.",
)
@click.option(
    "-m",
    "--model-type",
    type=click.Choice(["ResNet101", "ResNet50"], case_sensitive=True),
    required=True,
    help="Choice of Encoder.",
)
@click.option(
    "-a",
    "--augmentation",
    type=bool,
    default=True,
    help="Opt-in to apply augmentations to provided data. Also handles data splitting. Default - True",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=4,
    help="Batch size of data during training. Default - 4",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=25,
    help="Number of epochs during training. Default - 25",
)
def main(data_dir, eval_dir, augmentation, batch_size, epochs, model_type) -> None:
    try:
        click.echo(f"Data Pre-Processing Phase\n{'-'*10}")
        processData(data_dir=data_dir, augmentation=augmentation)
        click.echo(f"\nTraining Phase\n{'-'*10}")
        trainer(
            augmentation=augmentation,
            batches=batch_size,
            epochs=epochs,
            modelType=model_type,
            data_dir=data_dir,
        )
        click.echo(f"\nEvaluation Phase\n{'-'*10}")
        evaluator(eval_dir=eval_dir)
    except Exception as _:
        traceback.print_exc()

    return


if __name__ == "__main__":
    main()
