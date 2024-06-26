import traceback

import click

from evaluation import evaluator
from src.data_processing import processData
from train import trainer


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
    "-M",
    "--model-type",
    prompt="Model Type",
    type=click.Choice(["ResNet101", "ResNet50", "Xception", "EfficientNetB5"], case_sensitive=True),
    required=True,
    help="Choice of Encoder.",
)
@click.option(
    "-A",
    "--augmentation",
    type=bool,
    default=True,
    help="Opt-in to apply augmentations to provided data. Default - True",
)
@click.option(
    "-S",
    "--split-data",
    type=bool,
    default=True,
    help="Opt-in to split data into Training and Validation set. Default - True",
)
@click.option(
    "--stop-early",
    type=bool,
    default=True,
    help="Opt-in to stop Training early if val_loss isn't improving. Default - True",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=4,
    help="Batch size of data during training. Default - 4",
)
@click.option(
    "-E",
    "--epochs",
    type=int,
    default=25,
    help="Number of epochs during training. Default - 25",
)
def main(
    data_dir: str,
    eval_dir: str,
    augmentation: bool,
    split_data: bool,
    stop_early: bool,
    batch_size: int,
    epochs: int,
    model_type: str,
) -> None:
    """
    A DeepLab V3+ Decoder based Binary Segmentation Model with choice of Encoders b/w ResNet101, ResNet50, Xception or EfficientNetB5.\n
    Please make sure that your data is structured according to the folder structure specified in the Github Repository.\n
    See: https://github.com/mukund-ks/DeepLabV3-Segmentation
    """
    try:
        click.echo(f"\nData Pre-Processing Phase\n{'-'*10}")
        processData(data_dir=data_dir, augmentation=augmentation, split_data=split_data)
        click.echo(f"\nTraining Phase\n{'-'*10}")
        trainer(
            stop_early=stop_early,
            batches=batch_size,
            epochs=epochs,
            modelType=model_type,
        )
        click.echo(f"\nEvaluation Phase\n{'-'*10}")
        evaluator(eval_dir=eval_dir)
    except Exception as _:
        traceback.print_exc()

    return


if __name__ == "__main__":
    main()
