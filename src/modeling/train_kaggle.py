""" Module with all functionality to train a regression model """
from typing import Literal, Callable, Dict, List
import logging
import os
import argparse

from sklearn.metrics import mean_absolute_error, mean_squared_error

from modeling.models import Model
from modeling.sklearn_models import SKLearnModel
from modeling.train import Trainer
from data_loading.load_train_data import KaggleTrainDataLoader
from utils.data_io import read_data, write_data
from utils import log


logger = logging.getLogger(os.getenv("LOGGER", "default"))


class KaggleSurveyTrainer(Trainer):
    """
    Class to orchestrate the model training and evaluation.
    Args:
        model (``modeling.models.Model``): Model to be trained.
        data_loader (``data_loading.load_train_data.KaggleTrainDataLoader``): Data loader to load training data.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, model: Model, data_loader: KaggleTrainDataLoader, **kwargs):
        super().__init__(model, data_loader, **kwargs)
        self.data_loader.setup()

    def train(self) -> None:
        """Fits the model with the training data."""
        self.model.fit(self.data_loader.X_train, self.data_loader.y_train)

    def evaluate(
        self, metrics: List[Literal["mae", "mse"]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the model for the train and the test set.
        Args:
            metrics (List[str]): Name of the metrics to use for calculating the model score.
                Allowable: `'mae'`, `'mse'`. Defaults to [`'mae'`, `'mse'`].

        Returns:
            None.
        """
        y_pred_train = self.model.predict(self.data_loader.X_train)
        y_pred_test = self.model.predict(self.data_loader.X_test)
        metrics = ["mae", "mse"] if metrics is None else metrics
        evaluation_summary = {"train": {}, "test": {}}
        for metric in metrics:
            metric_calculation = self.__get_metric_calculation(metric)
            evaluation_summary["train"][metric] = metric_calculation(
                self.data_loader.y_train, y_pred_train
            )
            evaluation_summary["test"][metric] = metric_calculation(
                self.data_loader.y_test, y_pred_test
            )
        return evaluation_summary

    @staticmethod
    def __get_metric_calculation(metric: Literal["mae", "mse"]) -> Callable:
        """
        Returns the callable function to calculate a metric from the predictions and the ground truth.
        Args:
            metric (str): Name of the metric to use for calculating the model score. Allowable: `'mae'`, `'mse'`.
                Defaults to `'mae'`.

        Returns:
            Callable metric function.
        """
        if metric == "mae":
            return mean_absolute_error
        if metric == "mse":
            return mean_squared_error
        raise ValueError(f"Metric {metric} is currently not supported.")


def main(feature_path: str, target_path: str, model_path: str, **kwargs) -> None:
    """
    Trains and evaluates a new model with the features and targets. Stores the trained model to the given path.
    Args:
        feature_path (str): Path to the features to train the model with.
        target_path (str): Path to the targets to train the model with.
        model_path (str): Path to store the trained model.
        **kwargs (optional): Optional keyword arguments:
            - hyperparameters_path (str): Path to read hyperparameters to initialize the model.

    Returns:
        None.
    """
    features, targets = read_data(filepath=feature_path), read_data(
        filepath=target_path
    )
    data_loader = KaggleTrainDataLoader(features=features, targets=targets)
    hyperparameters_path = kwargs.get("hyperparameters_path", None)
    hyperparameters = read_data(hyperparameters_path) if hyperparameters_path else {}
    model = SKLearnModel(hyperparameters=hyperparameters)
    trainer = KaggleSurveyTrainer(model=model, data_loader=data_loader)
    trainer.train()
    metrics = trainer.evaluate()
    metrics_path = kwargs.get("metrics_path", None)
    if metrics_path:
        write_data(data=metrics, filepath=metrics_path)
    model.save(filename=model_path)


if __name__ == "__main__":
    log.setup_logger("default")
    parser = argparse.ArgumentParser(
        description="Arguments to execute the model training."
    )
    parser.add_argument(
        "--input-feature-path",
        "-f",
        dest="input_feature_path",
        required=True,
        help="Path to the transformed feature data file of the kaggle survey data.",
    )
    parser.add_argument(
        "--input-target-path",
        "-t",
        dest="input_target_path",
        required=True,
        help="Path to the transformed target data file of the kaggle survey data.",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        dest="model_path",
        required=False,
        help="Path with file ending to store the trained model",
    )
    parser.add_argument(
        "--hyperparameters-path",
        "-p",
        dest="hyperparameters_path",
        required=False,
        default=None,
        help="Path with file ending to load hyperparameter.",
    )
    parser.add_argument(
        "--metrics-path",
        "-e",
        dest="metrics_path",
        required=False,
        default=None,
        help="Path with file ending to store metrics.",
    )

    args = parser.parse_args()
    logger.info("Read cli arguments: %s", args)

    main(
        feature_path=args.input_feature_path,
        target_path=args.input_target_path,
        model_path=args.model_path,
        hyperparameters_path=args.hyperparameters_path,
        metrics_path=args.metrics_path,
    )
