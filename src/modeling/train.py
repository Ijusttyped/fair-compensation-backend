""" Module containing the base class for implementing new trainers. """
from abc import ABC, abstractmethod
from typing import Dict, List

from modeling.models import Model


class Trainer(ABC):
    """
    Class to orchestrate the model training and evaluation.
    Args:
        model (``modeling.models.Model``): Model to be trained.
        data_loader (``data_loading.load_train_data.KaggleTrainDataLoader``): Data loader to load training data.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, model: Model, data_loader, **kwargs):
        self.model = model
        self.data_loader = data_loader
        self.kwargs = kwargs

    @abstractmethod
    def train(self) -> None:
        """Fits the model with the training data."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Evaluates the model for the train and the test set."""
        raise NotImplementedError
