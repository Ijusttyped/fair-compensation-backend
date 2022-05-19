""" Module containing the base model class """
from abc import ABC
from typing import Dict, Union

import pandas as pd
import numpy as np


class Model(ABC):
    """
    Base class for implementing new models.
    Args:
        hyperparameters (Dict[str, Union[str, float, bool]], optional): Dictionary with the hyperparameters to
            initialize the model. Defaults to None.
        **kwargs: Model specific keyword arguments
    """

    def __init__(
        self, hyperparameters: Dict[str, Union[str, float, bool]] = None, **kwargs
    ):
        self.hyperparameters = hyperparameters
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the model with the input data."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Calculates predictions for the input data using the model."""
        raise NotImplementedError

    def load(self, filename: str) -> None:
        """Loads the model from a filename."""
        raise NotImplementedError

    def save(self, filename: str) -> None:
        """Saves the model to disk."""
        raise NotImplementedError
