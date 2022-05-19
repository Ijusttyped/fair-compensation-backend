""" Module containing all model classes """
from typing import Literal, Dict, Union
import logging
import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from modeling.models import Model
from utils.data_io import read_data, write_data


logger = logging.getLogger(os.getenv("LOGGER", "default"))


class SKLearnModel(Model):
    """
    Class wrapping the SKLearn models.
    Args:
        hyperparameters (Dict[str, Union[str, float, bool]], optional): Dictionary with the hyperparameters to
            initialize the model. Defaults to None.
        model_type (str, optional): Name of the model object to initialize.
            Allowable: `'RandomForest'`, `'GradientBoosting'`. Defaults to `'RandomForest'`.
    """

    def __init__(
        self,
        hyperparameters: Dict[str, Union[str, float, bool]] = None,
        model_type: Literal["RandomForest", "GradientBoosting"] = "RandomForest",
    ):
        super().__init__(hyperparameters)
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.model_type = model_type
        self.model = self.__initialize_model()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the model with the given input data.
        Args:
            X (pd.DataFrame): Features for training the model.
            y (pd.Series): Targets to learn during training.

        Returns:
            None.
        """
        self.model.fit(X, y)
        logger.info("Fitted model with %d train records.", len(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts values for the given input data using the model.
        Args:
            X (pd.DataFrame): Data to calculate predictions for. Features have to be the same used during training.

        Returns:
            Array with the prediction.
        """
        prediction = self.model.predict(X)
        logger.info("Calculated predictions for %d records.", len(X))
        return prediction

    def load(self, filename: str) -> None:
        """Loads a model from a given filename."""
        self.model = read_data(filepath=filename)

    def save(self, filename: str):
        """Stores the model object to the given filename."""
        write_data(data=self.model, filepath=filename)

    def __initialize_model(self):
        """Initializes a new model based on the model_type."""
        if self.model_type == "RandomForest":
            return RandomForestRegressor(**self.hyperparameters)
        if self.model_type == "GradientBoosting":
            return GradientBoostingRegressor(**self.hyperparameters)
        raise ValueError(f"Model type {self.model_type} not supported.")
