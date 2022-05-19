""" Module containing base class to implement data loaders for training. """
from abc import ABC
from typing import Tuple

import pandas as pd


class DataLoader(ABC):
    """
    Base class to load data for the model training.
    Args:
        features (pd.DataFrame): The feature data.
        targets (pd.Series): The target data.
        test_size (float, optional): The size of the test set. Defaults to 0.2.
        random_state (int): Value to set the random seed. Defaults to 42.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.features = features
        self.targets = targets
        self.test_size = test_size
        self.random_state = random_state

    def setup(self):
        """Sets up the datasets for modeling"""
        raise NotImplementedError

    def train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the train features and targets"""
        raise NotImplementedError

    def test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the test features and targets"""
        raise NotImplementedError

    def val_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the validation features and targets"""
        raise NotImplementedError
