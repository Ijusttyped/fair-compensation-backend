""" Module for loading the preprocessed data for model training """
from typing import Tuple
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loading.load import DataLoader
from utils.data_models import TransformedTargetsSchema


logger = logging.getLogger(os.getenv("LOGGER", "default"))


class KaggleTrainDataLoader(DataLoader):
    """
    Class to load the kaggle survey data for the model training.
    Args:
        features (pd.DataFrame): The feature data.
        targets (pd.Series): The target data.
        test_size (float, optional): The size of the test set. Defaults to 0.33.
        random_state (int): Value to set the random seed. Defaults to 42.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        test_size: float = 0.33,
        random_state: int = 42,
    ):
        super().__init__(features, targets, test_size, random_state)
        self.target_column = TransformedTargetsSchema.get_column_names()[0]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def setup(self):
        """Sets up the datasets for modeling"""
        valid_data = KaggleTrainDataLoader.match(
            features=self.features, targets=self.targets
        )
        self._split_train_test(data=valid_data)

    @staticmethod
    def match(features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """
        Matches the features and targets to one valid dataframe.
        Args:
            features (pd.DataFrame): The feature data.
            targets (pd.Series): The target data.

        Returns:
            Matched features and targets.
        """
        valid_data = pd.merge(
            features, targets, left_index=True, right_index=True, how="inner"
        )
        return valid_data

    def _split_train_test(self, data: pd.DataFrame) -> None:
        """
        Splits the data in train and test set. Sets will be stored as class variables.
        Args:
            data (pd.DataFrame): The data to split.

        Returns:
            None.
        """
        features = data.copy()
        targets = features.pop(self.target_column)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, targets, test_size=self.test_size, random_state=self.random_state
        )

    def train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the train features and targets"""
        return self.X_train, self.y_train

    def test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the test features and targets"""
        return self.X_test, self.y_test

    def val_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the validation features and targets"""
        raise NotImplementedError
