""" This module contains all functionality to transform cleaned data for modeling """
import os
from typing import Dict, List, Union
import logging
import argparse

import pandas as pd

from preprocessing.data_processor import DataProcessor
from utils import log
from utils.data_io import read_data, write_data
from utils.data_models import (
    CleanedFeaturesSchema,
    TransformedFeaturesSchema,
    ExecutionMode,
)


logger = logging.getLogger(os.getenv("LOGGER", "default"))


class KaggleFeatureTransformer(DataProcessor):
    """
    Transforms the features so that they can be consumed by the model stage. Steps:
        - Encoding of categorical values
    Args:
        data (pd.DataFrame): Data to transform.
        mode (``utils.data_models.ExecutionMode``): Mode to execute: `TRAIN` or `INFERENCE`.
        **kwargs: Additional keyword arguments:
            - labels_path (str): Path to load labels used during training. Mandatory argument for mode `INFERENCE`.
    """

    def __init__(self, data: pd.DataFrame, mode: ExecutionMode, **kwargs):
        data = CleanedFeaturesSchema.to_schema()(data)
        super().__init__(data, mode, **kwargs)
        self.labels_path = self.kwargs.get("labels_path", None)
        self.labels = (
            self._load_encoding_labels()
            if self.mode is ExecutionMode.INFERENCE
            else None
        )

    def execute(self) -> pd.DataFrame:
        """
        Executes the feature transforming.
        Returns:
            Transformed features. Labels used for encoding.
        """
        data = self.encode_categorical_features(labels=self.labels)
        data = TransformedFeaturesSchema.to_schema()(
            data[TransformedFeaturesSchema.get_column_names()]
        )
        return data

    def encode_categorical_features(
        self, labels: Union[None, Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Encodes all categorical columns in the input data.
        Options for encoding:
            - based on already present labels by passing a dictionary as the labels argument
            - by creating new labels
        Args:
            labels (Dict[str, List[str]], optional): Dictionary of type {'column': ['label1', ...], ...}.
                Defaults to None.
        Returns:
            The encoded data.
        """
        encoded_data = self.data.copy()
        if labels is not None:
            for column, values in labels.items():
                encoded_data[column] = encoded_data[column].apply(
                    self.encode_based_on_list, label_values=values
                )
                logger.info("Successfully encoded column %s", column)
                # Apply doesn't work on columns where all entries are nan
                if encoded_data[column].isnull().all():
                    encoded_data[column] = -1
                    logger.warning(
                        "Column %s only contains nan values. Encoding with -1 successful.",
                        column,
                    )
        else:
            labels = {}
            for column in CleanedFeaturesSchema.get_category_columns():
                labels[column] = encoded_data[column].cat.categories.tolist()
                encoded_data[column] = encoded_data[column].cat.codes
                logger.info("Successfully encoded column %s", column)
            self.labels = labels
        return encoded_data

    @staticmethod
    def encode_based_on_list(value: str, label_values: list):
        """
        Encodes a input value by index of the input labels
        Args:
            value (str): Value to encode.
            label_values (list): Reference list.

        Returns:
            Encoded value.
        """
        if pd.isna(value):
            encoding = -1
        elif value not in label_values:
            encoding = len(label_values)
        else:
            encoding = label_values.index(value)
        return encoding

    def _load_encoding_labels(self) -> Dict[str, List[str]]:
        """Loads in the labels dictionary"""
        if self.labels_path is None:
            raise ValueError(
                "Providing a path to load the labels used in training is mandatory in mode 'INFERENCE. "
                "Please provide 'labels_path' argument to KaggleFeatureTransformer."
            )
        labels = read_data(filepath=self.labels_path)
        return labels


def main(input_path: str, output_path: str, mode: str, **kwargs) -> None:
    """
    Loads data, executes the transforming stage and stores the transformed data.
    Args:
        input_path (str): Path with file ending to load the data from.
        output_path (str): Path with file ending to store the transformed data.
        mode (str): Name of the execution mode. Either 'train' or 'inference'.
        **kwargs: Additional keyword arguments:
            - | labels_path (str): Path to load / store labels used for encoding during training.
              | Mandatory argument for mode `INFERENCE`.

    Returns:
        None.
    """
    data = read_data(filepath=input_path)
    transformer = KaggleFeatureTransformer(
        data=data, mode=ExecutionMode(mode), **kwargs
    )
    transformed_data = transformer.execute()
    write_data(data=transformed_data, filepath=output_path)
    if ExecutionMode(mode) is ExecutionMode.TRAIN and kwargs.get("labels_path", None):
        write_data(data=transformer.labels, filepath=kwargs.get("labels_path"))


if __name__ == "__main__":
    log.setup_logger("default")
    parser = argparse.ArgumentParser(
        description="Arguments to execute the feature transforming for the kaggle survey data."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        dest="input_path",
        required=True,
        help="Path to the cleaned data file of the kaggle survey data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="Path with file ending to store the transformed feature data.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        dest="mode",
        required=True,
        help="Mode to execute the cleaning step. Either 'train' or 'inference'.",
    )
    parser.add_argument(
        "--labels-path",
        "-l",
        dest="labels_path",
        required=False,
        default=None,
        help="Path to load / store labels for encoding.",
    )

    args = parser.parse_args()
    logger.info("Read cli arguments: %s", args)

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        mode=args.mode,
        labels_path=args.labels_path,
    )
