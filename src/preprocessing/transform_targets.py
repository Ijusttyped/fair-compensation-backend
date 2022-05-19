""" This module contains all functionality to transform the targets for model training """
import logging
import os
import argparse

import pandas as pd

from preprocessing.data_processor import DataProcessor
from utils.data_models import (
    ExecutionMode,
    CleanedTargetsSchema,
    TransformedTargetsSchema,
)
from utils.data_io import read_data, write_data
from utils import log

logger = logging.getLogger(os.getenv("LOGGER", "default"))


class KaggleTargetTransformer(DataProcessor):
    """
    Class containing all utility functions to transform the targets from the Kaggle survey.
    Args:
        data (pd.DataFrame): The read data.
    """

    def __init__(self, data: pd.DataFrame, mode: ExecutionMode = ExecutionMode.TRAIN):
        data = CleanedTargetsSchema.to_schema()(data)
        super().__init__(data, mode)
        self.data = data
        self.target_column = TransformedTargetsSchema.get_column_names()[0]

    def execute(self) -> pd.DataFrame:
        """Executes all transforming steps for the target column"""
        transformed_data = self.data.copy()
        transformed_targets = TransformedTargetsSchema.to_schema()(transformed_data)
        return transformed_targets


def main(input_path: str, output_path: str, **kwargs) -> None:
    """
    Loads data, executes the target transforming stage and stores the transformed data.
    Args:
        input_path (str): Path with file ending to load the data from.
        output_path (str): Path with file ending to store the transformed target data.
        **kwargs: Additional keyword arguments.

    Returns:
        None.
    """
    data = read_data(filepath=input_path)
    transformer = KaggleTargetTransformer(data=data, **kwargs)
    transformed_targets = transformer.execute()
    write_data(data=transformed_targets, filepath=output_path)


if __name__ == "__main__":
    log.setup_logger("default")
    parser = argparse.ArgumentParser(
        description="Arguments to execute the target transforming for the kaggle survey data."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        dest="input_path",
        required=True,
        help="Path to the cleaned target file of the kaggle survey data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="Path with file ending to store the transformed target data.",
    )

    args = parser.parse_args()
    logger.info("Read cli arguments: %s", args)

    main(input_path=args.input_path, output_path=args.output_path)
