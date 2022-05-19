""" This module contains all functionality to clean the targets """
import logging
import os
import argparse

import pandas as pd

from preprocessing.data_processor import DataProcessor
from utils.data_models import ExecutionMode, RawInputSchema, CleanedTargetsSchema
from utils.data_io import read_data, write_data
from utils import log

logger = logging.getLogger(os.getenv("LOGGER", "default"))


class KaggleTargetCleaner(DataProcessor):
    """
    Class containing all utility functions to clean the targets from the Kaggle survey.
    Args:
        data (pd.DataFrame): The read data.
    """

    def __init__(self, data: pd.DataFrame, mode: ExecutionMode = ExecutionMode.TRAIN):
        data = RawInputSchema.to_schema()(data)
        super().__init__(data, mode)
        self.data = data
        self.target_column = CleanedTargetsSchema.get_column_names()[0]

    def execute(self) -> pd.DataFrame:
        """Executes all cleaning steps for the target column"""
        cleaned = self.data.dropna(subset=[self.target_column])
        cleaned_data = self.remove_outliers(data=cleaned)
        cleaned_targets = CleanedTargetsSchema.to_schema()(cleaned_data)
        return cleaned_targets

    @staticmethod
    def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers to only keep data in range(Q1 - 1.5 x IQR, Q3 + 1.5 x IQR).
        Args:
            data (pd.DataFrame): Data to restrict.

        Returns:
            Data without outliers.
        """
        salary = data["Salary_Yearly"]
        q1 = salary.quantile(0.25)
        q3 = salary.quantile(0.75)
        iqr = q3 - q1

        lower_lim = q1 - 1.5 * iqr
        upper_lim = q3 + 1.5 * iqr
        cleaned_data = data[(salary >= lower_lim) & (salary <= upper_lim)]
        logger.info(
            "Dropped %s records to keep target in range [%d,%d]",
            len(data) - len(cleaned_data),
            lower_lim,
            upper_lim,
        )
        return cleaned_data


def main(input_path: str, output_path: str, **kwargs) -> None:
    """
    Loads data, executes the target cleaning stage and stores the cleaned data.
    Args:
        input_path (str): Path with file ending to load the data from.
        output_path (str): Path with file ending to store the cleaned target data.
        **kwargs: Additional keyword arguments.

    Returns:
        None.
    """
    data = read_data(filepath=input_path)
    cleaner = KaggleTargetCleaner(data=data, **kwargs)
    cleaned_targets = cleaner.execute()
    write_data(data=cleaned_targets, filepath=output_path)


if __name__ == "__main__":
    log.setup_logger("default")
    parser = argparse.ArgumentParser(
        description="Arguments to execute the target cleaning for the kaggle survey data."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        dest="input_path",
        required=True,
        help="Path to the raw data file of the kaggle survey data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="Path with file ending to store the cleaned target data.",
    )

    args = parser.parse_args()
    logger.info("Read cli arguments: %s", args)

    main(input_path=args.input_path, output_path=args.output_path)
