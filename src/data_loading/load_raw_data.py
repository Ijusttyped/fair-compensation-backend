""" Module to load in raw data """
from typing import List
import logging
import os
import argparse

import pandas as pd

from utils.data_io import read_data, write_data
from utils import log
from utils.data_models import RawInputSchema


logger = logging.getLogger(os.getenv("LOGGER", "default"))


KAGGLE_COLUMN_NAMES = {
    "Zeitstempel": "Timestamp",
    "Your level": "Seniority",
    "Seniority level": "Seniority",
    "Position (without seniority)": "Position",
    "Years of Experience": "Years_of_Experience",
    "Years of experience": "Years_of_Experience",
    "Total years of experience": "Years_of_Experience",
    "Current Salary": "Salary_Yearly",
    "Yearly brutto salary (without bonus and stocks)": "Salary_Yearly",
    "Yearly brutto salary (without bonus and stocks) in EUR": "Salary_Yearly",
    "Company size": "Company_Size",
    "Company type": "Company_Type",
}


class KaggleRawDataLoader:
    """
    Class to load in data and store them in one data frame.
    Args:
        input_paths (List[str]): List of input paths to read in.
    """

    _column_names = KAGGLE_COLUMN_NAMES

    def __init__(self, input_paths: List[str]):
        self.input_paths = input_paths

    def load(self) -> pd.DataFrame:
        """
        Loads all data into one dataframe and validates it based on ``utils.data_models.RawInputSchema``.

        Returns:
            Concatenated raw data.

        """
        all_files = [self.rename_columns(read_data(file)) for file in self.input_paths]
        data = pd.concat(all_files, ignore_index=True, axis=0)
        data = RawInputSchema.to_schema()(data)
        logger.info(
            "Read %d files with a total of %d records", len(self.input_paths), len(data)
        )
        return data

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Strips and renames the columns of the data"""
        data.columns = data.columns.str.strip()
        return data.rename(columns=self._column_names)


def main(input_paths: List[str], output_path: str) -> None:
    """

    Args:
        input_paths (List[str]): List of input paths to read in.
        output_path (str): Path with file ending to store the read data.

    Returns:
        None.
    """
    loader = KaggleRawDataLoader(input_paths=input_paths)
    data = loader.load()
    write_data(data=data, filepath=output_path)


if __name__ == "__main__":
    log.setup_logger("default")
    parser = argparse.ArgumentParser(
        description="Arguments to execute the data loading."
    )
    parser.add_argument(
        "--input-paths",
        "-i",
        dest="input_paths",
        nargs="+",
        required=True,
        help="List of file paths to the kaggle survey data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="Path with file ending to store the loaded raw data.",
    )

    args = parser.parse_args()
    logger.info("Read cli arguments: %s", args)

    main(
        input_paths=args.input_paths,
        output_path=args.output_path,
    )
