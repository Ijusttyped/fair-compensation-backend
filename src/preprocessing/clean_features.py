""" This module contains functionality to clean loaded raw data """
import os
import logging
import argparse

import numpy as np
import pandas as pd

from preprocessing.data_processor import DataProcessor
from preprocessing.kaggle_survey_mappings import MAPPINGS, REGEX_MAPPINGS
from utils import log
from utils.data_io import read_data, write_data
from utils.data_models import RawInputSchema, CleanedFeaturesSchema, ExecutionMode


logger = logging.getLogger(os.getenv("LOGGER", "default"))


class KaggleFeatureCleaner(DataProcessor):
    """
    Class containing all utility functions to clean raw data from the Kaggle survey.
    Args:
        data (pd.DataFrame): The read raw data.
        mode (``utils.data_models.ExecutionMode``): Mode to execute: `TRAIN` or `INFERENCE`.
    """

    def __init__(self, data: pd.DataFrame, mode: ExecutionMode, **kwargs):
        data = RawInputSchema.to_schema()(data)
        super().__init__(data, mode, **kwargs)

    def execute(self) -> pd.DataFrame:
        """
        Executes all steps to clean the Kaggle survey data.

        Returns:
            The cleaned data.
        """
        cleaned_data = self.data.copy()
        cleaned_data["Years_of_Experience"] = self.clean_years_of_experience_column(
            years_of_experience=cleaned_data["Years_of_Experience"],
            age=cleaned_data["Age"],
        )
        cleaned_data["Position"] = self.clean_position_column(
            position=cleaned_data["Position"], seniority=cleaned_data["Seniority"]
        )
        cleaned_data["Year"] = self.timestamp_to_year(
            timestamp=cleaned_data["Timestamp"]
        )
        cleaned_data = self.unify_row_values(data=cleaned_data)
        if self.mode is ExecutionMode.TRAIN:
            cleaned_data = self.remove_null_and_duplicate_records(data=cleaned_data)
            category_columns = CleanedFeaturesSchema.get_category_columns()
            cleaned_data[category_columns] = self.reduce_cardinality(
                data=cleaned_data[category_columns]
            )
        cleaned_data = CleanedFeaturesSchema.to_schema()(cleaned_data)
        return cleaned_data

    @staticmethod
    def reduce_cardinality(data: pd.DataFrame, percentage: float = 0.02):
        """
        Reduces the cardinality of categorical features by grouping low cardinality values in a new category 'other'.
        If for the frequency of a value f is true: `2 <= f < `percentage * #records``, it gets replaced.
        Args:
            data (pd.DataFrame): Data with all categorical columns to group.
            percentage (float, optional): Percentage value under which the values are assigned to the new category.
                Defaults to `0.02`.

        Returns:
            The data with newly grouped values.
        """
        cleaned = data.copy()
        min_num_records = round(percentage * len(data))

        for column in data.columns:
            counts = data[column].value_counts()
            change_to_other = counts[counts < min_num_records].index
            if len(change_to_other) >= 2:
                cleaned[column] = cleaned[column].replace(
                    {value: "other" for value in change_to_other}
                )
                logger.info(
                    "Successfully grouped %d levels in category %s",
                    len(change_to_other),
                    column,
                )
        return cleaned

    @staticmethod
    def remove_null_and_duplicate_records(data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes null values of non-nullable columns and drops duplicate rows.
        Args:
            data (pd.Dataframe): The dataframe to remove records from.

        Returns:
            The cleaned data.
        """
        cleaned_nulls = data.dropna(
            subset=CleanedFeaturesSchema.get_non_nullable_columns()
        )
        cleaned = cleaned_nulls.drop_duplicates(
            subset=list(set(CleanedFeaturesSchema.get_column_names()) - {"Timestamp"})
        )
        logger.info(
            "Removed %d null values and %d duplicates.",
            len(data) - len(cleaned_nulls),
            len(cleaned_nulls) - len(cleaned),
        )
        return cleaned

    @staticmethod
    def unify_row_values(data: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces values in the data:
            - by lowering and stripping all string characters
            - with manual defined mappings from ``preprocessing.kaggle_survey_mappings`` module
            - all empty strings with nan
        Args:
            data (pd.DataFrame): Data to apply mappings to.

        Returns:
            Data with applied mappings.
        """
        cleaned = data.copy()
        for column in CleanedFeaturesSchema.get_category_columns():
            cleaned[column] = cleaned[column].str.lower().str.strip()
        cleaned = cleaned.replace(MAPPINGS)
        cleaned = cleaned.replace(REGEX_MAPPINGS, regex=True)
        cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)
        logger.info(
            "Applied mappings, lowering and replacement of empty strings with nan."
        )
        return cleaned

    @staticmethod
    def timestamp_to_year(timestamp: pd.Series) -> pd.Series:
        """
        Cleans the column `"Timestamp"` by only considering the year.
        Args:
            timestamp (pd.Series): Series with timestamp.

        Returns:
            Series with the year of the timestamps.
        """
        year_timestamp = timestamp.dt.year
        year_timestamp = year_timestamp.rename("Year")
        logger.info("Transformed column 'Timestamp' to years.")
        logger.debug("Got data from following years: %s", year_timestamp.unique())
        return year_timestamp

    @staticmethod
    def clean_position_column(position: pd.Series, seniority: pd.Series) -> pd.Series:
        """
        Cleans the column `"Position"` by replacing the seniority level from the position name.
        Args:
            position (pd.Series): Series of the position column.
            seniority (pd.Series): Series with the seniority level.

        Returns:
            The cleaned series of the position names.
        """

        cleaned = pd.DataFrame(data=[position, seniority]).T.apply(
            KaggleFeatureCleaner._replace_seniority_in_position_name, axis=1
        )
        cleaned = cleaned.astype("object").rename("Position")
        logger.info("Cleaned column 'Position'.")
        return cleaned

    @staticmethod
    def clean_years_of_experience_column(
        years_of_experience: pd.Series, age: pd.Series
    ) -> pd.Series:
        """
        Cleans the column `"Years_of_Experience"` by:
            - replacing `,` with `.`
            - starting experience at age 18
        Args:
            years_of_experience (pd.Series): Series with the years of experience data.
            age (pd.Series): Series with the age data.

        Returns:
            Cleaned Series.
        """
        cleaned = years_of_experience.str.replace(",", ".")
        cleaned = cleaned.apply(KaggleFeatureCleaner._transform_to_float)
        cleaned[(age - cleaned) < 18] = age - 18
        logger.info("Cleaned column 'Years_of_Experience'")
        return cleaned

    @staticmethod
    def _replace_seniority_in_position_name(row: pd.DataFrame):
        """Removes the seniority of the position"""
        if pd.isna(row["Seniority"]):
            return row["Position"]
        if not pd.isna(row["Position"]):
            return row["Position"].replace(row["Seniority"], "").strip()
        return np.NAN

    @staticmethod
    def _transform_to_float(value: str) -> float:
        """
        Transforms a string to float if possible. If not nan is returned.
        Args:
            value (str): Value to be transformed.

        Returns:
            The float value.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan


def main(input_path: str, output_path: str, mode: str, **kwargs) -> None:
    """
    Loads data, executes the cleaning stage and stores the cleaned data.
    Args:
        input_path (str): Path with file ending to load the data from.
        output_path (str): Path with file ending to store the cleaned data.
        mode (str): Name of the execution mode. Either 'train' or 'inference'.

    Returns:
        None.
    """
    data = read_data(filepath=input_path)
    cleaner = KaggleFeatureCleaner(data=data, mode=ExecutionMode(mode), **kwargs)
    cleaned_data = cleaner.execute()
    write_data(data=cleaned_data, filepath=output_path)


if __name__ == "__main__":
    log.setup_logger("default")
    parser = argparse.ArgumentParser(
        description="Arguments to execute the feature cleaning for the kaggle survey data."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        dest="input_path",
        required=True,
        help="Path to the loaded raw data file of the kaggle survey data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="Path with file ending to store the cleaned feature data.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        dest="mode",
        required=True,
        help="Mode to execute the cleaning step. Either 'train' or 'inference'.",
    )

    args = parser.parse_args()
    logger.info("Read cli arguments: %s", args)

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        mode=args.mode,
    )
