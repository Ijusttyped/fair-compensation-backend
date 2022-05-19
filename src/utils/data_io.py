""" Utility functions to input and output data """
from typing import Union, Any
from pathlib import Path
import logging
import json
import joblib

import pandas as pd

from utils.data_models import FileEnding


logger = logging.getLogger("default")


def read_data(
    filepath: str, file_ending: FileEnding = None
) -> Union[pd.DataFrame, dict, Any]:
    """
    Reads in data into memory.
    Args:
        filepath (str): Path to the file to read.
        file_ending (``utils.data_models.FileEnding``, optional): Enforce reading with a specific file ending.
            Defaults to None.

    Returns:
        The data as pandas dataframe.
    """
    if not file_ending:
        file_ending = FileEnding(Path(filepath).suffix)
    if file_ending is FileEnding.CSV:
        data = pd.read_csv(filepath)
    elif file_ending is FileEnding.PARQUET:
        data = pd.read_parquet(filepath)
    elif file_ending is FileEnding.JSON:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_ending is FileEnding.JOBLIB:
        data = joblib.load(filepath)
    else:
        raise ValueError(
            f"File ending {file_ending.value} currently not supported to be read."
        )
    logger.info("Successfully read file from %s", filepath)
    return data


def write_data(
    data: Union[pd.DataFrame, dict, Any], filepath: str, store_index: bool = True
) -> None:
    """
    Writes a given dataframe to disk.
    Args:
        data (pd.DataFrame): The data to store.
        filepath (str): Path with file ending to the storage location.
        store_index (bool, optional): Whether to store the index of the input data. Defaults to ``True``.

    Returns:
        None.
    """
    file_ending = FileEnding(Path(filepath).suffix)
    if not Path(filepath).parent.is_dir():
        Path(filepath).parent.mkdir(parents=True)
    if file_ending is FileEnding.CSV:
        data.to_csv(filepath, index=store_index)
    elif file_ending is FileEnding.PARQUET:
        data.to_parquet(filepath, index=store_index)
    elif file_ending is FileEnding.JSON:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif file_ending is FileEnding.JOBLIB:
        joblib.dump(data, filepath)
    else:
        raise ValueError(
            f"File ending {file_ending.value} currently not supported to be read."
        )
    logger.info("Successfully wrote file to %s", filepath)
