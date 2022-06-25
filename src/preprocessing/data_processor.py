""" Base classes for implementing Feature and Target Processors """
from abc import ABC, abstractmethod

import pandas as pd

from utils.data_models import ExecutionMode


class DataProcessor(ABC):
    """
    Base class to implement new data processors.
    Args:
        data (pd.DataFrame): Data to process.
        mode (``utils.data_models.ExecutionMode``): Mode to execute: `TRAIN` or `INFERENCE`.
        **kwargs: Additional keyword arguments specific for implemented data processors.
    """

    def __init__(self, data: pd.DataFrame, mode: ExecutionMode, **kwargs):
        self.data = data
        self.mode = mode
        self.kwargs = kwargs

    @abstractmethod
    def execute(self) -> pd.DataFrame:
        """Executes the processing steps."""
        raise NotImplementedError
