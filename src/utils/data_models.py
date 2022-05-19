""" Utilities used throughout the whole project """
import datetime
from enum import Enum
from typing import List, Optional

import pandas as pd

from pandera import Field, SchemaModel, dataframe_check
from pandera.typing import Series, Index
from pandera.dtypes import DateTime, Category, Float32, Int64
from pydantic import BaseModel


class ExecutionMode(str, Enum):
    """Class to formalize the execution mode"""

    TRAIN = "train"
    INFERENCE = "inference"


class FileEnding(str, Enum):
    """Class to formalize the file endings"""

    CSV = ".csv"
    PARQUET = ".parquet"
    JSON = ".json"
    JOBLIB = ".joblib"


class BaseSchema(SchemaModel):
    """Base schema for all consecutive data schemas"""

    # pylint: disable=no-self-use,no-self-argument

    id: Index[int] = Field(coerce=True, unique=True)

    class Config:
        """Base schema config"""

        strict = "filter"

    @dataframe_check
    def check_index_sorted(cls, df: pd.DataFrame) -> bool:
        """Checks if the index is sorted"""
        return df.index.sort_values().equals(df.index)

    @classmethod
    def get_column_names(cls) -> List[str]:
        """Gets all column names of the schema"""
        return list(cls.to_schema().columns.keys())

    @classmethod
    def get_category_columns(cls) -> List[str]:
        """Gets all category columns of a schema"""
        return [
            name
            for name, dtype in cls.to_schema().dtypes.items()
            if isinstance(dtype, Category)
        ]

    @classmethod
    def get_non_nullable_columns(cls) -> List[str]:
        """Gets all non-nullable columns of a schema"""
        return [
            name
            for name, column in cls.to_schema().columns.items()
            if column.nullable is False
        ]


class RawInputSchema(BaseSchema):
    """Schema for the raw input data"""

    Timestamp: Series[DateTime] = Field(coerce=True, nullable=True)
    Age: Series[float] = Field(coerce=True, nullable=True)
    Gender: Series[str] = Field(coerce=True, nullable=True)
    City: Series[str] = Field(coerce=True, nullable=True)
    Seniority: Series[str] = Field(coerce=True, nullable=True)
    Position: Series[str] = Field(coerce=True, nullable=True)
    Years_of_Experience: Series[str] = Field(coerce=True, nullable=True)
    Company_Size: Series[str] = Field(coerce=True, nullable=True)
    Company_Type: Series[str] = Field(coerce=True, nullable=True)
    Salary_Yearly: Optional[Series[float]] = Field(coerce=True, nullable=True)

    class Config:
        """Raw schema config"""

        name = "RawData"


class CleanedFeaturesSchema(BaseSchema):
    """Schema for the cleaned data"""

    # pylint: disable=no-self-use,no-self-argument

    Year: Series[Int64] = Field(coerce=True, nullable=False)
    Age: Series[Int64] = Field(
        coerce=True, nullable=False, in_range={"min_value": 18, "max_value": 100}
    )
    Gender: Series[Category] = Field(coerce=True, nullable=False)
    City: Series[Category] = Field(coerce=True, nullable=True)
    Seniority: Series[Category] = Field(coerce=True, nullable=True)
    Position: Series[Category] = Field(coerce=True, nullable=True)
    Years_of_Experience: Series[Float32] = Field(coerce=True, nullable=False)
    Company_Size: Series[Category] = Field(coerce=True, nullable=True)
    Company_Type: Series[Category] = Field(coerce=True, nullable=True)

    class Config:
        """Cleaned features schema config"""

        name = "CleanedFeatures"

    @dataframe_check
    def years_of_experience_reasonable_with_age(cls, df: pd.DataFrame) -> Series[bool]:
        """Checks whether experience starts earliest with age 18."""
        return df["Years_of_Experience"] <= df["Age"] - 18


class CleanedTargetsSchema(BaseSchema):
    """Schema for the cleaned targets"""

    Salary_Yearly: Series[Float32] = Field(coerce=True, nullable=False)

    class Config:
        """Cleaned targets schema config"""

        name = "CleanedTargets"


class TransformedFeaturesSchema(BaseSchema):
    """Schema for the transformed data"""

    Year: Series[Int64] = Field(coerce=True, nullable=False)
    Age: Series[Int64] = Field(coerce=True, nullable=False)
    Gender: Series[Int64] = Field(coerce=True, nullable=False)
    City: Series[Int64] = Field(coerce=True, nullable=True)
    Seniority: Series[Int64] = Field(coerce=True, nullable=True)
    Position: Series[Int64] = Field(coerce=True, nullable=True)
    Years_of_Experience: Series[Float32] = Field(coerce=True, nullable=False)
    Company_Size: Series[Int64] = Field(coerce=True, nullable=True)
    Company_Type: Series[Int64] = Field(coerce=True, nullable=True)

    class Config:
        """Transformed features schema config"""

        name = "TransformedFeatures"
        ordered = True


class TransformedTargetsSchema(BaseSchema):
    """Schema for the transformed targets"""

    Salary_Yearly: Series[Float32] = Field(coerce=True, nullable=False)

    class Config:
        """Transformed targets schema config"""

        name = "TransformedTargets"


class RequestInputInference(BaseModel):
    """Raw inference request input"""

    Timestamp: datetime.datetime
    Age: int
    Gender: str
    City: str
    Seniority: str
    Position: str
    Years_of_Experience: float
    Company_Size: str
    Company_Type: str


class RequestInputTrain(RequestInputInference):
    """Raw train request input"""

    Salary_Yearly: float


class PreprocessedRequestInference(BaseModel):
    """Return of the inference preprocessing service"""

    Year: int
    Age: int
    Gender: int
    City: int
    Seniority: int
    Position: int
    Years_of_Experience: float
    Company_Size: int
    Company_Type: int


class PreprocessedRequestTrain(PreprocessedRequestInference):
    """Return of the train preprocessing service"""

    Salary_Yearly: float


class RequestOutput(BaseModel):
    """Return of the prediction service"""

    Salary_Yearly: float
