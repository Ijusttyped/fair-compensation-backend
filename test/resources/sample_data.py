"""Collection of sample data used in tests."""
from datetime import datetime

import pandas as pd
import numpy as np

from utils.data_models import (
    CleanedFeaturesSchema,
    CleanedTargetsSchema,
    TransformedFeaturesSchema,
)


RAW_DATA_1 = pd.DataFrame(
    {
        "Company size": ["101-1000", "1-100", np.NaN],
        "Current Salary": [100000.0, 50000.0, np.NaN],
        "Age": [24.0, 40.0, np.NaN],
        "Gender": ["male", "female", np.NaN],
        "City": ["Berlin", "Cologne", np.NaN],
        "Seniority": ["Mid", "Junior", np.NaN],
        "Position": ["Developer", "Engineer", np.NaN],
        "Total years of experience": [8.0, 20.0, np.NaN],
        "Company type": ["Product", "Startup", np.NaN],
        "Timestamp": ["2020/01/01", "2020/09/15", np.NaN],
    }
)

RAW_DATA_2 = pd.DataFrame(
    {
        "Zeitstempel": ["2022/01/22", "2021/08/29", "2021/08/29"],
        "Company size": ["11-50", "50-100", "50-100"],
        "Current Salary": [55000.0, 65000.0, 65000.0],
        "Age": [34.0, 33.0, 33.0],
        "Gender": ["diverse", "female", "female"],
        "City": ["Berlin", "Cologne", "Cologne"],
        "Seniority": ["Senior", "Mid", "Mid"],
        "Position": ["Data Scientist", "Manager", "Manager"],
        "Total years of experience": [10.0, 9.0, 9.0],
        "Company type": ["Consulting", "Product", "Product"],
    }
)

RAW_DATA_COMBINED = pd.DataFrame(
    {
        "Company_Size": ["101-1000", "1-100", np.NaN, "11-50", "50-100", "50-100"],
        "Salary_Yearly": [100000.0, 50000.0, np.NaN, 55000.0, 65000.0, 65000.0],
        "Age": [24.0, 40.0, np.NaN, 34.0, 33.0, 33.0],
        "Gender": ["male", "female", np.NaN, "diverse", "female", "female"],
        "City": ["Berlin", "Cologne", np.NaN, "Berlin", "Cologne", "Cologne"],
        "Seniority": ["Mid", "Junior", np.NaN, "Senior", "Mid", "Mid"],
        "Position": [
            "Developer",
            "Engineer",
            np.NaN,
            "Data Scientist",
            "Manager",
            "Manager",
        ],
        "Years_of_Experience": ["8.0", "20.0", np.NaN, "10.0", "9.0", "9.0"],
        "Company_Type": [
            "Product",
            "Startup",
            np.NaN,
            "Consulting",
            "Product",
            "Product",
        ],
        "Timestamp": [
            datetime(2020, 1, 1),
            datetime(2020, 9, 15),
            np.NaN,
            datetime(2022, 1, 22),
            datetime(2021, 8, 29),
            datetime(2021, 8, 29),
        ],
    }
)

CLEANED_FEATURES = CleanedFeaturesSchema(
    pd.DataFrame(
        {
            "Company_Size": ["101-1000", "1-100", "1-100", "1-100"],
            "Age": [24, 40, 34, 33],
            "Gender": ["male", "female", "diverse", "female"],
            "City": ["berlin", "cologne", "berlin", "cologne"],
            "Seniority": ["mid", "junior", "senior", "mid"],
            "Position": ["software developer", "engineer", "data scientist", "manager"],
            "Years_of_Experience": [6.0, 20.0, 10.0, 9.0],
            "Company_Type": ["product", "startup", "consulting or agency", "product"],
            "Year": [2020, 2020, 2022, 2021],
        },
        index=pd.Index([0, 1, 3, 4]),
    )
)

CLEANED_TARGETS = CleanedTargetsSchema(
    pd.DataFrame(
        {
            "Salary_Yearly": [50000.0, 55000.0, 65000.0, 65000.0],
        },
        index=pd.Index([1, 3, 4, 5]),
    )
)

TRANSFORMED_FEATURES = TransformedFeaturesSchema(
    pd.DataFrame(
        {
            "Year": [2020, 2020, 2022, 2021],
            "Age": [24, 40, 34, 33],
            "Gender": [2, 1, 0, 1],
            "City": [0, 1, 0, 1],
            "Seniority": [1, 0, 2, 1],
            "Position": [3, 1, 0, 2],
            "Years_of_Experience": [6.0, 20.0, 10.0, 9.0],
            "Company_Size": [1, 0, 0, 0],
            "Company_Type": [1, 2, 0, 1],
        },
        index=pd.Index([0, 1, 3, 4]),
    )
)

TRANSFORMED_TARGETS = CLEANED_TARGETS

TRAIN_DATA = pd.DataFrame(
    {
        "Year": [2020, 2022, 2021],
        "Age": [40, 34, 33],
        "Gender": [1, 0, 1],
        "City": [1, 0, 1],
        "Seniority": [0, 2, 1],
        "Position": [1, 0, 2],
        "Years_of_Experience": [20.0, 10.0, 9.0],
        "Company_Size": [0, 0, 0],
        "Company_Type": [2, 0, 1],
        "Salary_Yearly": [50000.0, 55000.0, 65000.0],
    },
    index=pd.Index([1, 3, 4]),
)
