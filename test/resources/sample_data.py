"""Collection of sample data used in tests."""
from datetime import datetime

import pandas as pd


RAW_DATA_1 = pd.DataFrame(
    {
        "Company size": ["100+", "1-10"],
        "Current Salary": [100000.0, 50000.0],
        "Age": [24.0, 40.0],
        "Gender": ["male", "female"],
        "City": ["Berlin", "Cologne"],
        "Seniority": ["Mid", "Junior"],
        "Position": ["Developer", "Engineer"],
        "Total years of experience": [4, 20],
        "Company type": ["Product", "Startup"],
        "Timestamp": ["2020/01/01", "2020/09/15"],
    }
)

RAW_DATA_2 = pd.DataFrame(
    {
        "Zeitstempel": ["2022/01/22", "2021/08/29"],
        "Company size": ["100", "50-100"],
        "Current Salary": [55000.0, 65000.0],
        "Age": [34.0, 33.0],
        "Gender": ["diverse", "female"],
        "City": ["Berlin", "Cologne"],
        "Seniority": ["Senior", "Mid"],
        "Position": ["Data Scientist", "Manager"],
        "Total years of experience": [10, 9],
        "Company type": ["Consulting", "Product"],
    }
)

RAW_DATA_COMBINED = pd.DataFrame(
    {
        "Company_Size": ["100+", "1-10", "100", "50-100"],
        "Salary_Yearly": [100000.0, 50000.0, 55000.0, 65000.0],
        "Age": [24.0, 40.0, 34.0, 33.0],
        "Gender": ["male", "female", "diverse", "female"],
        "City": ["Berlin", "Cologne", "Berlin", "Cologne"],
        "Seniority": ["Mid", "Junior", "Senior", "Mid"],
        "Position": [
            "Developer",
            "Engineer",
            "Data Scientist",
            "Manager",
        ],
        "Years_of_Experience": ["4", "20", "10", "9"],
        "Company_Type": ["Product", "Startup", "Consulting", "Product"],
        "Timestamp": [
            datetime(2020, 1, 1),
            datetime(2020, 9, 15),
            datetime(2022, 1, 22),
            datetime(2021, 8, 29),
        ],
    }
)

FEATURES = pd.DataFrame(
    {
        "Year": [2, 4, 5, -1, 3, 2],
        "Age": [2, 4, 5, -1, 3, 2],
        "Gender": [2, 4, 5, -1, 3, 2],
        "City": [2, 4, 5, -1, 3, 2],
        "Seniority": [2, 4, 5, -1, 3, 2],
        "Position": [2, 4, 5, -1, 3, 2],
        "Years_of_Experience": [2, 4, 5, -1, 3, 2],
        "Company_Size": [2, 4, 5, -1, 3, 2],
        "Company_Type": [2, 4, 5, -1, 3, 2],
    }
)

TARGETS = pd.Series([50000, 67000, 66000, 89000, 76000], name="Salary_Yearly")
