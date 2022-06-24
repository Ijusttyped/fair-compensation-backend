""" Tests the cleaning of features. """
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from test.resources.sample_data import RAW_DATA_COMBINED, CLEANED_FEATURES
from parameterized import parameterized

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from preprocessing.clean_features import KaggleFeatureCleaner, main
from utils.data_models import ExecutionMode


class KaggleFeatureCleanerStaticFunctionsTest(unittest.TestCase):
    """Test case for the static functions for cleaning specific columns."""
    # pylint: disable=no-self-use,too-many-arguments

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {
                        "Position": [
                            "software developer",
                            "cto",
                            "software developer",
                            "software developer",
                            "qa engineer",
                            "cto",
                        ],
                        "Company_Type": [
                            "product",
                            "product",
                            "startup",
                            "product",
                            "finance",
                            "outsource",
                        ],
                        "City": [
                            "berlin",
                            "berlin",
                            "berlin",
                            "cologne",
                            "zurich",
                            "amsterdam",
                        ],
                    }
                ),
                # Minimum 2 records with same value
                0.4,
                pd.DataFrame(
                    {
                        "Position": [
                            "software developer",
                            "cto",
                            "software developer",
                            "software developer",
                            "qa engineer",
                            "cto",
                        ],
                        "Company_Type": [
                            "product",
                            "product",
                            "other",
                            "product",
                            "other",
                            "other",
                        ],
                        "City": [
                            "berlin",
                            "berlin",
                            "berlin",
                            "other",
                            "other",
                            "other",
                        ],
                    }
                ),
            ),
        ]
    )
    def test_reduce_cardinality(self, data, percentage, expected):
        """Tests if cardinality of column values under a percentage is grouped together."""
        actual = KaggleFeatureCleaner.reduce_cardinality(
            data=data, percentage=percentage
        )
        assert_frame_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {
                        "Age": [25, 32, 40, 32],
                        "Gender": ["female", "male", "female", "male"],
                    }
                ),
                ["Age", "Gender"],
                pd.DataFrame(
                    {
                        "Age": [25, 32, 40],
                        "Gender": ["female", "male", "female"],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "Gender": ["female", "male", "male", np.NaN, "female"],
                        "Position": [
                            "developer",
                            "developer",
                            "developer",
                            np.NaN,
                            "developer",
                        ],
                    }
                ),
                ["Gender", "Position"],
                pd.DataFrame(
                    {
                        "Gender": ["female", "male"],
                        "Position": ["developer", "developer"],
                    }
                ),
            ),
        ]
    )
    @patch("preprocessing.clean_features.CleanedFeaturesSchema.get_column_names")
    @patch(
        "preprocessing.clean_features.CleanedFeaturesSchema.get_non_nullable_columns"
    )
    def test_remove_null_and_duplicate_records(
        self, data, columns, expected, column_mock_null, column_mock_names
    ):
        """Tests if null and duplicate values are removed."""
        column_mock_null.return_value = columns
        column_mock_names.return_value = columns
        actual = KaggleFeatureCleaner.remove_null_and_duplicate_records(
            data=data,
        )
        column_mock_null.assert_called()
        column_mock_names.assert_called()
        assert_frame_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {
                        "City": ["München", "Köln", "Kiev", ""],
                        "Position": [
                            "Project Manager",
                            "Big Data Engineer",
                            "Python Software Developer",
                            "",
                        ],
                        "Age": [35, 30, 22, 57],
                    }
                ),
                ["City", "Position"],
                pd.DataFrame(
                    {
                        "City": ["munich", "cologne", "kyiv", np.NaN],
                        "Position": [
                            "manager",
                            "data engineer",
                            "software developer",
                            np.NaN,
                        ],
                        "Age": [35, 30, 22, 57],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "Company_Size": ["50-100", "up to 10", "1000+"],
                        "Seniority": ["Entry Level", "Senior", "  "],
                    }
                ),
                ["Company_Size", "Seniority"],
                pd.DataFrame(
                    {
                        "Company_Size": ["1-100", "1-100", "1000+"],
                        "Seniority": ["junior", "senior", np.NaN],
                    }
                ),
            ),
        ]
    )
    @patch("preprocessing.clean_features.CleanedFeaturesSchema.get_category_columns")
    def test_unify_row_values(self, data, columns, expected, column_mock):
        """Tests if row values are unified by applying rules and mappings."""
        column_mock.return_value = columns
        actual = KaggleFeatureCleaner.unify_row_values(
            data=data,
        )
        column_mock.assert_called()
        assert_frame_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.Series([np.NaN, np.NaN], name="Timestamp", dtype="datetime64[ns]"),
                pd.Series([np.NaN, np.NaN], name="Year", dtype="float"),
            ),
            (
                pd.Series(
                    [pd.Timestamp("20220513"), pd.Timestamp("20200312")],
                    name="Timestamp",
                    dtype="datetime64[ns]",
                ),
                pd.Series([2022, 2020], name="Year", dtype="int"),
            ),
            (
                pd.Series(
                    [pd.Timestamp("20190603"), np.NaN],
                    name="Timestamp",
                    dtype="datetime64[ns]",
                ),
                pd.Series([2019, np.NaN], name="Year", dtype="float"),
            ),
        ]
    )
    def test_timestamp_to_year(self, timestamp, expected):
        """Tests the conversion of timestamp to year."""
        actual = KaggleFeatureCleaner.timestamp_to_year(
            timestamp=timestamp,
        )
        assert_series_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.Series([np.NaN, np.NaN], name="Position", dtype="object"),
                pd.Series([np.NaN, np.NaN], name="Seniority", dtype="object"),
                pd.Series([np.NaN, np.NaN], name="Position", dtype="object"),
            ),
            (
                pd.Series(
                    ["Senior Developer", "Python Engineer", "Junior Devops Engineer"],
                    name="Position",
                    dtype="object",
                ),
                pd.Series(
                    ["Senior", "Mid", "Junior"], name="Seniority", dtype="object"
                ),
                pd.Series(
                    ["Developer", "Python Engineer", "Devops Engineer"],
                    name="Position",
                    dtype="object",
                ),
            ),
            (
                pd.Series(
                    ["Expert Manager", np.NaN, "Principle Engineer"],
                    name="Position",
                    dtype="object",
                ),
                pd.Series(["Lead", "Senior", np.NaN], name="Seniority", dtype="object"),
                pd.Series(
                    ["Expert Manager", np.NaN, "Principle Engineer"],
                    name="Position",
                    dtype="object",
                ),
            ),
        ]
    )
    def test_clean_position_column(self, position, seniority, expected):
        """Tests cleaning of the position column."""
        actual = KaggleFeatureCleaner.clean_position_column(
            position=position,
            seniority=seniority,
        )
        assert_series_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.Series([np.NaN, np.NaN], dtype="object"),
                pd.Series([np.NaN, np.NaN], dtype="float"),
                pd.Series([np.NaN, np.NaN]),
            ),
            (
                pd.Series(["15,3", "12", "3,5", np.NaN], dtype="object"),
                pd.Series([35, 28, 25, 40], dtype="float"),
                pd.Series([15.3, 10, 3.5, np.NaN]),
            ),
            (
                pd.Series(["2.5", np.NaN, "17"], dtype="object"),
                pd.Series([18, np.NaN, np.NaN], dtype="float"),
                pd.Series([0, np.NaN, 17.0]),
            ),
        ]
    )
    def test_clean_years_of_experience_column(self, experience, age, expected):
        """Tests cleaning of the years of experience column."""
        actual = KaggleFeatureCleaner.clean_years_of_experience_column(
            years_of_experience=experience,
            age=age,
        )
        assert_series_equal(expected, actual)


class KaggleFeatureCleanerTrainTest(unittest.TestCase):
    """Test case for the kaggle survey data."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.cleaner = KaggleFeatureCleaner(
            data=RAW_DATA_COMBINED,
            mode=ExecutionMode.TRAIN,
        )

    def test_execute(self):
        """Tests full execution."""
        expected = CLEANED_FEATURES
        actual = self.cleaner.execute()
        assert_frame_equal(expected, actual)


class MainTest(unittest.TestCase):
    """Test case for the main method."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = "mocked.parquet"
        self.output_path = Path(self.temp_dir, "preprocessed.parquet")
        self.mode = "train"

    def tearDown(self) -> None:
        """Tears down written files."""
        shutil.rmtree(self.temp_dir)

    @patch("preprocessing.clean_features.read_data", return_value=RAW_DATA_COMBINED)
    def test_main(self, read_data_mock):
        """Test execution of the main method."""
        main(
            input_path=self.input_path,
            output_path=self.output_path.as_posix(),
            mode=self.mode,
        )
        read_data_mock.assert_called()
        self.assertTrue(self.output_path.is_file())


if __name__ == "__main__":
    unittest.main()
