""" Tests for loading the raw data. """
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from test.resources.sample_data import RAW_DATA_1, RAW_DATA_2, RAW_DATA_COMBINED

from parameterized import parameterized
import pandas as pd
from pandas.testing import assert_frame_equal
from pandera.errors import SchemaError

from data_loading.load_raw_data import KaggleRawDataLoader, main


class KaggleRawDataLoaderTest(unittest.TestCase):
    """Test case for loading the raw kaggle data."""

    def setUp(self) -> None:
        """Setup of the test prerequisites."""
        self.input_paths = ["dummy.csv", "dummy2.csv"]
        self.data_loader = KaggleRawDataLoader(input_paths=self.input_paths)

    @patch("data_loading.load_raw_data.read_data", side_effect=[RAW_DATA_1, RAW_DATA_2])
    def test_load(self, read_data_mock):
        """Test the loading."""
        expected = RAW_DATA_COMBINED
        actual = self.data_loader.load()
        read_data_mock.assert_called()
        assert_frame_equal(expected, actual)

    @patch(
        "data_loading.load_raw_data.read_data",
        return_value=pd.DataFrame(
            {
                "Salary_Yearly": [50000, 52000, 63233],
                "Age": [23, 40, 33],
            }
        ),
    )
    def test_load_raises(self, read_data_mock):
        """Test if the loading raises an error if columns are missing."""
        with self.assertRaises(SchemaError):
            _ = self.data_loader.load()
            read_data_mock.assert_called()

    @parameterized.expand(
        [
            (pd.DataFrame(), pd.DataFrame()),
            (
                pd.DataFrame(
                    {
                        "Some column": ["some", "values"],
                        "Current Salary": [100000, 50000],
                    }
                ),
                pd.DataFrame(
                    {
                        "Some column": ["some", "values"],
                        "Salary_Yearly": [100000, 50000],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "Zeitstempel": ["22.01.2022", "29.08.2021"],
                        "Company size": ["100", "50-100"],
                    }
                ),
                pd.DataFrame(
                    {
                        "Timestamp": ["22.01.2022", "29.08.2021"],
                        "Company_Size": ["100", "50-100"],
                    }
                ),
            ),
        ]
    )
    def test_rename_columns(self, data, expected):
        """Test if renaming of columns works."""
        actual = self.data_loader.rename_columns(data)
        assert_frame_equal(expected, actual)


class MainTest(unittest.TestCase):
    """Tests the main method of the raw data loading."""

    def setUp(self) -> None:
        """Setup of the test prerequisites."""
        self.input_paths = ["dummy1.csv", "dummy2.csv"]
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir, "out.parquet")

    def tearDown(self) -> None:
        """Tear down the written data."""
        shutil.rmtree(self.temp_dir)

    @patch("data_loading.load_raw_data.read_data", side_effect=[RAW_DATA_1, RAW_DATA_2])
    def test_main(self, read_data_mock):
        """Tests the main method."""
        main(
            input_paths=self.input_paths,
            output_path=self.output_path.as_posix(),
        )
        read_data_mock.assert_called()
        self.assertTrue(self.output_path.is_file())


if __name__ == "__main__":
    unittest.main()
