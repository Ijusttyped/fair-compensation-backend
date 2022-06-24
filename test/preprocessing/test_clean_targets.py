"""Test cases for cleaning target values."""
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from test.resources.sample_data import RAW_DATA_COMBINED, CLEANED_TARGETS

import numpy as np
import pandas as pd
from parameterized import parameterized

from pandas.testing import assert_frame_equal

from preprocessing.clean_targets import KaggleTargetCleaner, main


class KaggleTargetCleanerTest(unittest.TestCase):
    """Test case for the kaggle survey data."""

    # pylint: disable=no-self-use

    def setUp(self) -> None:
        """Sets up training prerequisites."""
        self.cleaner = KaggleTargetCleaner(
            data=RAW_DATA_COMBINED,
        )

    def test_execute(self):
        """Tests the whole execution."""
        expected = CLEANED_TARGETS
        actual = self.cleaner.execute()
        assert_frame_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {
                        "Salary_Yearly": [
                            500000.0,
                            120000.0,
                            55000.0,
                            40000.0,
                            31000.0,
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "Salary_Yearly": [120000.0, 55000.0, 40000.0, 31000.0],
                    },
                    index=pd.Index([1, 2, 3, 4]),
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "Salary_Yearly": [1000000.0, np.NaN, 40000.0, 98000.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "Salary_Yearly": [1000000.0, 40000.0, 98000.0],
                    },
                    index=pd.Index([0, 2, 3]),
                ),
            ),
        ]
    )
    def test_remove_outliers(self, data, expected):
        """Tests if too high and too low values are removed."""
        actual = KaggleTargetCleaner.remove_outliers(data=data)
        assert_frame_equal(expected, actual)


class MainTest(unittest.TestCase):
    """Tests the main method."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = "mocked.parquet"
        self.output_path = Path(self.temp_dir, "cleaned.parquet")

    def tearDown(self) -> None:
        """Tears down written files."""
        shutil.rmtree(self.temp_dir)

    @patch("preprocessing.clean_targets.read_data", return_value=RAW_DATA_COMBINED)
    def test_main(self, read_data_mock):
        """Tests the main method."""
        main(
            input_path=self.input_path,
            output_path=self.output_path.as_posix(),
        )
        read_data_mock.assert_called()
        self.assertTrue(self.output_path.is_file())


if __name__ == "__main__":
    unittest.main()
