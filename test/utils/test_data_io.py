"""Test cases for reading and writing data."""
import shutil
import tempfile
import unittest
from pathlib import Path
from parameterized import parameterized

import pandas as pd
from pandas.testing import assert_frame_equal


from utils import data_io


class DataIOTest(unittest.TestCase):
    """Test case for data input and output."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.temp_dir = tempfile.mkdtemp()
        self.resources_dir = Path("test/resources")

    def tearDown(self) -> None:
        """Tears down written files"""
        shutil.rmtree(self.temp_dir)

    @parameterized.expand(
        [
            ("sample.json", {"key1": "value1", "value2": {"nested": "pair"}}),
            (
                "sample.parquet",
                pd.DataFrame(
                    {
                        "feature_1": [1, 2, 3],
                        "feature_2": ["some", "test", "values"],
                    }
                ),
            ),
        ]
    )
    def test_read_data(self, filename, expected):
        """Tests reading in data."""
        actual = data_io.read_data(filepath=self.resources_dir / filename)
        if isinstance(actual, dict):
            self.assertDictEqual(expected, actual)
        elif isinstance(actual, pd.DataFrame):
            assert_frame_equal(expected, actual)

    def test_read_data_raises(self):
        """Tests if read data raises an error if file format not supported."""
        with self.assertRaises(ValueError):
            _ = data_io.read_data(filepath="not_supported.tsv")

    @parameterized.expand(
        [
            (
                {
                    "key": "value",
                    "key2": 3,
                },
                "sample_file.json",
            ),
            (
                pd.DataFrame({"testcol": [1, 2, 3], "testcol2": [4, 5, 6]}),
                "sample_file.parquet",
            ),
        ]
    )
    def test_write_date(self, data, filename):
        """Tests if data is written to disk."""
        filepath = Path(self.temp_dir, filename)
        data_io.write_data(data=data, filepath=filepath.as_posix())
        self.assertTrue(filepath.is_file())


if __name__ == "__main__":
    unittest.main()
