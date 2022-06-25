"""Test cases for transforming the target values."""
import unittest
from unittest.mock import patch
import tempfile
import shutil
from test.resources.sample_data import CLEANED_TARGETS, TRANSFORMED_TARGETS
from pathlib import Path

from pandas.testing import assert_frame_equal

from preprocessing.transform_targets import KaggleTargetTransformer, main


class KaggleTargetTransformerTest(unittest.TestCase):
    """Test case for the kaggle survey data."""

    def setUp(self) -> None:
        self.transformer = KaggleTargetTransformer(
            data=CLEANED_TARGETS,
        )

    def test_execute(self):
        """Tests the whole execution"""
        expected = TRANSFORMED_TARGETS
        actual = self.transformer.execute()
        assert_frame_equal(expected, actual)


class MainTest(unittest.TestCase):
    """Test case for the main method."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = "mocked.parquet"
        self.output_path = Path(self.temp_dir, "transformed.parquet")

    def tearDown(self) -> None:
        """Tears down written files."""
        shutil.rmtree(self.temp_dir)

    @patch("preprocessing.transform_targets.read_data", return_value=CLEANED_TARGETS)
    def test_transform_targets_main(self, read_data_mock):
        """Tests the transform targets main method."""
        main(
            input_path=self.input_path,
            output_path=self.output_path.as_posix(),
        )
        read_data_mock.assert_called()
        self.assertTrue(self.output_path.is_file())


if __name__ == "__main__":
    unittest.main()
