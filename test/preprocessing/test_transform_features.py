"""Test cases to test transformation of cleaned features."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import shutil
from test.resources.sample_data import CLEANED_FEATURES, TRANSFORMED_FEATURES

import pandas as pd
from parameterized import parameterized

import numpy as np
from pandas.testing import assert_frame_equal

from preprocessing.transform_features import KaggleFeatureTransformer, main
from utils.data_models import ExecutionMode


class KaggleFeatureTransformerTest(unittest.TestCase):
    """Test case for the kaggle survey data."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.transformer = KaggleFeatureTransformer(
            data=CLEANED_FEATURES,
            mode=ExecutionMode.TRAIN,
        )

    @parameterized.expand(
        [
            (
                None,
                TRANSFORMED_FEATURES,
            ),
            (
                {
                    "Gender": ["diverse", "female", "male"],
                    "City": ["berlin", "cologne"],
                    "Seniority": ["junior", "mid", "senior"],
                    "Position": [
                        "data scientist",
                        "engineer",
                        "manager",
                        "software developer",
                    ],
                    "Company_Size": ["1-100", "101-1000"],
                    "Company_Type": ["consulting or agency", "product", "startup"],
                },
                TRANSFORMED_FEATURES,
            ),
        ]
    )
    def test_execute(self, labels, expected):
        """Tests encoding of categorical features."""
        self.transformer.labels = labels
        actual = self.transformer.execute()
        assert_frame_equal(expected, actual)

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {
                        "Position": [np.NaN, np.NaN],
                    }
                ),
                {
                    "Position": ["engineer", "developer"],
                },
                pd.DataFrame(
                    {
                        "Position": [-1, -1],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "City": [np.NaN, np.NaN, np.NaN],
                        "Gender": ["female", np.NaN, "diverse"],
                    }
                ),
                {
                    "City": ["berlin", "amsterdam"],
                    "Gender": ["diverse", "female"],
                },
                pd.DataFrame(
                    {
                        "City": [-1, -1, -1],
                        "Gender": [1, -1, 0],
                    }
                ),
            ),
        ]
    )
    def test_encode_categorical_features(self, data, labels, expected):
        """Tests the edge case of encoding."""
        self.transformer.data = data
        actual = self.transformer.encode_categorical_features(labels=labels)
        assert_frame_equal(expected, actual)

    @parameterized.expand(
        [
            (np.NaN, ["label", "label2"], -1),
            ("not inside", ["random", "label"], 2),
            ("male", ["male", "female", "diverse"], 0),
        ]
    )
    def test_encode_based_on_list(self, value, labels, expected):
        """Tests if labels are encoded correctly on inference."""
        actual = KaggleFeatureTransformer.encode_based_on_list(
            value=value,
            label_values=labels,
        )
        self.assertEqual(expected, actual)


class MainTest(unittest.TestCase):
    """Test case for the main method."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = "mocked.parquet"
        self.output_path = Path(self.temp_dir, "transformed.parquet")
        self.mode = "train"

    def tearDown(self) -> None:
        """Tears down written files."""
        shutil.rmtree(self.temp_dir)

    @patch("preprocessing.transform_features.read_data", return_value=CLEANED_FEATURES)
    def test_transform_features_main(self, read_data_mock):
        """Tests the transform features main method."""
        main(
            input_path=self.input_path,
            output_path=self.output_path.as_posix(),
            mode=self.mode,
        )
        read_data_mock.assert_called()
        self.assertTrue(self.output_path.is_file())


if __name__ == "__main__":
    unittest.main()
