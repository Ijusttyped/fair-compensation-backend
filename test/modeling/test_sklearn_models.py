""" Test cases for the SKLearn models. """
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from modeling.sklearn_models import SKLearnModel


class SKLearnTest(unittest.TestCase):
    """Test case for basic SKLearn model."""

    def setUp(self) -> None:
        """Sets up the prerequisites"""
        self.temp_dir = tempfile.mkdtemp()
        self.model = SKLearnModel()
        self.out_file = Path(self.temp_dir, "sample.joblib")

    def tearDown(self) -> None:
        """Tears down written files"""
        shutil.rmtree(self.temp_dir)

    def test_fit(self):
        """Tests fit method."""
        before_fit_model = self.model.model
        self.model.fit(
            X=pd.DataFrame(
                {
                    "feature_1": [1, 2, 3, 4],
                    "feature_2": [1, 3, 5, 7],
                }
            ),
            y=pd.Series([10, 11, 12, 13], name="target"),
        )
        self.assertFalse(
            assert_array_equal(
                before_fit_model.feature_importances_,
                self.model.model.feature_importances_,
            )
        )

    @patch("modeling.sklearn_models.SKLearnModel.predict")
    def test_predict(self, model_mock):
        """Tests the prediction."""
        expected = np.ones((5, 5))
        model_mock.return_value = expected
        actual = self.model.predict(pd.DataFrame())
        assert_array_equal(expected, actual)

    @patch("modeling.sklearn_models.read_data", return_value=True)
    def test_load(self, read_data_mock):
        """Tests loading a stored model."""
        self.model.load(filename="mock.joblib")
        read_data_mock.assert_called()
        self.assertTrue(self.model.model)

    def test_save(self):
        """Tests persisting the model."""
        self.model.save(filename=self.out_file.as_posix())
        self.assertTrue(self.out_file.is_file())


if __name__ == "__main__":
    unittest.main()
