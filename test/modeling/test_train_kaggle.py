""" Test cases for training models with the kaggle data. """
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from test.data.sample_data import FEATURES, TARGETS

import numpy as np
from numpy.testing import assert_array_equal

from modeling.train_kaggle import KaggleSurveyTrainer, main
from modeling.sklearn_models import SKLearnModel
from data_loading.load_train_data import KaggleTrainDataLoader


class KaggleSurveyTest(unittest.TestCase):
    """Simple test case for the kaggle survey."""

    def setUp(self) -> None:
        """Sets up test prerequisites."""
        self.trainer = KaggleSurveyTrainer(
            model=SKLearnModel(),
            data_loader=KaggleTrainDataLoader(
                features=FEATURES,
                targets=TARGETS,
            ),
        )

    def test_train(self):
        """Tests the training of a model."""
        model_before_training = self.trainer.model.model
        self.trainer.train()
        self.assertFalse(
            assert_array_equal(
                model_before_training.feature_importances_,
                self.trainer.model.model.feature_importances_,
            )
        )

    @patch(
        "modeling.train_kaggle.SKLearnModel.predict",
        side_effect=[np.ones((5, 5)), np.ones((5, 5))],
    )
    @patch.object(
        KaggleSurveyTrainer,
        "_KaggleSurveyTrainer__get_metric_calculation",
        side_effect=lambda m: lambda x, y: 0.0,
    )
    def test_evaluate(self, class_mock, predict_mock):
        """Tests the evaluation of a trained model."""
        expected = {"train": {"mae": 0.0, "mse": 0.0}, "test": {"mae": 0.0, "mse": 0.0}}
        actual = self.trainer.evaluate(metrics=["mae", "mse"])
        predict_mock.assert_called()
        class_mock.assert_called()
        self.assertDictEqual(expected, actual)


class MainTrainingExecutionTest(unittest.TestCase):
    """Test case for executing the main method."""

    def setUp(self) -> None:
        """Sets up training prerequisites."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir, "model.joblib")
        self.metrics_path = Path(self.temp_dir, "metrics.json")

    @patch("modeling.train_kaggle.read_data", side_effect=[FEATURES, TARGETS])
    def test_main(self, read_data_mock):
        """Tests the main method."""
        main(
            feature_path="mocked.csv",
            target_path="mocked.csv",
            model_path=self.model_path.as_posix(),
            metrics_path=self.metrics_path.as_posix(),
        )
        read_data_mock.assert_called()
        self.assertTrue(self.model_path.is_file())
        self.assertTrue(self.metrics_path.is_file())


if __name__ == "__main__":
    unittest.main()
