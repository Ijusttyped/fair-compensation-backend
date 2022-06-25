""" Tests the data loading of the training data. """
import unittest
from test.resources.sample_data import (
    TRANSFORMED_FEATURES,
    TRANSFORMED_TARGETS,
    TRAIN_DATA,
)

from pandas.testing import assert_frame_equal, assert_series_equal

from data_loading.load_train_data import KaggleTrainDataLoader


class KaggleTrainDataLoaderTest(unittest.TestCase):
    """Tests the data loading of the kaggle training data."""

    # pylint: disable=too-many-instance-attributes,no-member
    def setUp(self) -> None:
        """Setup of the test prerequisites."""
        self.random_state = 26
        self.test_size = 0.2
        self.features = TRANSFORMED_FEATURES
        self.targets = TRANSFORMED_TARGETS
        self.train_data = TRAIN_DATA
        self.train_data_length = int(
            min([len(self.features), len(self.targets)]) * (1 - 0.2)
        )
        self.test_data_length = int(min([len(self.features), len(self.targets)]) * 0.2)
        self.data_loader = KaggleTrainDataLoader(
            features=self.features,
            targets=self.targets,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def test_match(self):
        """Tests if features and targets are merged together correctly."""
        expected = self.train_data
        actual = KaggleTrainDataLoader.match(
            features=self.features, targets=self.targets
        )
        assert_frame_equal(expected, actual, check_dtype=False)

    def test_train_data(self):
        """Tests the generation of train data."""
        self.data_loader.setup()
        expected_x = self.features.loc[[1, 3]]
        expected_y = self.targets.loc[[1, 3]].squeeze()
        X_train, y_train = self.data_loader.train_data()
        assert_frame_equal(expected_x, X_train)
        assert_series_equal(expected_y, y_train)

    def test_train_data_without_setup(self):
        """Tests the generation of train data."""
        X_train, y_train = self.data_loader.train_data()
        self.assertEqual(None, X_train)
        self.assertEqual(None, y_train)

    def test_test_data(self):
        """Tests the generation of test data."""
        self.data_loader.setup()
        expected_x = self.features.loc[[4]]
        expected_y = self.targets.loc[[4]].squeeze(axis=1)
        X_test, y_test = self.data_loader.test_data()
        assert_frame_equal(expected_x, X_test)
        assert_series_equal(expected_y, y_test, check_names=False)

    def test_test_data_without_setup(self):
        """Tests the generation of test data."""
        X_test, y_test = self.data_loader.test_data()
        self.assertEqual(None, X_test)
        self.assertEqual(None, y_test)

    def test_val_data(self):
        """Tests the generation of validation data."""
        with self.assertRaises(NotImplementedError):
            _, _ = self.data_loader.val_data()


if __name__ == "__main__":
    unittest.main()
