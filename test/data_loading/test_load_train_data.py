""" Tests the data loading of the training data. """
import unittest
from test.resources.sample_data import FEATURES, TARGETS

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from data_loading.load_train_data import KaggleTrainDataLoader


class KaggleTrainDataLoaderTest(unittest.TestCase):
    """Tests the data loading of the kaggle training data."""

    def setUp(self) -> None:
        """Setup of the test prerequisites."""
        self.random_state = 26
        self.test_size = 0.2
        self.train_data_length = int(min([len(FEATURES), len(TARGETS)]) * (1 - 0.2))
        self.test_data_length = int(min([len(FEATURES), len(TARGETS)]) * 0.2)
        self.data_loader = KaggleTrainDataLoader(
            features=FEATURES,
            targets=TARGETS,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    # pylint: disable=no-self-use
    def test_match(self):
        """Tests if features and targets are merged together correctly."""
        expected = pd.DataFrame(
            {
                "Year": [2, 4, 5, -1, 3],
                "Age": [2, 4, 5, -1, 3],
                "Gender": [2, 4, 5, -1, 3],
                "City": [2, 4, 5, -1, 3],
                "Seniority": [2, 4, 5, -1, 3],
                "Position": [2, 4, 5, -1, 3],
                "Years_of_Experience": [2, 4, 5, -1, 3],
                "Company_Size": [2, 4, 5, -1, 3],
                "Company_Type": [2, 4, 5, -1, 3],
                "Salary_Yearly": [50000, 67000, 66000, 89000, 76000],
            }
        )
        actual = KaggleTrainDataLoader.match(features=FEATURES, targets=TARGETS)
        assert_frame_equal(expected, actual)

    def test_train_data(self):
        """Tests the generation of train data."""
        self.data_loader.setup()
        expected_x = FEATURES.iloc[:-2].iloc[::-1]
        expected_y = TARGETS.iloc[:-1].iloc[::-1]
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
        expected_x = FEATURES.iloc[-2]
        expected_y = TARGETS.iloc[-1:]
        X_test, y_test = self.data_loader.test_data()
        assert_series_equal(expected_x, X_test.squeeze())
        assert_series_equal(expected_y, y_test)

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
