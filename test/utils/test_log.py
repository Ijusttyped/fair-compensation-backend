"""Test cases for testing logger setup."""
import unittest
import logging
from parameterized import parameterized

from utils import log


class LogTest(unittest.TestCase):
    """Test case for correct logger setup."""

    def tearDown(self) -> None:
        """Tears down test resources."""
        logging.shutdown()

    @parameterized.expand(
        [
            "default",
            "not_default",
        ]
    )
    def test_setup_custom_logger(self, name):
        """Tests if logger is registered with given name and contains handler."""
        log.setup_logger(name=name)
        manager = logging.Logger.manager.loggerDict
        self.assertTrue(name in manager)
        self.assertIsInstance(manager[name].handlers[0], logging.StreamHandler)


if __name__ == "__main__":
    unittest.main()
