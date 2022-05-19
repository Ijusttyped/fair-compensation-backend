""" Logging setup """
import logging
import os


LOG_LEVELS = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
}


def setup_logger(name: str = "default") -> None:
    """
    Sets up a custom logging configuration.
    Args:
        name (str, optional): Name of the logger object.

    Returns:
        None.
    """
    logger = logging.getLogger(name)
    log_level = LOG_LEVELS.get(os.getenv("LOGLEVEL", "INFO"))
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(module)s %(funcName)s line %(lineno)d %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
