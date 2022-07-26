import logging

LOGGER_NAME = "skin_parser"


def setup_main_logger() -> logging.Logger:
    logging_level = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging_level)

    file_handler = logging.FileHandler("logger.txt")
    file_handler.setLevel(logging_level)
    logger.addHandler(file_handler)

    return logger