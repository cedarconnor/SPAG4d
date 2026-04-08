"""Contains logging related utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure(log_level: int, log_path: Path | None = None, prefix: str | None = None) -> None:
    """Configure logger globally.

    Args:
        log_level: The desired verbosity level.
        log_path: The path to write logs to.
        prefix: The prefix of the logger.
    """
    logger = logging.getLogger(prefix)

    # Reset logger to initial state (e.g. to avoid side effects from imports).
    for handler in logger.handlers:
        logger.removeHandler(handler)

    for filter in logger.filters:
        logger.removeFilter(filter)

    # Set level.
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Set up console handler.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Set up file handler.
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
