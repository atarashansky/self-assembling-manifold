"""Logging infrastructure for the SAM algorithm.

This module provides a centralized logging configuration for the samalg package.
All logging throughout the package should use the logger obtained from get_logger().
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Package-level logger name
LOGGER_NAME = "samalg"

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for the samalg package.

    Parameters
    ----------
    name : str | None, optional
        The name for the logger. If None, returns the root samalg logger.
        If provided, returns a child logger (e.g., 'samalg.sam').

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Examples
    --------
    >>> logger = get_logger()  # Returns 'samalg' logger
    >>> logger = get_logger('sam')  # Returns 'samalg.sam' logger
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    stream: bool = True,
) -> None:
    """Configure logging for the samalg package.

    Parameters
    ----------
    level : int | str, optional
        The logging level. Can be an integer (e.g., logging.DEBUG) or
        a string (e.g., 'DEBUG'). Default is logging.INFO.
    format_string : str | None, optional
        The format string for log messages. If None, uses SIMPLE_FORMAT.
    stream : bool, optional
        Whether to add a StreamHandler to output to stderr. Default is True.

    Examples
    --------
    >>> setup_logging(level='DEBUG')
    >>> setup_logging(level=logging.WARNING, format_string='%(message)s')
    """
    logger = get_logger()

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    if stream:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        formatter = logging.Formatter(format_string or SIMPLE_FORMAT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)


def set_verbosity(verbose: bool) -> None:
    """Set the verbosity level of the samalg logger.

    Parameters
    ----------
    verbose : bool
        If True, set logging level to INFO.
        If False, set logging level to WARNING.

    Examples
    --------
    >>> set_verbosity(True)   # Show INFO messages
    >>> set_verbosity(False)  # Only show WARNING and above
    """
    logger = get_logger()
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


# Initialize the logger with default settings
# This ensures the logger exists even if setup_logging is not called
_logger = get_logger()
if not _logger.handlers:
    # Only add handler if none exist (avoids duplicate handlers)
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# Convenience: export the main logger directly
logger = get_logger()
