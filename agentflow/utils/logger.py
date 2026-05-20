"""Logging configuration for AgentFlow."""

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging."""
    fmt = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)
