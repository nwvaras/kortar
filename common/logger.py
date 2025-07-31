"""
Centralized logging configuration using structlog.
"""

import structlog
import logging
import sys
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog with sensible defaults for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            # Add log level to event dict
            structlog.processors.add_log_level,
            # Add timestamp 
            structlog.processors.TimeStamper(fmt="iso"),
            # Filter out log records with logging level below this level
            structlog.stdlib.filter_by_level,
            # Perform %-style formatting
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add caller information (file, line, function)
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FILENAME,
                           structlog.processors.CallsiteParameter.LINENO,
                           structlog.processors.CallsiteParameter.FUNC_NAME]
            ),
            # Stack info and exception info
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            # Pretty print for development
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name (typically __name__ or module name)
        **context: Additional context to bind to the logger
        
    Returns:
        Configured structlog logger instance
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


# Configure logging on module import
configure_logging()