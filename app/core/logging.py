"""
Centralized logging config - Based on Loguru with user context support
"""
import sys
from pathlib import Path
from loguru import logger


class UserContextFilter:
    """User context filter - injects user_id into log records"""

    def filter(self, record):
        try:
            from app.core.context import get_user_id
            user_id = get_user_id()
            record["user_id"] = user_id if user_id is not None else "-"
        except ImportError:
            record["user_id"] = "-"
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure Loguru logging system"""
    logger.remove()

    user_filter = UserContextFilter()

    # Console output
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<yellow>uid:{user_id}</yellow> - "
            "<level>{message}</level>"
        ),
        filter=user_filter.filter,
        colorize=True,
    )

    # File output
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "super_agent_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | uid:{user_id} | "
            "{message}"
        ),
        filter=user_filter.filter,
    )

    logger.info("Logging initialized with user context")


def get_logger(name: str = None):
    """Get logger instance bound to module name"""
    if name:
        return logger.bind(name=name)
    return logger