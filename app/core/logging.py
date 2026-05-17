"""
集中日志配置 - 基于 Loguru
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """配置 Loguru 日志系统"""
    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    )

    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "super_agent_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )

    logger.info("Logging initialized")


def get_logger(name: str = None):
    """获取绑定了模块名的 logger 实例"""
    if name:
        return logger.bind(name=name)
    return logger
