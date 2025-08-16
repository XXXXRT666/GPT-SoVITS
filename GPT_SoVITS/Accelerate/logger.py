import sys

from loguru import logger
from rich.console import Console
from rich.traceback import Traceback, install

install()


def rich_format(record):
    level = record["level"].name
    color = {
        "DEBUG": "green",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    }.get(level, "black")
    return f"[bold {color}][{level}][/bold {color}] {record['message']}"


def tb(show_locals: bool = True):
    exc_type, exc_value, exc_tb = sys.exc_info()
    assert exc_type
    assert exc_value
    tb = Traceback.from_exception(exc_type, exc_value, exc_tb, show_locals=show_locals)

    return tb


console = Console()

logger.remove()
logger.add(console.print, format=rich_format)

__all__ = ["logger", "console", "tb"]
