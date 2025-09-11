"""Configure the root logger."""

from __future__ import annotations

import atexit
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
import time
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Literal, override

from pydantic import BaseModel, Field, ValidationError

from project_name.config import LOGGER_CONFIG_FILE, PACKAGE_NAME

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class FormatKeys(BaseModel):
    level: str | None = Field(..., alias="levelname")
    message: str | None
    timestamp: str | None
    logger: str | None = Field(..., alias="name")
    module: str | None
    function: str | None = Field(..., alias="funcName")
    line_number: int | None = Field(..., alias="lineno")
    thread_name: str | None = Field(..., alias="threadName")
    model_config = {
        "extra": "allow",
    }


class FormatterConfig(BaseModel):
    factory: str | None = Field('logging.Formatter', alias="()")
    format: str | None
    datefmt: str | None
    # fmt_keys: FormatKeys | None = None
    model_config = {
        "extra": "allow",
    }


class HandlerConfig(BaseModel):
    class_: str = Field(..., alias="class")
    level: LogLevel | None
    formatter: str | None
    stream: str | None
    filename: str | None
    max_bytes: int | None = Field(..., alias="maxBytes")
    backup_count: int | None = Field(..., alias="backupCount")
    handlers: list[str] | None
    respect_handler_level: bool



class LoggerConfig(BaseModel):
    level: LogLevel
    handlers: list[str]


class LoggingConfig(BaseModel):
    version: int
    disable_existing_loggers: bool = False
    formatters: dict[str, FormatterConfig]
    handlers: dict[str, HandlerConfig]
    loggers: dict[str, LoggerConfig]
    root: LoggerConfig | None


STDOUT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
FILE_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(padded_module)s:L%(padded_lineno)s | %(padded_funcName)s: %(message)s"

ROOT_LOGGER_NAME = PACKAGE_NAME

__all__ = ("ROOT_LOGGER_NAME", "setup_logging")


RecordAttrs = Enum(
    "RecordAttrs",
    (
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName"
    ),
    type=str,
)


class ColouredFormatter(logging.Formatter):
    """Coloured log formatter."""

    @staticmethod
    def _utc_time(timestamp: float | None = None) -> time.struct_time:
        """Enforce UTC timestamps regardless of local timezone."""
        return time.gmtime(timestamp)

    converter = _utc_time

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        log_level_colours = {
            logging.CRITICAL: "\033[31;1;40m",  # Red, bold
            logging.ERROR: "\033[31;40m",  # Red
            logging.WARNING: "\033[33;40m",  # Yellow
            logging.INFO: "\033[32;40m",  # Green
            logging.DEBUG: "\033[36;40m",  # Cyan
        }
        reset = "\033[0m"
        colour = log_level_colours.get(record.levelno, reset)

        record.levelname = f"{record.levelname:^8}"

        # Save original values to restore later
        original_msg = record.msg
        original_levelname = record.levelname

        # Apply colour
        record.msg = f"{colour}{original_msg}{reset}"
        record.levelname = f"{colour}{original_levelname}{reset}"

        # Format the record with the temporary values
        formatted_message = super().format(record)

        # Restore the original values to avoid side-effects
        record.msg = original_msg
        record.levelname = original_levelname

        return formatted_message


class CustomQueueHandler(logging.handlers.QueueHandler):
    """Custom queue handler."""

    @override
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        queue_handler = logging.getHandlerByName("queue_handler")
        if queue_handler is None:
            super().__init__(*args, **kwargs)


class JSONLogFormatter(logging.Formatter):
    """Custom JSON log formatter."""

    @override
    def __init__(self, *, fmt_keys: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        log_dict = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC),
        }

        for key, value in self.fmt_keys.items():
            log_dict[key] = getattr(record, value)

        if record.exc_info:
            log_dict["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_dict["stack_info"] = self.formatStack(record.stack_info)

        standard_attrs = set(RecordAttrs.__members__)
        standard_attrs.update(log_dict.keys())

        extra_attrs = record.__dict__.keys() - log_dict.keys() - standard_attrs

        for key in extra_attrs:
            log_dict[key] = record.__dict__[key]

        return log_dict


def setup_logging() -> None:
    """Set up logging."""
    (Path.cwd() / "logs").mkdir(exist_ok=True)
    logger_data = LOGGER_CONFIG_FILE.read_text(encoding="utf-8")
    logging_config = tomllib.loads(logger_data)

    try:
        validated_config = LoggingConfig.model_validate(logging_config)
    except ValidationError:
        print(f"Error: Invalid logging config in {LOGGER_CONFIG_FILE}")  # noqa: T201
        raise

    logging.config.dictConfig(validated_config.model_dump(by_alias=True))

    queue_handler = logging.getHandlerByName("queue_handler")

    if not isinstance(queue_handler, logging.handlers.QueueHandler):
        return

    if not (listener := getattr(queue_handler, "listener", None)):
        return

    listener.start()
    atexit.register(listener.stop)
