"""
Production-grade structured logging for CLDB.
"""
import logging
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from functools import wraps
from contextvars import ContextVar
import threading
import uuid


request_id: ContextVar[str] = ContextVar('request_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')


class JSONFormatter(logging.Formatter):
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if request_id.get():
            log_data["request_id"] = request_id.get()
        if user_id.get():
            log_data["user_id"] = user_id.get()
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        if self.include_extra and hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        log_data["process_id"] = record.process
        log_data["thread_id"] = record.thread
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{color}{timestamp} [{record.levelname:8}] {record.name}: {record.getMessage()}{self.RESET}"


class CLDBLogger:
    _instances: Dict[str, logging.Logger] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str = "cldb", level: str = "INFO", json_format: bool = False):
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.json_format = json_format or level == "production"
        with self._lock:
            if name not in self._instances:
                self._instances[name] = self._create_logger()
    
    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.propagate = False
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.level)
        if self.json_format:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
        return logger
    
    def _extra_dict(self, kwargs) -> Dict[str, Any]:
        return kwargs if kwargs else {}
    
    def debug(self, message: str, **kwargs):
        self._instances[self.name].debug(message, extra=self._extra_dict(kwargs))
    
    def info(self, message: str, **kwargs):
        self._instances[self.name].info(message, extra=self._extra_dict(kwargs))
    
    def warning(self, message: str, **kwargs):
        self._instances[self.name].warning(message, extra=self._extra_dict(kwargs))
    
    def error(self, message: str, **kwargs):
        self._instances[self.name].error(message, extra=self._extra_dict(kwargs))
    
    def critical(self, message: str, **kwargs):
        self._instances[self.name].critical(message, extra=self._extra_dict(kwargs))
    
    def exception(self, message: str, **kwargs):
        self._instances[self.name].exception(message, extra=self._extra_dict(kwargs))
    
    def child(self, name: str) -> 'CLDBLogger':
        child_name = f"{self.name}.{name}"
        return CLDBLogger(child_name, level=logging.getLevelName(self.level), json_format=self.json_format)


def get_logger(name: str = "cldb", level: Optional[str] = None) -> CLDBLogger:
    import os
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    json_format = os.environ.get("LOG_FORMAT", "text").lower() == "json"
    return CLDBLogger(name, level=level, json_format=json_format)


class RequestContext:
    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
    
    def __enter__(self):
        request_id.set(self.request_id)
        if self.user_id:
            user_id.set(self.user_id)
        return self
    
    def __exit__(self, *args):
        request_id.set('')
        user_id.set('')


def log_execution_time(logger: CLDBLogger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            logger.info(f"Starting {func.__name__}", function=func.__name__)
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds()
                logger.info(f"Completed {func.__name__}", function=func.__name__, duration_ms=f"{duration * 1000:.2f}")
                return result
            except Exception as e:
                duration = (datetime.now() - start).total_seconds()
                logger.error(f"Failed {func.__name__}", function=func.__name__, duration_ms=f"{duration * 1000:.2f}", error=str(e))
                raise
        return wrapper
    return decorator