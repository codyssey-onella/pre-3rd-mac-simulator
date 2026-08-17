"""Mini NPU Simulator용 파일 로거 (표준 라이브러리 logging 사용)."""

import logging
from datetime import datetime
from pathlib import Path

LOG_DIR: Path = Path("logs")
_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "mini_npu") -> logging.Logger:
    """이름별 로거를 반환한다. 최초 호출 시 logs/ 폴더에 파일 핸들러를 붙인다."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file: Path = LOG_DIR / f"mini_npu_{datetime.now().strftime('%Y%m%d')}.log"

    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler: logging.FileHandler = logging.FileHandler(
            log_file, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
