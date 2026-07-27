"""
palm_preproc.log
----------------
Logging for the palm_preproc pipeline, styled after PALM-GeM's logger
(Institute of Computer Science of the CAS; Martin Bures, Jaroslav Resler):
timestamped lines, bold `progress()` step announcements, indented detail
lines, level tags only for warnings and errors, and lazy '{}' message
formatting (progress('Creating {}', name)).

Line format:
    14:03:17  Generating domain rectangles          <- progress (bold)
    14:03:17    [domains] child: 504 x 384 pts ...  <- info (indented)
    14:03:17    detail in grey                      <- debug
    14:03:17  WARNING  [merge] ...                  <- warnings/errors tagged

--log-datetime switches the timestamp to full date+time; -v/-q select
DEBUG/WARNING verbosity.
"""

import logging
import sys

# ------------------------------
# 1. ANSI COLOURS (disabled automatically on non-TTY streams)
# ------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"

_LEVEL_TAGS = {
    logging.WARNING:  ("WARNING", _YELLOW),
    logging.ERROR:    ("ERROR  ", _RED),
    logging.CRITICAL: ("CRITICAL", _RED + _BOLD),
}


class _PalmFormatter(logging.Formatter):
    def __init__(self, use_colour=True, log_datetime=False):
        super().__init__()
        self._use_colour = use_colour
        self._datefmt = "%Y-%m-%d %H:%M:%S" if log_datetime else "%H:%M:%S"

    def _c(self, code, text):
        return f"{code}{text}{_RESET}" if self._use_colour else text

    def format(self, record):
        ts = self._c(_GREY, self.formatTime(record, self._datefmt))
        msg = record.getMessage()

        if record.levelno >= logging.WARNING:
            tag, colour = _LEVEL_TAGS.get(record.levelno,
                                          ("WARNING", _YELLOW))
            line = f"{ts}  {self._c(colour, tag)}  {msg}"
        elif getattr(record, "progress", False):
            line = f"{ts}  {self._c(_BOLD + _CYAN, msg)}"
        elif record.levelno <= logging.DEBUG:
            line = f"{ts}    {self._c(_GREY, msg)}"
        else:
            line = f"{ts}    {msg}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ------------------------------
# 2. SETUP
# ------------------------------
def setup_logging(verbosity="info", log_datetime=False):
    """Initialise and return the palm_preproc logger.

    verbosity : "debug" | "info" | "warning"
    log_datetime : bool - full date in the timestamp instead of HH:MM:SS.
    """
    level = {"debug": logging.DEBUG, "info": logging.INFO,
             "warning": logging.WARNING, "warn": logging.WARNING
             }.get(str(verbosity).lower(), logging.INFO)

    logger = logging.getLogger("palm_preproc")
    logger.setLevel(level)
    if logger.handlers:            # avoid duplicates on re-init
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_PalmFormatter(use_colour=sys.stderr.isatty(),
                                        log_datetime=log_datetime))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger():
    """Return the palm_preproc logger, initialising defaults if needed."""
    logger = logging.getLogger("palm_preproc")
    if not logger.handlers:
        return setup_logging()
    return logger


# ------------------------------
# 3. PALM-GeM-STYLE MESSAGE HELPERS ('{}' lazy formatting)
# ------------------------------
def _fmt(msg, args):
    return msg.format(*args) if args else msg


def progress(msg, *args):
    """Bold step announcement, e.g. progress('Creating grid')."""
    get_logger().info(_fmt(msg, args), extra={"progress": True})


def info(msg, *args):
    get_logger().info(_fmt(msg, args))


def debug(msg, *args):
    get_logger().debug(_fmt(msg, args))


def warning(msg, *args):
    get_logger().warning(_fmt(msg, args))


def error(msg, *args):
    get_logger().error(_fmt(msg, args))
