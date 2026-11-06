"""
Centralized logging configuration for iTemplate2.

Usage in any module:
    import logging
    logger = logging.getLogger('iTemplate2')

Call ``setup_project_file_logging(project_path)`` once when a project is
opened (or switched) to route log output to ``<project>/processing_log.txt``.
"""

import logging
import os

# ---------------------------------------------------------------------------
# Package-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger('iTemplate2')
logger.setLevel(logging.DEBUG)  # allow all levels; handlers filter further

# Prevent duplicate handlers if this module is re-imported
if not logger.handlers:
    _fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler — mirrors what print() used to do
    _console = logging.StreamHandler()
    _console.setLevel(logging.DEBUG)
    _console.setFormatter(_fmt)
    logger.addHandler(_console)

# Tag used to find and replace the project file handler
_FILE_HANDLER_TAG = '_project_file_handler'


def setup_project_file_logging(project_path: str) -> None:
    """
    Add (or replace) a FileHandler that writes to
    ``<project_path>/processing_log.txt``.

    Called from ``MainWindow.on_project_changed()`` whenever the user
    opens or switches projects.
    """
    # Remove any previous project file handler
    for h in logger.handlers[:]:
        if getattr(h, '_tag', None) == _FILE_HANDLER_TAG:
            h.close()
            logger.removeHandler(h)

    if not project_path or not os.path.isdir(project_path):
        return

    log_path = os.path.join(project_path, 'processing_log.txt')
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    fh._tag = _FILE_HANDLER_TAG  # type: ignore[attr-defined]
    logger.addHandler(fh)
    logger.info('Logging started for project: %s', project_path)
