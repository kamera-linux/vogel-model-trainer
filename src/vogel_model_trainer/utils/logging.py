"""
Console log capture for ``--log``.

Output is mirrored to a log file in::

    /var/log/vogel-kamera-linux/YYYY/KWxx/<timestamp>_<command>.log

If that directory is not writable (normal for non-root users), the log is
written to ``$XDG_STATE_HOME/vogel-model-trainer/logs/YYYY/KWxx/`` instead
(default ``~/.local/state/...``). ``VOGEL_LOG_DIR`` overrides the base directory.
"""

import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

SYSTEM_LOG_DIR = Path('/var/log/vogel-kamera-linux')


class Tee:
    """Write to several streams at once (console and log file)."""

    def __init__(self, primary, *others):
        self.primary = primary
        self.files = (primary,) + others

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
        return len(data)

    def flush(self):
        for f in self.files:
            f.flush()

    # Libraries such as tqdm and transformers query these on sys.stdout/stderr.
    def isatty(self):
        return getattr(self.primary, 'isatty', lambda: False)()

    def fileno(self):
        return self.primary.fileno()

    @property
    def encoding(self):
        return getattr(self.primary, 'encoding', 'utf-8')

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _user_log_dir() -> Path:
    state = os.environ.get('XDG_STATE_HOME') or str(Path.home() / '.local' / 'state')
    return Path(state) / 'vogel-model-trainer' / 'logs'


def _candidate_base_dirs():
    override = os.environ.get('VOGEL_LOG_DIR')
    if override:
        return [Path(override).expanduser()]
    return [SYSTEM_LOG_DIR, _user_log_dir()]


def open_log_file(command: str) -> Path:
    """Create the log file for ``command`` and return its path."""
    now = datetime.now()
    sub = Path(now.strftime('%Y')) / f"KW{now.strftime('%V')}"
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{command}.log"

    last_error: Optional[Exception] = None
    for base in _candidate_base_dirs():
        try:
            log_dir = base / sub
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / filename
            path.touch()
            return path
        except OSError as exc:
            last_error = exc
    raise PermissionError(f"No writable log directory found: {last_error}")


@contextlib.contextmanager
def capture_output(command: str, enabled: bool = True) -> Iterator[Optional[Path]]:
    """Mirror stdout/stderr into a log file while the block runs."""
    if not enabled:
        yield None
        return

    from vogel_model_trainer.i18n import _

    path = open_log_file(command)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(path, 'w', encoding='utf-8') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(_('log_file', path=str(path)))
            yield path
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
