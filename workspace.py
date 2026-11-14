"""The session workspace is the directory where fduplex was launched."""
import os
from pathlib import Path


def session_workspace():
    return Path(os.getenv('FDUPLEX_LAUNCH_DIR') or Path.cwd()).expanduser().resolve()
