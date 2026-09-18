"""Owner-only writes for credential material kept on disk.

OAuth refresh tokens are long-lived and grant full order-placement rights, so
they must not land at the umask default (typically ``0644``, world-readable).
``Path.write_text`` gives no control over the mode, and setting it afterwards
leaves a window where the file exists with the wrong permissions — so the file
is created with ``O_CREAT`` and an explicit mode instead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Owner read/write only.
SECRET_FILE_MODE = 0o600
#: Owner read/write/execute only, for the containing directory.
SECRET_DIR_MODE = 0o700


def write_secret_file(path: Path | str, content: str) -> Path:
    """Write ``content`` to ``path`` readable only by the current user.

    Creates the parent directory with mode ``0o700`` if it does not exist.
    An existing file is truncated, and its mode is corrected in case it was
    created before this helper existed.

    Args:
        path: Destination file.
        content: Text to write.

    Returns:
        The resolved path that was written.

    Raises:
        OSError: If the file cannot be written. Callers that treat credential
            persistence as best-effort should catch this themselves.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=SECRET_DIR_MODE)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, SECRET_FILE_MODE)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except Exception:
        # fdopen takes ownership of the descriptor on success, so it is only
        # ours to close when fdopen itself failed.
        os.close(fd)
        raise
    with handle:
        handle.write(content)

    # O_CREAT does not change the mode of a file that already existed.
    try:
        os.chmod(path, SECRET_FILE_MODE)
    except OSError as e:
        logger.warning("Could not restrict permissions on %s: %s", path, e)

    return path
