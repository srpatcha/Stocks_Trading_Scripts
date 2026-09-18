"""Tests for owner-only credential file writes.

OAuth refresh tokens are long-lived and grant full order-placement rights.
They were written with Path.write_text, which lands at the umask default
(typically 0644, world-readable).
"""

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.utils.secret_file import (
    SECRET_DIR_MODE,
    SECRET_FILE_MODE,
    write_secret_file,
)


class TestWriteSecretFile:
    def test_writes_the_content(self, tmp_path):
        target = tmp_path / "token.txt"
        write_secret_file(target, "s3cret-refresh-token")
        assert target.read_text(encoding="utf-8") == "s3cret-refresh-token"

    def test_file_is_owner_only(self, tmp_path):
        target = tmp_path / "token.txt"
        write_secret_file(target, "s3cret")
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == SECRET_FILE_MODE, f"expected 0o600, got {oct(mode)}"

    def test_group_and_other_cannot_read(self, tmp_path):
        target = tmp_path / "token.txt"
        write_secret_file(target, "s3cret")
        mode = target.stat().st_mode
        assert not mode & stat.S_IRGRP
        assert not mode & stat.S_IROTH
        assert not mode & stat.S_IWGRP
        assert not mode & stat.S_IWOTH

    def test_creates_missing_parent_directory(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "token.txt"
        write_secret_file(target, "s3cret")
        assert target.exists()
        assert stat.S_IMODE(target.parent.stat().st_mode) == SECRET_DIR_MODE

    def test_existing_loose_permissions_are_tightened(self, tmp_path):
        """A token file written before this helper existed must be repaired."""
        target = tmp_path / "token.txt"
        target.write_text("old", encoding="utf-8")
        os.chmod(target, 0o644)
        assert stat.S_IMODE(target.stat().st_mode) == 0o644

        write_secret_file(target, "new")

        assert stat.S_IMODE(target.stat().st_mode) == SECRET_FILE_MODE
        assert target.read_text(encoding="utf-8") == "new"

    def test_overwrite_truncates(self, tmp_path):
        target = tmp_path / "token.txt"
        write_secret_file(target, "a-very-long-original-token")
        write_secret_file(target, "short")
        assert target.read_text(encoding="utf-8") == "short"

    def test_accepts_a_string_path(self, tmp_path):
        target = tmp_path / "token.txt"
        result = write_secret_file(str(target), "s3cret")
        assert result == target
        assert target.read_text(encoding="utf-8") == "s3cret"

    def test_raises_on_unwritable_location(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(OSError):
            write_secret_file(blocker / "token.txt", "s3cret")
