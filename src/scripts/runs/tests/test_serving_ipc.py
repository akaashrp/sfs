"""Exercise real Unix socket binds and isolate worker temporary directories."""
import os
from pathlib import Path
import tempfile

import pytest

from scripts.runs.serving_ipc import qwen_ipc_environment, validate_ipc_paths


@pytest.fixture
def short_local():
    root = Path(__file__).resolve().parents[4]
    # Keep test socket paths below the OS limit without using system temp dirs.
    with tempfile.TemporaryDirectory(prefix="i", dir=root) as name:
        yield Path(name)


def test_torch_socket_limit_is_checked_independently_of_vllm(short_local):
    # Exceed the limit independently of the checkout's pathname length.
    temporary = short_local / ('x' * 108)
    with pytest.raises(ValueError, match="107 bytes"):
        validate_ipc_paths(short_local/"ipc", temporary)


def test_worker_environment_overrides_long_inherited_temp_and_binds(short_local):
    inherited = {"TMPDIR": "/a/very/long/inherited/directory", "KEEP": "yes"}
    updated = qwen_ipc_environment(short_local, inherited)
    assert updated["KEEP"] == "yes" and inherited["TMPDIR"].startswith("/a/")
    assert {updated[k] for k in ("TMPDIR", "TMP", "TEMP", "TEMPDIR")} == {str(short_local/"tmp")}
    assert updated["VLLM_RPC_BASE_PATH"] == str(short_local/"ipc")
    assert not list(short_local.rglob("symm_mem-*"))
    assert validate_ipc_paths(short_local/"ipc", short_local/"tmp", bind=True)["status"] == "PASS"


def test_bind_check_preserves_existing_socket_path(short_local):
    temporary = short_local/"tmp"
    temporary.mkdir()
    existing = temporary/f"symm_mem-{os.getpid()}"
    existing.write_text("preserve")
    with pytest.raises(ValueError, match="Refusing to replace"):
        validate_ipc_paths(short_local/"ipc", temporary, bind=True)
    assert existing.read_text() == "preserve"
