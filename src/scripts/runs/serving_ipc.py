"""Check both vLLM and PyTorch socket paths before loading model weights."""
from __future__ import annotations

import os
from pathlib import Path
import socket
from uuid import uuid4


def validate_ipc_paths(ipc_dir, temporary_dir, *, bind=False):
    ipc, temporary = Path(ipc_dir), Path(temporary_dir)
    # PyTorch 2.8 IpcChannel uses TMPDIR/symm_mem-<pid>, independently of
    # VLLM_RPC_BASE_PATH. Linux sockaddr_un has 108 bytes including the NUL.
    paths = {"vllm": ipc / str(uuid4()),
             "torch_symmetric_memory": temporary / "symm_mem-2147483647"}
    for path in paths.values():
        if not path.is_absolute() or any(p == Path("/tmp") or p == Path("/var/tmp")
                                        for p in (path, *path.parents)):
            raise ValueError("Use an absolute workspace or job-local IPC directory")
        if len(os.fsencode(path)) > 107:
            raise ValueError(f"Unix socket pathname exceeds 107 bytes: {path}")
    if bind:
        for kind, worst_path in paths.items():
            worst_path.parent.mkdir(parents=True, exist_ok=True)
            path = (worst_path if kind == "vllm" else
                    temporary / f"symm_mem-{os.getpid()}")
            if path.exists():
                raise ValueError(f"Refusing to replace an existing IPC socket: {path}")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
                try:
                    channel.bind(str(path))
                finally:
                    path.unlink(missing_ok=True)
    return {"status": "PASS", "bound_on_cpu": bind,
            "socket_path_bytes": {k: len(os.fsencode(p)) for k, p in paths.items()}}


def qwen_ipc_environment(local, inherited):
    local = Path(local)
    ipc, temporary = local / "ipc", local / "tmp"
    validate_ipc_paths(ipc, temporary, bind=True)
    return dict(inherited, VLLM_RPC_BASE_PATH=str(ipc), TMPDIR=str(temporary),
                TMP=str(temporary), TEMP=str(temporary), TEMPDIR=str(temporary))
