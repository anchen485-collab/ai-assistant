"""同步状态的两个小工具：跨进程互斥，以及完整写入后再替换 JSON。"""

from __future__ import annotations

import errno
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def sync_lock(path: Path, timeout: float = 60):
    """同一目录只允许一个同步者；进程退出后操作系统自动释放锁。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        # 锁定范围可以超过 EOF；不写占位字节，避免首次并发创建时写入已锁范围。
        deadline = time.monotonic() + timeout
        while True:
            try:
                file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"等待知识库同步锁超时: {path}") from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, payload: dict) -> None:
    """临时文件与目标在同一目录，避免中途失败留下半份清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
