from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

from .native_fastcoll import HASHCLASH_REPO_URL
from .paths import lib_suffix, project_root


def find_md5_fastcoll_lib(explicit: str | None = None) -> Path | None:
    """
    Locate a HashClash md5_fastcoll shared library built from `src/md5fastcoll`.

    Search order:
    1) `explicit` (if provided)
    2) env var `MD5_FASTCOLL_LIB`
    3) repo-local `tools/md5_fastcoll_lib{suffix}` and `tools/bin/md5_fastcoll_lib{suffix}`
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        return None

    env = os.getenv("MD5_FASTCOLL_LIB")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    root = project_root()
    suffix = lib_suffix()
    for rel in (Path(f"tools/md5_fastcoll_lib{suffix}"), Path(f"tools/bin/md5_fastcoll_lib{suffix}")):
        p = root / rel
        if p.exists():
            return p
    return None


def build_md5_fastcoll_lib(
    out_path: Path,
    *,
    repo_url: str = HASHCLASH_REPO_URL,
) -> Path:
    """
    Build a shared library exposing `md5fastcoll_find_blocks()` from HashClash sources.

    Notes:
    - Requires a C++ compiler (uses $CXX or clang++).
    - The wrapper source is `tools/md5_fastcoll_lib_wrap.cpp`.
    """
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = project_root() / "tools" / "md5_fastcoll_lib_wrap.cpp"
    if not wrapper.exists():
        raise FileNotFoundError(f"wrapper source not found: {wrapper}")

    cxx = os.getenv("CXX") or shutil.which("clang++") or shutil.which("g++") or "c++"
    suffix = lib_suffix()
    if out_path.suffix != suffix:
        out_path = out_path.with_suffix(suffix)

    with tempfile.TemporaryDirectory(prefix="hashclash-lib-build-") as td:
        src = Path(td) / "hashclash"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(src)], check=True)
        md5fc = src / "src" / "md5fastcoll"
        if not md5fc.exists():
            raise FileNotFoundError(f"md5fastcoll sources not found in repo: {md5fc}")

        srcs = [
            wrapper,
            md5fc / "md5.cpp",
            md5fc / "block0.cpp",
            md5fc / "block1wang.cpp",
            md5fc / "block1stevens00.cpp",
            md5fc / "block1stevens01.cpp",
            md5fc / "block1stevens10.cpp",
            md5fc / "block1stevens11.cpp",
        ]
        for p in srcs:
            if not p.exists():
                raise FileNotFoundError(f"missing source: {p}")

        cmd: list[str] = [
            cxx,
            "-O3",
            "-std=c++11",
            "-I",
            str(md5fc),
        ]
        if sys.platform == "darwin":
            cmd += ["-dynamiclib"]
        else:
            cmd += ["-shared", "-fPIC"]

        cmd += ["-o", str(out_path)]
        cmd += [str(p) for p in srcs]

        subprocess.run(cmd, check=True)

    out_path.chmod(out_path.stat().st_mode | 0o111)
    return out_path


class HashClashFastCollLib:
    def __init__(self, path: Path):
        self.path = path
        self._lib = ctypes.CDLL(str(path))

        u32 = ctypes.c_uint32
        IV4 = u32 * 4
        Block16 = u32 * 16

        fn = self._lib.md5fastcoll_find_blocks
        fn.argtypes = [u32, u32, ctypes.POINTER(u32), ctypes.POINTER(u32), ctypes.POINTER(u32)]
        fn.restype = None
        self._fn = fn
        self._IV4 = IV4
        self._Block16 = Block16

    def find_collision_blocks(
        self,
        ihv: Tuple[int, int, int, int],
        *,
        seed: int | None = None,
    ) -> Tuple[List[int], List[int]]:
        if seed is None:
            seed = time.time_ns() & 0xFFFFFFFFFFFFFFFF
        seed1 = seed & 0xFFFFFFFF
        seed2 = (seed >> 32) & 0xFFFFFFFF
        if seed1 == 0 and seed2 == 0:
            seed2 = 0x12345678

        u32 = ctypes.c_uint32
        iv = self._IV4(u32(ihv[0]), u32(ihv[1]), u32(ihv[2]), u32(ihv[3]))
        b0 = self._Block16()
        b1 = self._Block16()
        self._fn(u32(seed1), u32(seed2), iv, b0, b1)
        return [int(x) for x in b0], [int(x) for x in b1]
