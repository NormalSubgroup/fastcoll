from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

HASHCLASH_REPO_URL = "https://github.com/cr-marcstevens/hashclash"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_md5_fastcoll_bin(explicit: str | None = None) -> Path | None:
    """
    Locate a native md5_fastcoll-compatible binary.

    Search order:
    1) `explicit` (if provided)
    2) env var `MD5_FASTCOLL_BIN`
    3) repo-local `tools/md5_fastcoll` and `tools/bin/md5_fastcoll`
    4) PATH: `md5_fastcoll`, then `fastcoll`
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        return None

    env = os.getenv("MD5_FASTCOLL_BIN")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    root = _project_root()
    for rel in (Path("tools/md5_fastcoll"), Path("tools/bin/md5_fastcoll")):
        p = root / rel
        if p.exists():
            return p

    for name in ("md5_fastcoll", "fastcoll"):
        hit = shutil.which(name)
        if hit:
            return Path(hit)
    return None


def run_md5_fastcoll(
    bin_path: Path,
    *,
    out1: Path,
    out2: Path,
    prefixfile: Path | None = None,
    ihv_hex: str | None = None,
    seed: int | None = None,
    quiet: bool = False,
) -> None:
    """
    Run the native md5_fastcoll binary to generate a collision pair.

    Notes:
    - `-o/--out` must be the last option (as in the original tool).
    - When `prefixfile` is provided, the tool computes the IHV after prefix itself.
    """
    cmd: list[str] = [str(bin_path)]

    if quiet:
        cmd.append("-q")

    if seed is not None:
        seed1 = seed & 0xFFFFFFFF
        seed2 = (seed >> 32) & 0xFFFFFFFF
        cmd += ["--seed1", str(seed1), "--seed2", str(seed2)]

    if prefixfile is not None:
        cmd += ["-p", str(prefixfile)]
    elif ihv_hex is not None:
        cmd += ["-i", ihv_hex]

    cmd += ["-o", str(out1), str(out2)]
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)


def build_md5_fastcoll(
    out_path: Path,
    *,
    repo_url: str = HASHCLASH_REPO_URL,
    jobs: int | None = None,
) -> Path:
    """
    Build `bin/md5_fastcoll` from the HashClash repo and copy it to `out_path`.
    """
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = jobs or (os.cpu_count() or 1)

    with tempfile.TemporaryDirectory(prefix="hashclash-build-") as td:
        src = Path(td) / "hashclash"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(src)], check=True)
        subprocess.run(["autoreconf", "-i"], cwd=src, check=True)
        subprocess.run(["./configure"], cwd=src, check=True)
        subprocess.run(["make", f"-j{jobs}", "bin/md5_fastcoll"], cwd=src, check=True)

        built = src / "bin" / "md5_fastcoll"
        if not built.exists():
            raise FileNotFoundError(f"build succeeded but binary not found: {built}")
        shutil.copy2(built, out_path)
        out_path.chmod(out_path.stat().st_mode | 0o111)
    return out_path
