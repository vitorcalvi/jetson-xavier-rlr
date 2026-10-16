"""End-to-end tests for the safety-kernel protection guard.

``scripts/check_protected.sh`` must reject a diff that touches a protected
kernel path and accept the same diff once ``KERNEL_CHANGE=1`` acknowledges it.
Each test builds a throwaway git repository under ``tmp_path`` so the suite
never mutates the checkout it runs from.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_protected.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="git and bash are required for the protection guard tests",
)


def _clean_env(**overrides: str) -> dict[str, str]:
    """Return an isolated environment with git discovery vars removed."""
    env = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
        "KERNEL_CHANGE",
        "KERNEL_PROTECTED_PATHS",
    ):
        env.pop(key, None)
    # Ignore any user/system git config so commits never depend on the host.
    # Use a committed empty config rather than /dev/null: some CI containers
    # mount /dev/null incorrectly as a regular file with unexpected content.
    env["GIT_CONFIG_GLOBAL"] = str(REPO_ROOT / "tests" / "empty-gitconfig")
    env["GIT_CONFIG_SYSTEM"] = str(REPO_ROOT / "tests" / "empty-gitconfig")
    env.update(overrides)
    return env


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = [
        "git",
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test Harness",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "init.defaultBranch=main",
        *args,
    ]
    return subprocess.run(
        cmd, cwd=str(repo), env=_clean_env(), capture_output=True, text=True
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "-q").returncode == 0
    (repo / "README.txt").write_text("base\n", encoding="utf-8")
    assert _git(repo, "add", "README.txt").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "base").returncode == 0
    return repo


def _rev(repo: Path, rev: str) -> str:
    result = _git(repo, "rev-parse", rev)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit_file(repo: Path, relpath: str, content: str, message: str) -> str:
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    assert _git(repo, "add", relpath).returncode == 0
    result = _git(repo, "commit", "-q", "-m", message)
    assert result.returncode == 0, result.stderr
    return _rev(repo, "HEAD")


def _run_guard(repo: Path, *range_args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *range_args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )


class TestCheckProtected:
    def test_blocks_governor_change_without_override(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(
            repo, "src/reflective_loop_runtime/governor.py", "x = 1\n", "touch governor"
        )

        proc = _run_guard(repo, base, head, env=_clean_env())

        assert proc.returncode == 1
        assert "governor.py" in (proc.stdout + proc.stderr)

    def test_allows_governor_change_with_kernel_change(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(
            repo, "src/reflective_loop_runtime/governor.py", "x = 1\n", "touch governor"
        )

        proc = _run_guard(repo, base, head, env=_clean_env(KERNEL_CHANGE="1"))

        assert proc.returncode == 0, proc.stderr

    def test_kernel_change_zero_still_blocks(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(
            repo, "src/reflective_loop_runtime/executor.py", "y = 1\n", "touch executor"
        )

        proc = _run_guard(repo, base, head, env=_clean_env(KERNEL_CHANGE="0"))

        assert proc.returncode == 1

    def test_allows_unprotected_change(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(repo, "docs/notes.md", "hello\n", "docs only")

        proc = _run_guard(repo, base, head, env=_clean_env())

        assert proc.returncode == 0, proc.stderr

    def test_accepts_single_range_argument(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(
            repo, "src/reflective_loop_runtime/kernel/__init__.py", "", "kernel package"
        )

        blocked = _run_guard(repo, f"{base}..{head}", env=_clean_env())
        allowed = _run_guard(
            repo, f"{base}..{head}", env=_clean_env(KERNEL_CHANGE="1")
        )

        assert blocked.returncode == 1
        assert allowed.returncode == 0, allowed.stderr

    @pytest.mark.parametrize(
        "relpath",
        [
            "src/reflective_loop_runtime/kernel/thing.py",
            "src/reflective_loop_runtime/executor.py",
            "tests/kernel/test_something.py",
            "policy/rules.json",
            "sandbox/limits.toml",
        ],
    )
    def test_every_protected_glob_blocks(self, tmp_path: Path, relpath: str):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(repo, relpath, "payload\n", f"touch {relpath}")

        proc = _run_guard(repo, base, head, env=_clean_env())

        assert proc.returncode == 1
        assert relpath in (proc.stdout + proc.stderr)

    def test_override_env_accepts_alternate_list(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(repo, "src/reflective_loop_runtime/governor.py", "x\n", "governor")

        alt = tmp_path / "alt_protected.txt"
        alt.write_text("docs/\n", encoding="utf-8")

        proc = _run_guard(
            repo, base, head, env=_clean_env(KERNEL_PROTECTED_PATHS=str(alt))
        )

        assert proc.returncode == 0, proc.stderr

    def test_missing_protection_list_is_usage_error(self, tmp_path: Path):
        repo = _init_repo(tmp_path)
        base = _rev(repo, "HEAD")
        head = _commit_file(repo, "docs/notes.md", "hello\n", "docs only")

        proc = _run_guard(
            repo,
            base,
            head,
            env=_clean_env(KERNEL_PROTECTED_PATHS=str(tmp_path / "nope.txt")),
        )

        assert proc.returncode == 2
