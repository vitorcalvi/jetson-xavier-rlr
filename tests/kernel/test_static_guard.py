"""Static guard: no process/network primitive may be imported outside the kernel.

The runtime's whole safety story rests on there being exactly one place that can
spawn a process (the sandboxed :mod:`~reflective_loop_runtime.kernel.toolbroker`)
and exactly one place that may open a network connection (the model HTTP client).
This test walks every ``*.py`` under ``src/`` with :mod:`ast` and fails if any
other module imports ``subprocess``, ``socket``, ``urllib.request``,
``http.client``, ``requests`` or ``httpx``, or calls ``os.system``/``os.popen``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

# Modules whose import is forbidden everywhere except the allow-list below.
FORBIDDEN_MODULES = frozenset(
    {
        "subprocess",
        "socket",
        "urllib.request",
        "http.client",
        "requests",
        "httpx",
    }
)
# ``os.system`` / ``os.popen`` are the shell-out escape hatches inside ``os``.
FORBIDDEN_OS_ATTRS = frozenset({"system", "popen"})

# Explicit allow-list. These are the ONLY files permitted to import a process or
# network primitive:
#   * reflective_loop_runtime/kernel/toolbroker.py -- the bubblewrap sandbox
#     wrapper; the one sanctioned place a child process may be spawned.
#   * reflective_loop_runtime/qwen.py -- the existing Qwen model client, which
#     speaks HTTP to the LLM gateway through ``urllib.request``.
#   * reflective_loop_runtime/trading/http.py -- the paper trader's single
#     egress module; https to the two Alpaca hosts only.
ALLOWED_FILES = frozenset(
    {
        "reflective_loop_runtime/kernel/toolbroker.py",
        "reflective_loop_runtime/qwen.py",
        "reflective_loop_runtime/trading/http.py",
    }
)


def _forbidden_module(name: str) -> bool:
    return any(name == module or name.startswith(module + ".") for module in FORBIDDEN_MODULES)


def scan_source(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, description)`` for every forbidden construct in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:  # a broken file is itself a failure
        return [(0, f"could not parse: {exc}")]

    findings: list[tuple[int, str]] = []
    os_aliases = {"os"}  # every local name bound to the ``os`` module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _forbidden_module(alias.name):
                    findings.append((node.lineno, f"import {alias.name}"))
                if alias.name == "os":
                    os_aliases.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _forbidden_module(module):
                findings.append((node.lineno, f"from {module} import ..."))
            for alias in node.names:
                candidate = f"{module}.{alias.name}" if module else alias.name
                if _forbidden_module(candidate):
                    findings.append((node.lineno, f"from {module} import {alias.name}"))
                if module == "os" and alias.name in FORBIDDEN_OS_ATTRS:
                    findings.append((node.lineno, f"from os import {alias.name}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in os_aliases
                and func.attr in FORBIDDEN_OS_ATTRS
            ):
                findings.append((node.lineno, f"{func.value.id}.{func.attr}(...)"))
    return findings


def _src_modules() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if "__pycache__" not in path.parts)


def test_forbidden_primitives_stay_in_the_allow_list():
    """No module under ``src/`` outside the allow-list may import a primitive."""
    violations: list[str] = []
    for path in _src_modules():
        relative = path.relative_to(SRC).as_posix()
        if relative in ALLOWED_FILES:
            continue
        for lineno, description in scan_source(path):
            violations.append(f"{relative}:{lineno}: {description}")
    assert not violations, (
        "forbidden process/network primitive(s) imported outside the allow-list:\n  "
        + "\n  ".join(violations)
    )


def test_allow_list_files_exist():
    for relative in sorted(ALLOWED_FILES):
        assert (SRC / relative).is_file(), f"allow-listed file is missing: {relative}"


def test_guard_flags_synthetic_violations(tmp_path):
    """The scanner is not vacuous: it must catch every forbidden shape."""
    sample = tmp_path / "synthetic.py"
    sample.write_text(
        "import subprocess\n"
        "import socket\n"
        "import urllib.request\n"
        "from http import client\n"
        "from urllib import request\n"
        "import os\n"
        "import os as operating_system\n"
        "def f():\n"
        "    os.system('true')\n"
        "    operating_system.popen('true')\n",
        encoding="utf-8",
    )
    found = {description for _, description in scan_source(sample)}
    assert "import subprocess" in found
    assert "import socket" in found
    assert "import urllib.request" in found
    assert "from http import client" in found
    assert "from urllib import request" in found
    assert any(item.startswith("os.system(") for item in found)
    # ``import os as operating_system`` is tracked too, so the aliased popen call
    # is still caught rather than sneaking past the guard.
    assert any(item.startswith("operating_system.popen(") for item in found)


def test_guard_ignores_benign_stdlib_usage(tmp_path):
    """``os.path``/``urllib.parse`` and friends are not forbidden."""
    sample = tmp_path / "benign.py"
    sample.write_text(
        "import os\n"
        "from urllib.parse import urlparse\n"
        "import json\n"
        "def f():\n"
        "    return os.path.join(urlparse('x').path, json.dumps({}))\n",
        encoding="utf-8",
    )
    assert scan_source(sample) == []
