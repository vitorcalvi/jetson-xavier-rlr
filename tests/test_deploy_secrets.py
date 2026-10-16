"""The shipped Jetson deploy templates must never carry secrets.

``deploy/`` is committed as the source of truth for the soak install, so it must
hold only non-secret defaults. ``rlr.env.example`` is a copy-me template whose
values are placeholders, and no deploy file may embed a key, token or private
key. This guards that property the same way the runtime guards every other
boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent.parent / "deploy"

# Credential shapes that must never appear in a shipped template.
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"ssh-(rsa|ed25519|dss) "),
    re.compile(r"\b(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,})\b"),
    re.compile(r'''(?i)\b(?:authorization|bearer|api[_-]?key|password|token)\b\s*[:=]\s*["'][^"'$]{8,}["']'''),
    # Unquoted config/header literals. Keep this line-oriented so Python keyword
    # arguments such as api_key=api_key, are not mistaken for credentials.
    re.compile(
        r'''(?im)^[ \t-]*(?:authorization|api[_-]?key|password|token)'''
        r'''(?![A-Za-z0-9_])\s*:\s*(?:bearer\s+)?(?=[^"'$#\s])\S+'''
    ),
    re.compile(
        r'''(?im)^[ \t]*(?:api[_-]?key|password|token)'''
        r'''(?![A-Za-z0-9_])=(?=[^"'$#\s])[^,\s#]+[ \t]*(?:#.*)?$'''
    ),
    # Exact credential segments in shell/env variable names. TOKEN does not
    # match TOKENS, so RLR_QWEN_MAX_TOKENS stays non-secret.
    re.compile(
        r'''(?m)^[ \t]*(?:export[ \t]+)?'''
        r'''(?:[A-Z0-9]+_)*(?:API_?KEY|PASSWORD|SECRET|TOKEN)(?:_[A-Z0-9]+)*'''
        r'''[ \t]*=[ \t]*(?:["'][^"'$]+["']|(?=[^$#\s])\S+)'''
    ),
)

# The only keys, and their non-secret placeholder values, the template may hold.
_EXPECTED_ENV = {
    "RLR_TICK_SECONDS": "1.0",
    "RLR_ADVISOR_ENABLE": "0",
    "RLR_MODEL_NAME": "falcon06",
    "RLR_LLAMA_BASE_URL": "http://127.0.0.1:8080",
    "RLR_QWEN_TIMEOUT_SECONDS": "15",
    "RLR_QWEN_MAX_TOKENS": "121",
    "RLR_POLICY_PATH": "/home/vitor/.config/rlr/self-maintenance.policy.json",
    "RLR_POLICY_PUBLIC_KEY_PATH": "/home/vitor/.config/rlr/operator-public.pem",
    "RLR_SANDBOX_PIDS_CGROUP": "/sys/fs/cgroup/pids/rlr-sandbox",
    "RLR_SANDBOX_PIDS_CGROUP_EXEMPT": "0",
    "RLR_LAN_LLM_ENABLE": "0",
    "RLR_LAN_LLM_BASE_URL": "http://192.168.1.45:4000/v1",
    "RLR_LAN_LLM_MODEL": "g1-maintenance",
    "RLR_LAN_LLM_TIMEOUT_SECONDS": "1.5",
    "RLR_LAN_LLM_MAX_TOKENS": "768",
    "RLR_LAN_ESCALATION_THRESHOLD": "0.55",
    "RLR_LAN_LLM_API_KEY": "",
}


def _env_pairs(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


def test_env_example_holds_only_placeholders():
    pairs = _env_pairs((_DEPLOY / "rlr.env.example").read_text(encoding="utf-8"))
    assert pairs == _EXPECTED_ENV
    assert all(key.startswith("RLR_") for key in pairs)
    for value in pairs.values():
        assert not any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def test_no_deploy_file_embeds_a_credential():
    offenders: list[str] = []
    for path in sorted(_DEPLOY.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.name} matches {pattern.pattern!r}")
    assert not offenders, offenders


def test_secret_patterns_keep_old_coverage_and_add_unquoted_forms():
    hostile = (
        'password = "verysecretvalue"',
        'client(authorization="Bearer verysecretvalue")',
        '  - Authorization: "Bearer verysecretvalue"',
        'Authorization: Bearer x',
        'password=x',
        'RLR_LAN_LLM_API_KEY="secretvalue"',
        'RLR_GITHUB_ISSUE_TOKEN=x',
    )
    for sample in hostile:
        assert any(pattern.search(sample) for pattern in _SECRET_PATTERNS), sample

    safe = (
        'RLR_QWEN_MAX_TOKENS=121',
        'RLR_LAN_LLM_MAX_TOKENS=768',
        'RLR_LAN_LLM_API_KEY=',
        'RLR_GITHUB_ISSUE_TOKEN=',
        'token=$FROM_ENV',
        'api_key = os.environ.get("RLR_LAN_LLM_API_KEY", "")',
        'api_key=api_key,',
    )
    for sample in safe:
        assert not any(pattern.search(sample) for pattern in _SECRET_PATTERNS), sample
