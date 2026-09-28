import io
import os
import tempfile
from urllib.error import HTTPError

import pytest

import orchestrator


class Response:
    def __init__(self, status=200, lines=()):
        self.status = status
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_health_wait_retries_until_server_returns_200():
    statuses = iter([ConnectionError("loading"), Response(503), Response(200)])
    sleeps = []

    def get(*_, **__):
        item = next(statuses)
        if isinstance(item, Exception):
            raise item
        return item

    orchestrator._wait_for_health("http://llm:8080", get=get, sleep=sleeps.append)
    assert sleeps == [1, 1]


def test_lock_fails_when_another_run_holds_it():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "run.lock")
        held = orchestrator._acquire_run_lock(path)
        try:
            with pytest.raises(RuntimeError, match="another prototype run"):
                orchestrator._acquire_run_lock(path)
        finally:
            os.close(held)


def test_llm_retries_503_with_bounded_exponential_backoff():
    errors = [HTTPError("url", 503, "loading", {}, io.BytesIO()) for _ in range(2)]
    responses = iter([*errors, Response(lines=[b"data: {\"content\":\"ok\"}\n", b"data: [DONE]\n"])])
    sleeps = []

    def open_url(*_, **__):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    text, _, _ = orchestrator._llm("prompt", "http://llm:8080", 5, opener=open_url, sleep=sleeps.append)
    assert text == "ok"
    assert sleeps == [0.5, 1.0]
