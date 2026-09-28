import json

from rlr.llm import LlamaServer


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()


class FakeOpener:
    def __init__(self, expected_max_tokens):
        self.expected_max_tokens = expected_max_tokens

    def open(self, req, timeout):
        body = json.loads(req.data)
        assert body["max_tokens"] == self.expected_max_tokens
        assert timeout == 90
        return FakeResponse()


def test_completion_bounds_generation_by_schema(monkeypatch):
    import rlr.llm as llm

    for schema_name, expected in (("rlr_action", 96),
                                  ("rlr_self_report", 96),
                                  ("rlr_reflection", 384)):
        monkeypatch.setattr(llm.request, "build_opener",
                            lambda *_args, expected=expected: FakeOpener(expected))
        assert LlamaServer().complete("prompt", {}, schema_name) == "{}"
