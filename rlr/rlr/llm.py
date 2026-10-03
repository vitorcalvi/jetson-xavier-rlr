"""Minimal standard-library client for llama-server's OpenAI-compatible API."""
import json
from urllib.parse import urlparse
from urllib import request


class NoRedirect(request.HTTPRedirectHandler):
    """Never let a local model endpoint redirect the client onto the network."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LlamaServer:
    def __init__(self, base_url="http://127.0.0.1:8080", model="Qwen2.5-1.5B", timeout=90):
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("RLR only supports a loopback llama-server endpoint")
        self.model = model
        self.timeout = timeout

    def complete(self, prompt, schema, schema_name):
        # Constrained actions and reports are short. An unbounded default let
        # small models generate hundreds of tokens for a one-action response.
        max_tokens = {"rlr_action": 96, "rlr_self_report": 96,
                      "rlr_reflection": 384,
                      "rlr_temperature_forecast": 64}.get(schema_name, 128)
        body = {"model": self.model,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema_name, "strict": True, "schema": schema}}}
        req = request.Request(self.base_url + "/v1/chat/completions",
                              data=json.dumps(body).encode("utf-8"),
                              headers={"Content-Type": "application/json"}, method="POST")
        opener = request.build_opener(NoRedirect())
        with opener.open(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]
