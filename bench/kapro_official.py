"""Preview reproduction against the authors' released KAware preview.

Scope warning, which the final report must preserve: the public repository
contains 8 samples (one per boundary x topology bucket) against the paper's
1,076. Any number produced here is a preview figure. It is not a reproduction
of the paper's results and must never be reported as one.

Both branches use the same endpoint, temperature 0, and the same tool
descriptions. Acc_know, Acc_act and KAS are recomputed here from the raw traces
rather than taken from any precomputed value the authors shipped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kapro_scoring as ks  # noqa: E402

DATA_SUBDIRS = ("External_Function", "Internal_Function", "Hybrid_Function")
BOUNDARY_MAP = {"External_Function": "external", "Internal_Function": "internal",
                "Hybrid_Function": "hybrid"}


def read_preview(dataset_dir):
    rows = []
    for subdir in DATA_SUBDIRS:
        for path in sorted((Path(dataset_dir) / subdir).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_file"] = str(path)
                row["_bucket"] = subdir
                rows.append(row)
    return rows


def parse_tool_doc(raw):
    """Same traversal the authors' runner uses, so tools render identically."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            return [json.loads(raw)]
        except ValueError:
            return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            out.extend(parse_tool_doc(item))
        return out
    if isinstance(raw, dict) and "name" in raw:
        return [raw]
    if isinstance(raw, dict):
        out = []
        for item in raw.values():
            out.extend(parse_tool_doc(item))
        return out
    return []


def target_names(sample):
    """The ground-truth tool set T, as bare tool names."""
    raw = sample.get("target_tool") or ""
    names = []
    for part in str(raw).split(","):
        if "::" in part:
            names.append(part.split("::")[-1].strip())
        elif part.strip():
            names.append(part.strip())
    return [n for n in names if n]


def knowing_prompt(sample):
    """The authors' build_prompt_call, reproduced verbatim in substance."""
    tools = json.dumps([{t.get("name"): {"description": t.get("description", ""),
                                         "input_schema": t.get("input_schema", {})}}
                        for t in parse_tool_doc(sample.get("tool_doc"))],
                       ensure_ascii=False, indent=1)
    return ("# Role\nYou are an agent responsible for evaluating if a user's task "
            "requires tools.\n\n# Task\n%s\n\n# Available Tools\n%s\n\n# Instructions\n"
            "Assess each tool's necessity according to task specifications.\n"
            "- Set a tool to `true` if its functionality is essential for task completion.\n"
            "- Set a tool to `false` if internal capabilities are sufficient.\n\n"
            "# Output Format\nReturn a valid JSON object ONLY in the following format:\n"
            "{\n  \"tool_decisions\": {\n    \"ToolA\": true,\n    \"ToolB\": false\n  }\n}"
            % (sample["task"], tools))


def parse_knowing(raw, tool_names):
    """Turn a tool_decisions object into the set of tools deemed necessary."""
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        decisions = value.get("tool_decisions") if isinstance(value, dict) else None
        if not isinstance(decisions, dict):
            return None, "no tool_decisions object"
        return {n for n, flag in decisions.items()
                if bool(flag) and n in set(tool_names)}, None
    except (ValueError, TypeError) as exc:
        return None, "unparseable: %s" % exc


def acting_messages(sample):
    """Chat messages for the Acting branch: the task, with tools registered.

    No tool_doc hints about necessity are added, and the model's own Knowing
    answer is never included, so the Acting branch is not primed by it.
    """
    tools = [{"type": "function",
              "function": {"name": t.get("name"),
                           "description": t.get("description", ""),
                           "parameters": t.get("input_schema", {}) or
                           {"type": "object", "properties": {}}}}
             for t in parse_tool_doc(sample.get("tool_doc"))]
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sample["task"]}], tools


def replay_responses(sample, calls):
    """Dataset-recorded tool results, so no live MCP server is contacted.

    The preview ships replayable responses. Using them keeps the run offline and
    makes the measurement depend on the model's choices rather than on the
    availability of third-party Smithery servers.
    """
    recorded = sample.get("tool_response") or {}
    results = []
    for call in calls:
        name = call.get("name")
        value = recorded.get(name) if isinstance(recorded, dict) else None
        results.append({"role": "tool", "name": name,
                        "content": json.dumps(value if value is not None
                                               else {"replayed": True})})
    return results


def run_sample(sample, client, model, max_rounds=3):
    """Knowing and Acting for one preview sample, recorded in full."""
    tools = target_names(sample)
    know_raw = client.complete(knowing_prompt(sample), max_tokens=512)
    know_set, know_error = parse_knowing(know_raw, tools)
    messages, definitions = acting_messages(sample)
    act_set, act_rounds, act_raw = set(), [], []
    for _ in range(max_rounds):
        reply = client.complete_chat(messages, definitions, max_tokens=512)
        act_raw.append(reply["content"])
        calls = reply.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            act_set.add(call.get("name"))
        messages = messages + [{"role": "assistant", "content": reply["content"] or "",
                                "tool_calls": reply.get("raw_tool_calls") or []},
                               ] + replay_responses(sample, calls)
        act_rounds.append(len(calls))
    return {
        "index": sample.get("index"), "bucket": sample["_bucket"],
        "boundary": BOUNDARY_MAP[sample["_bucket"]], "hop": sample.get("hop"),
        "T": tools, "E_know": sorted(know_set) if know_set is not None else None,
        "knowing_error": know_error, "E_act": sorted(act_set),
        "knowing_raw": know_raw, "acting_raw": act_raw,
        "rounds": len(act_rounds),
    }


class OpenAICompat:
    """Minimal stdlib client for the local llama-server, greedy decoding."""

    def __init__(self, base_url, model, timeout=180):
        from urllib import request as _request
        from urllib.parse import urlparse
        self._request = _request
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("only a loopback endpoint is permitted")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, body):
        import json as _json
        req = self._request.Request(
            self.base_url + "/v1/chat/completions",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with self._request.urlopen(req, timeout=self.timeout) as response:
            return _json.loads(response.read().decode("utf-8"))

    def complete(self, prompt, max_tokens=512):
        out = self._post({"model": self.model, "temperature": 0.0,
                          "max_tokens": max_tokens,
                          "messages": [{"role": "user", "content": prompt}]})
        return out["choices"][0]["message"]["content"]

    def complete_chat(self, messages, tools, max_tokens=512):
        out = self._post({"model": self.model, "temperature": 0.0,
                          "max_tokens": max_tokens, "messages": messages,
                          "tools": tools, "tool_choice": "auto"})
        message = out["choices"][0]["message"]
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            calls.append({"name": function.get("name"),
                          "arguments": function.get("arguments")})
        return {"content": message.get("content"), "tool_calls": calls,
                "raw_tool_calls": message.get("tool_calls") or []}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KAware preview reproduction")
    parser.add_argument("--dataset", default="/models/bench/KAware")
    parser.add_argument("--out", default="/models/bench/kapro-official/preview.json")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen2.5-1.5B")
    args = parser.parse_args()

    samples = read_preview(args.dataset)
    client = OpenAICompat(args.server, args.model)
    records, rows = [], []
    partial = Path(args.out).with_suffix(".partial.jsonl")
    partial.parent.mkdir(parents=True, exist_ok=True)
    # Write each record as it completes. The preview run shares the device with
    # the main benchmark and the thermal watchdog aborts it partway through, so
    # writing only at the end would discard every completed sample.
    with open(partial, "a", encoding="utf-8") as sink:
        for sample in samples:
            try:
                record = run_sample(sample, client, args.model)
            except OSError as exc:
                record = {"index": sample.get("index"), "bucket": sample["_bucket"],
                          "boundary": BOUNDARY_MAP[sample["_bucket"]],
                          "hop": sample.get("hop"), "T": target_names(sample),
                          "E_know": None, "E_act": None,
                          "knowing_error": "transport: %s" % exc, "acting_raw": None}
            records.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            if record.get("E_know") is None or record.get("E_act") is None:
                continue
            rows.append({"case_id": "kaware-%s" % record["index"],
                         "boundary": record["boundary"], "topology": record["hop"],
                         "condition": "kaware_preview",
                         "ground_truth": record["T"],
                         "knowing": {"required_actions": record["E_know"]},
                         "acting": {"E_act": record["E_act"],
                                    "tool_call_count": sum(1 for a in record["acting_raw"]
                                                           if a)},
                         "pass": True})
    summary = {
        "scope": "preview_reproduction",
        "samples_available": len(samples), "samples_scored": len(rows),
        "paper_task_count": 1076,
        "full_dataset_available": False,
        "overall": ks.aggregate(rows),
        "by_boundary": ks.aggregate(rows, breakdown="boundary"),
        "model": args.model, "temperature": 0.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records},
                              ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("samples_available", "samples_scored")}, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
