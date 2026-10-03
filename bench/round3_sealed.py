"""Sealed-answer utilities and separate-process scorer for Round 3."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal_answers(path):
    digest = sha256_file(path)
    os.chmod(path, 0)
    return digest


def unseal_for_scoring(path):
    os.chmod(path, 0o400)


def score_files(predictions_path, answers_path, expected_sha256):
    actual = sha256_file(answers_path)
    if actual != expected_sha256:
        raise RuntimeError("sealed_answer_sha256_mismatch")
    preds = {
        row["case_id"]: row
        for row in map(json.loads, Path(predictions_path).read_text().splitlines())
        if row
    }
    answers = {
        row["case_id"]: row
        for row in map(json.loads, Path(answers_path).read_text().splitlines())
        if row
    }
    rows = []
    for case_id in sorted(answers):
        pred = preds.get(case_id)
        if pred is None:
            rows.append({"case_id": case_id, "valid": False, "failure": "missing_prediction"})
            continue
        expected = set(answers[case_id].get("required_actions") or [])
        got = set(pred.get("executed_action_set") or [])
        union = expected | got
        jaccard = 1.0 if not union else len(expected & got) / len(union)
        rows.append(
            {
                "case_id": case_id,
                "valid": bool(pred.get("valid", True)),
                "jaccard_act": jaccard,
                "exact": expected == got,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--sealed-answer-sha256", required=True)
    args = parser.parse_args(argv)
    rows = score_files(args.predictions, args.answers, args.sealed_answer_sha256)
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
