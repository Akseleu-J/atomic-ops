"""Check new Ruff/format debt; syntax and undefined names cannot be baselined."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".github/lint-baseline.json"


def collect():
    files = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT).decode().split("\0")
    files = [f for f in files if f and not f.startswith("archive/")]
    command = [sys.executable, "-m", "ruff"]
    result = subprocess.run(
        [*command, "check", "--output-format=json", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    records = []
    for item in json.loads(result.stdout):
        path = Path(item["filename"]).resolve().relative_to(ROOT).as_posix()
        code = item["code"] or "invalid-syntax"
        if code.startswith("E9") or code in ("invalid-syntax", "F821", "F822", "F823"):
            raise RuntimeError(f"Non-baselinable error: {path}: {code}: {item['message']}")
        lines = (ROOT / path).read_text(encoding="utf-8-sig").splitlines()
        line = lines[item["location"]["row"] - 1].strip()
        fingerprint = hashlib.sha256(
            json.dumps([path, code, item["message"], line], ensure_ascii=False).encode()
        ).hexdigest()
        records.append({"fingerprint": fingerprint, "file": path, "code": code, "message": item["message"]})
    formatted = subprocess.run(
        [*command, "format", "--check", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if formatted.returncode not in (0, 1):
        raise RuntimeError(formatted.stderr)
    unformatted = sorted(
        line.removeprefix("Would reformat: ").replace("\\", "/")
        for line in formatted.stdout.splitlines()
        if line.startswith("Would reformat: ")
    )
    return records, unformatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true", help="Reviewed maintenance only; never called by CI")
    args = parser.parse_args()
    records, unformatted = collect()
    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "ruff_version": "0.14.0",
                    "scope": "Tracked active Python files, both generations, tests and tools; archive/ excluded",
                    "meaning": "Existing style and Bugbear diagnostics; these are not resolved by recording a baseline",
                    "diagnostics": records,
                    "unformatted_files": unformatted,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Recorded {len(records)} existing diagnostics, {len(unformatted)} unformatted files")
        return 0
    baseline = json.loads(BASELINE.read_text())
    old = Counter(r["fingerprint"] for r in baseline["diagnostics"])
    current = Counter(r["fingerprint"] for r in records)
    new = current - old
    new_format = set(unformatted) - set(baseline["unformatted_files"])
    summary = (
        f"Existing Ruff diagnostics: {len(records)}; new: {sum(new.values())}; "
        f"unformatted files: {len(unformatted)}; newly unformatted: {len(new_format)}"
    )
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(
                "## Lint baseline comparison\n\n"
                + summary
                + "\n\nPASS means no new debt, not a clean full Ruff audit.\n"
            )
    for r in records:
        if r["fingerprint"] in new:
            print(f"NEW: {r['file']}: {r['code']}: {r['message']}")
    for path in sorted(new_format):
        print("NEW FORMAT DEBT:", path)
    return int(bool(new or new_format))


if __name__ == "__main__":
    raise SystemExit(main())
