#!/usr/bin/env python3
"""Read a Harness rc.2 projection; gate resume without model calls or writes."""
import argparse
import json
import math
from pathlib import Path
import sys


def count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing or invalid token count")
    if not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError("invalid token count")
    return int(value)


def inspect(args):
    if not 0 < args.warn_tokens < args.rotate_tokens:
        raise ValueError("require 0 < warn-tokens < rotate-tokens")
    data = json.loads(Path(args.projection).read_text())
    if data.get("version") != 7:
        raise ValueError("unsupported projection version; verify installed schema")
    record = data["record"]
    if Path(record["identity"]["cwd"]).resolve() != Path(args.cwd).resolve():
        raise ValueError("projection cwd does not match target workspace")
    rows = record["rows"]
    usage = rows["tokenUsage"]["val"]
    pressure = rows["contextPressure"]["val"]
    boundary = rows["turnBoundary"]["val"]
    last = usage["last"]["buckets"]
    # These buckets are disjoint; cached input must not be subtracted again.
    keys = ("uncachedInputTokens", "cacheReadTokens", "cacheWriteTokens")
    last_input = sum(count(last[k]) for k in keys)
    recorded_pressure = count(pressure["pressureTokens"])
    totals = {k: count(usage["totals"][k]) for k in (*keys, "outputTokens")}
    observed = max(last_input, recorded_pressure)
    if boundary["openTurnStartSeq"] is not None:
        decision, code = "BUSY", 30
    elif args.phase_changed or observed >= args.rotate_tokens:
        decision, code = "ROTATE", 20
    elif observed >= args.warn_tokens:
        decision, code = "REVIEW", 10
    else:
        decision, code = "PASS", 0
    return {
        "decision": decision,
        "last_input_tokens": last_input,
        "recorded_pressure_tokens": recorded_pressure,
        "recorded_totals": totals,
        "phase_changed": args.phase_changed,
        "thresholds": {"warn": args.warn_tokens, "rotate": args.rotate_tokens},
        "boundary": "Local snapshot only; not a live process check or billing total.",
    }, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--phase-changed", action="store_true")
    parser.add_argument("--warn-tokens", type=int, default=128000)
    parser.add_argument("--rotate-tokens", type=int, default=256000)
    args = parser.parse_args()
    try:
        result, code = inspect(args)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        # Do not echo private paths, file contents, or JSON parse excerpts.
        result, code = {"decision": "UNKNOWN", "error_type": type(exc).__name__,
                        "action": "Verify local schema, required fields and cwd."}, 40
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
