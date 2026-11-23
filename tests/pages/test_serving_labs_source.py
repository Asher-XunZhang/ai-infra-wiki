"""Read frozen Git objects only; no SGLang imports, execution or checkout changes."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess

source = os.environ.get("SGLANG_SOURCE_DIR")
if not source:
    raise SystemExit("Set SGLANG_SOURCE_DIR to a checkout containing the documented Git objects.")
root = Path(__file__).parents[2]
text = (root / "pages/sglang/serving-labs/evidence.js").read_text()
evidence = json.loads(re.search(r"const e=(.*);if\(typeof", text, re.S)[1])
rev = evidence["revision"]
for entry in evidence["sources"].values():
    raw = subprocess.check_output(["git", "-C", source, "show", rev + ":" + entry["path"]]).decode()
    nodes = [n for n in ast.walk(ast.parse(raw)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and n.name == entry["symbol"] and n.lineno == entry["line"]]
    assert nodes, entry
for flag in ["--disable-overlap-schedule", "--cuda-graph-backend-decode",
             "--cuda-graph-backend-prefill", "--random-range-ratio",
             "--max-concurrency", "--output-details", "--output-file"]:
    subprocess.run(["git", "-C", source, "grep", "-q", "-F", "-e", flag, rev,
                    "--", "python/sglang"], check=True)
print("Verified", len(evidence["sources"]), "source anchors and documented flags against frozen Git objects; no SGLang/GPU execution.")
