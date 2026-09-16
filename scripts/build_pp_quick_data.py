#!/usr/bin/env python3
"""Derive beginner views from the detailed page's fixed illustrative model."""
import hashlib
import json
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages/sglang/pd-prefill-pp-loop"


class EmbeddedModel(HTMLParser):
    def __init__(self):
        super().__init__()
        self.srcdoc = None
        self.in_model = False
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "iframe":
            self.srcdoc = attrs.get("srcdoc")
        if tag == "script" and "data-model" in attrs:
            self.in_model = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_model = False

    def handle_data(self, text):
        if self.in_model:
            self.chunks.append(text)


def replace_embedded_model(model):
    """Refresh only the JSON in the existing iframe, preserving the renderer."""
    page = PAGE / 'index.html'
    html = page.read_text(encoding='utf-8')
    outer = EmbeddedModel(); outer.feed(html)
    inner = EmbeddedModel(); inner.feed(outer.srcdoc)
    original = ''.join(inner.chunks)
    old = escape(original, quote=True)
    assert html.count(old) == 1, 'Expected one embedded model'
    new = escape(json.dumps(model, ensure_ascii=False, separators=(',', ':')), quote=True)
    page.write_text(html.replace(old, new), encoding='utf-8')


def owner_batches(owner):
    """Use explicit model ownership, including the shared M1–M5 request set."""
    batches = {int(n) for n in re.findall(r"M(\d+)", owner)}
    for first, last in re.findall(r"M(\d+)[–-]M(\d+)", owner):
        batches.update(range(int(first), int(last) + 1))
    return sorted(b for b in batches if 1 <= b <= 5)


def derive_quick(data, source):
    assert data["baseline"] == "72d5c5bb73" and data["pp"] == 3 and data["depth"] == 0
    graph = data["graph"]
    phase_group = {p: i for i, phases in enumerate(["A", "BC", "DE", "FGH", "I"]) for p in phases}
    loops = []
    for original in data["loops"]:
        r, n = original["r"], original["n"]
        events = sorted((e for e in data["events"] if e["r"] == r and e["n"] == n and e["kind"] in ("cpu", "wait")), key=lambda e: (e["start"], e["end"]))
        released = [b for item in data["releases"] if item["r"] == r and item["n"] == n for b in item["batches"]]
        groups = []
        for i in range(5):
            subset = [e for e in events if phase_group[e["phase"]] == i]
            links = {}
            for event in subset:
                owners = released if event["id"] == f"{r}:{n}:release" else owner_batches(event["owner"])
                for batch in owners:
                    link = links.setdefault(batch, {"shared": True, "actions": []})
                    link["shared"] = link["shared"] and len(owners) > 1
                    action = event["label"].removeprefix("等：")
                    if action not in link["actions"]:
                        link["actions"].append(action)
            groups.append(None if not subset else {
                "start": subset[0]["start"], "end": subset[-1]["end"],
                "refs": sorted({e["ref"] for e in subset}),
                "wait": round(sum(e["end"] - e["start"] for e in subset if e["kind"] == "wait"), 3),
                "links": links,
            })
        present = [g for g in groups if g]
        assert abs(present[0]["start"] - original["start"]) < .002
        assert abs(present[-1]["end"] - original["end"]) < .002
        assert all(abs(a["end"] - b["start"]) < .002 for a, b in zip(present, present[1:]))
        assert abs(sum(e["end"]-e["start"] for e in events) - (original["end"]-original["start"])) < .01
        loops.append({**original, "groups": groups, "released": released})

    batches = []
    for batch in range(1, 6):
        ranks = []
        for r in range(3):
            current = next(l for l in loops if l["r"] == r and l["current"] == batch)
            old = next(l for l in loops if l["r"] == r and l["old"] == batch)
            release = next(item for item in data["releases"] if item["r"] == r and batch in item["batches"])
            select = graph[f'{r}:{current["n"]}:select']
            gpu = graph[f'{r}:{current["n"]}:gpu']
            kv = graph[f'{r}:{old["n"]}:kv']
            cleanup = graph[f'{r}:{release["n"]}:release']
            admission = graph[f'{r}:2:recv_bc']
            assert admission["owner"] == "M1–M5 准入"
            points = [graph[f'{r}:0:recv_req']["start"], select["start"], gpu["start"], gpu["end"], kv["start"], kv["end"], cleanup["end"]]
            assert all(a <= b for a, b in zip(points, points[1:]))
            # The release roster records entry into cleanup, not its completion.
            assert abs(cleanup["start"] - release["time"]) < .002
            phases = [{"start": points[i], "end": points[i+1], "loop": [2,current["n"],current["n"],old["n"],old["n"],release["n"]][i]} for i in range(6)]
            ranks.append({"r": r, "phases": phases, "admitted": admission["end"], "releaseStart": cleanup["start"], "currentLoop": current["n"], "resultLoop": old["n"], "releaseLoop": release["n"]})
        assert all(ranks[r]["phases"][2]["end"] <= ranks[r+1]["phases"][2]["start"] for r in range(2))
        batches.append({"batch": batch, "ranks": ranks})
    result = {"sourceCommit": "72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a", "sourceModelSha256": hashlib.sha256(source.encode()).hexdigest(), "end": data["end"], "loops": loops, "batches": batches}
    return result


def main():
    outer = EmbeddedModel()
    outer.feed((PAGE / "index.html").read_text(encoding='utf-8'))
    inner = EmbeddedModel()
    inner.feed(outer.srcdoc)
    source = "".join(inner.chunks)
    data = json.loads(source)
    result = derive_quick(data, source)
    (PAGE / "quick-data.js").write_text("// Generated by scripts/build_pp_quick_data.py from index.html; do not edit by hand.\nwindow.PP_QUICK_DATA = " + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding='utf-8')
    print("Derived", len(result["loops"]), "loops and", len(result["batches"]), "complete batch lifecycles; timeline coverage checked.")


if __name__ == "__main__":
    main()
