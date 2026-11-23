"""Check each implementation course against its declared fixed source baseline."""
from pathlib import Path
import json
import re

from source_baselines import source_revision, SERVING_DIRECTORIES, SERVING_REVISION

ROOT = Path(__file__).resolve().parents[2]
REVISION = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
count = 0
for p in (ROOT / 'pages/sglang').rglob('*'):
    if p.suffix not in ('.js', '.html', '.json'):
        continue
    expected = source_revision(p, REVISION)
    text = p.read_text()
    for rev in re.findall(r'sgl-project/sglang/(?:blob|tree)/([0-9a-f]{40})', text):
        assert rev == expected, (p, rev, expected)
        count += 1
    for rev in re.findall(r'(?:const\s+(?:revision|commit)|revision\s*:)\s*=*\s*[\'\"]([0-9a-f]{40})', text):
        assert rev == expected, (p, rev, expected)
notes = json.loads((ROOT / 'pages/sglang/design-notes.json').read_text())
assert all(entry.get('revision', REVISION) == REVISION for entry in notes.values())
for path in ('pd-execution/timeline.js', 'system-overview/opening.js'):
    text = (ROOT / 'pages/sglang' / path).read_text()
    hashes = set(re.findall(r'\b[0-9a-f]{40}\b', text))
    assert hashes == {REVISION}, (path, hashes)
assert source_revision(ROOT / 'pages/sglang/other/deployment-lab/index.html', REVISION) == REVISION
assert source_revision(ROOT / 'pages/foundations/deployment-lab/index.html', REVISION) == REVISION
for directory in SERVING_DIRECTORIES:
    assert source_revision(ROOT / 'pages/sglang' / directory / 'index.html', REVISION) == SERVING_REVISION
print(f'PASS: {count} static SGLang source links and declarations match course baselines; existing design notes remain on {REVISION[:10]}.')
