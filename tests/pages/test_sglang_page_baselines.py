"""Third-chapter implementation links share one commit; historical documents are independent."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
REVISION = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
count = 0
for p in (ROOT / 'pages/sglang').rglob('*'):
    if p.suffix not in ('.js', '.html', '.json'):
        continue
    text = p.read_text()
    for rev in re.findall(r'sgl-project/sglang/(?:blob|tree)/([0-9a-f]{40})', text):
        assert rev == REVISION, (p, rev)
        count += 1
    for rev in re.findall(r'(?:const\s+(?:revision|commit)|revision\s*:)\s*=*\s*[\'\"]([0-9a-f]{40})', text):
        assert rev == REVISION, (p, rev)
notes = json.loads((ROOT / 'pages/sglang/design-notes.json').read_text())
assert all(entry.get('revision', REVISION) == REVISION for entry in notes.values())
for path in ('pd-execution/timeline.js', 'system-overview/opening.js'):
    text = (ROOT / 'pages/sglang' / path).read_text()
    hashes = set(re.findall(r'\b[0-9a-f]{40}\b', text))
    assert hashes == {REVISION}, (path, hashes)
print(f'PASS: {count} static SGLang source links, dynamic baseline declarations and design notes agree on {REVISION[:10]}.')
