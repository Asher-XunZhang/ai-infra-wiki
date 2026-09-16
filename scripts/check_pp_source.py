"""Check teaching artifacts and, optionally, the pinned local SGLang checkout.

No SGLang imports, network access, or GPU execution. This detects baseline and
anchor drift; it is not a proof that the illustrative model simulates runtime.
"""
import argparse
import ast
import json
import re
import subprocess
from html import unescape
from pathlib import Path

from build_pp_quick_data import EmbeddedModel, PAGE
from pp_source_baseline import SOURCE_ANCHORS, SOURCE_COMMIT, SOURCE_FILES, SOURCE_SHORT, STEP_GUIDE_ANCHORS

ROOT = Path(__file__).resolve().parents[1]


def check_artifacts():
    outer = EmbeddedModel()
    outer.feed((PAGE / 'index.html').read_text(encoding='utf-8'))
    inner = EmbeddedModel()
    inner.feed(outer.srcdoc)
    models = [json.loads(''.join(inner.chunks))]
    quick_text = (PAGE / 'quick-data.js').read_text(encoding='utf-8')
    quick = json.loads(quick_text.split('window.PP_QUICK_DATA = ', 1)[1].strip().removesuffix(';'))
    assert quick['sourceCommit'] == SOURCE_COMMIT
    for path in (PAGE / 'scenarios').glob('*.json'):
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['quick']['sourceCommit'] == SOURCE_COMMIT, path
        models.append(payload['model'])
    refs = set()
    for model in models:
        assert model['sourceCommit'] == SOURCE_COMMIT and model['baseline'] == SOURCE_SHORT
        refs.update(node['ref'] for node in [*model['graph'].values(), *model['events']])
    assert refs == SOURCE_ANCHORS.keys(), ('source anchors changed', refs ^ SOURCE_ANCHORS.keys())
    guide = (PAGE / 'step-guide.js').read_text(encoding='utf-8')
    guide_refs = set(re.findall(r"'((?:pp|cache|controller|scheduler|policy|prefill):\d+)'", guide))
    assert guide_refs - SOURCE_ANCHORS.keys() == STEP_GUIDE_ANCHORS.keys(), 'Review changed step-guide anchors'

    # Both visible badges and source hyperlinks must identify the reviewed SHA.
    links = set()
    for path in [*(ROOT / 'pages').rglob('*.html'), ROOT / 'sglang/PD Prefill PP loop 步骤详解.md']:
        text = unescape(path.read_text(encoding='utf-8'))
        for commit, source, first, last in re.findall(
            r'https://github.com/sgl-project/sglang/blob/([0-9a-f]{40})/'
            r'(python/[^\s"<>\)]+?)#L(\d+)(?:-L(\d+))?', text
        ):
            assert commit == SOURCE_COMMIT, (path, commit)
            links.add((source, int(first), int(last or first)))
    for name in ('index.html', 'quick.html', 'notes.html'):
        assert SOURCE_SHORT in (PAGE / name).read_text(encoding='utf-8'), name
    assert '${data.sourceCommit}' in outer.srcdoc, 'Detail links must use the loaded model baseline'
    print(f'Checked {len(models)} models, {len(refs)} graph anchors, {len(links)} page source links.')
    return links


def check_checkout(root, links):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == SOURCE_COMMIT, f'Local HEAD {head} differs; review before changing SOURCE_COMMIT'
    changed = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--', 'python/sglang/srt'], text=True
    ).strip()
    assert not changed, 'Local SGLang source has changes beyond the pinned commit: ' + changed
    srt = root / 'python/sglang/srt'
    for ref, expected in (SOURCE_ANCHORS | STEP_GUIDE_ANCHORS).items():
        key, line = ref.split(':')
        actual = (srt / SOURCE_FILES[key]).read_text(encoding='utf-8').splitlines()[int(line) - 1].strip()
        assert actual == expected, (ref, expected, actual)
    for path, first, last in links:
        lines = (root / path).read_text(encoding='utf-8').splitlines()
        assert 1 <= first <= last <= len(lines), (path, first, last)

    def method(path, name):
        tree = ast.parse((srt / path).read_text(encoding='utf-8'))
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

    def calls(node):
        return [(n.lineno, ast.unparse(n.func)) for n in ast.walk(node) if isinstance(n, ast.Call)]

    def ordered(node, *names):
        positions = [min(line for line, name in calls(node) if name == target) for target in names]
        assert positions == sorted(set(positions)), (node.name, names, positions)

    loop = method(SOURCE_FILES['pp'], 'event_loop_pp_disagg_prefill')
    ordered(loop, 'self.process_prefill_chunk', 'self._process_hicache_events', 'self.get_new_batch_prefill')
    selection = method(SOURCE_FILES['scheduler'], '_get_new_batch_prefill_raw')
    assert not any(name == 'self.tree_cache.check_hicache_events' for _, name in calls(selection))
    admission = method(SOURCE_FILES['policy'], 'add_one_req')
    ordered(admission, 'self._select_prefill_admission',
            'self.prefill_delayer_single_pass.negotiate_should_allow_prefill',
            'self.tree_cache.init_load_back', 'self._commit_prefill_admission')
    events = method(SOURCE_FILES['cache'], 'check_hicache_events')
    ordered(events, 'self._drain_async_work', 'self._sync_hicache_ready_counts',
            'self.writing_check', 'self.loading_check')
    cleanup = method(SOURCE_FILES['prefill'], 'process_disagg_prefill_inflight_queue')
    ordered(cleanup, 'release_kv_cache', 'self.tree_cache.finish', 'req.disagg_kv_sender.clear')
    finish = method('mem_cache/base_prefix_cache.py', 'finish')
    assert any(isinstance(n, ast.If) and ast.unparse(n.test) == 'outcome != CacheRequestOutcome.SUCCESS'
               for n in ast.walk(finish)), 'Recheck SUCCESS finish semantics'
    print(f'Checked local {SOURCE_SHORT}: exact anchors and cache/admission/cleanup call order.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optional local SGLang checkout; must match the reviewed commit')
    args = parser.parse_args()
    source_links = check_artifacts()
    if args.source_root:
        check_checkout(args.source_root.resolve(), source_links)
