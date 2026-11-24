"""Check pinned source anchors; optional inputs are read without engine imports."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def definitions(tree, prefix=''):
    for node in getattr(tree, 'body', []):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + node.name, node.lineno
            yield from definitions(node, prefix + node.name + '.')


def check(sglang_source=None, vllm_cache=None):
    data = json.loads((ROOT / 'pages/interactive-lessons/sources.json').read_text())
    count = verified = 0
    for records in data.values():
        for record in records:
            count += 1
            expected = 'https://github.com/{repo}/blob/{commit}/{path}#L{line}'.format(**record)
            assert record['url'] == expected
            assert record['commit'] in {'72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a', '4ba787ec9959e566d3b587a796dff07673324da5'}
            raw = None
            if record['repo'] == 'sgl-project/sglang' and sglang_source:
                raw = subprocess.check_output(['git', '-C', str(sglang_source), 'show', record['commit'] + ':' + record['path']])
            elif record['repo'] == 'vllm-project/vllm' and vllm_cache:
                raw = (vllm_cache / record['path']).read_bytes()
            if raw is not None:
                assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']
                if record['path'].endswith('.py'):
                    assert (record['symbol'], record['line']) in set(definitions(ast.parse(raw))), record
                else:
                    assert b'select_worker' in raw.splitlines()[record['line'] - 1]
                verified += 1
    print(f'Checked {count} pinned source declarations; verified {verified} file hashes and function/class anchors. No engine imports or GPU execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang-source', type=Path)
    parser.add_argument('--vllm-cache', type=Path)
    args = parser.parse_args()
    check(args.sglang_source, args.vllm_cache)
