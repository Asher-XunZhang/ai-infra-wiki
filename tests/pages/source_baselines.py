"""Exact course-directory overrides; other artifacts retain their own default."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVING_REVISION = '72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a'
SERVING_DIRECTORIES = frozenset({
    'memory-budget', 'hicache-loadback', 'deployment-lab',
    'benchmark-lab', 'trace-lab', 'serving-labs',
})


def source_revision(path, default):
    try:
        relative = Path(path).resolve().relative_to(ROOT / 'pages/sglang')
    except ValueError:
        return default
    return SERVING_REVISION if len(relative.parts) > 1 and relative.parts[0] in SERVING_DIRECTORIES else default
