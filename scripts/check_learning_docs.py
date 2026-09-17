#!/usr/bin/env python3
"""Check repository-local learning links, images, and navigation without network access.

Uses the repository's inline/reference Markdown links and HTML href/src attributes.
Fenced examples and inline code are excluded. Remote source URLs are not fetched;
links to this Wiki's GitHub main branch are checked against the local files.
"""
import re
import subprocess
import sys
from collections import Counter
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIRECTORIES = {"images", "pages", "scripts"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".ico"}
INLINE = re.compile(r'!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)')
REFERENCE = re.compile(r'^\s*\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)', re.M)
ATTRIBUTE = re.compile(r'\b(?:href|src)\s*=\s*["\x27]([^"\x27]+)["\x27]')


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker:
            value = marker[1]
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    return "\n".join(lines)


def destinations(path, text):
    if path.suffix == '.md':
        text = prose(text)
        text = re.sub(r'(`+)([^`\n]*?)\1', '', text)
        for pattern in (INLINE, REFERENCE):
            for match in pattern.finditer(text):
                token = match[1]
                if token.startswith('<'):
                    token = token[1:token.index('>')]
                else:
                    token = re.split(r'\s+["\x27]', token, maxsplit=1)[0]
                yield token.strip()
    for match in ATTRIBUTE.finditer(text):
        yield unescape(match[1])


def anchors(path, text):
    ids = set(re.findall(r'\b(?:id|name)=["\x27]([^"\x27]+)["\x27]', text))
    if path.suffix == '.md':
        used = Counter()
        for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', prose(text), re.M):
            heading = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', heading)
            heading = re.sub(r'<[^>]*>', '', heading)
            slug = re.sub(r'[^\w\- ]', '', unescape(heading).lower()).replace(' ', '-')
            number = used[slug]
            ids.add(slug + (f'-{number}' if number else ''))
            used[slug] += 1
    return ids


def resolve(source, token):
    url = urlsplit(unescape(token))
    if url.scheme or url.netloc:
        match = re.match(r'^/Asher-XunZhang/ai-infra-wiki/(?:blob|tree)/main/(.*)$', unquote(url.path), re.I)
        if url.netloc.lower() != 'github.com' or not match:
            return None
        target = (ROOT / match[1]).resolve()
    else:
        target = (source.parent / unquote(url.path)).resolve() if url.path else source
    if target.is_dir():
        for name in ('README.md', 'index.html'):
            if (target / name).is_file():
                target = target / name
                break
    return target, unquote(url.fragment)


def main():
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    files = sorted({ROOT / name for name in names if name and (ROOT / name).is_file()})
    docs = {p: p.read_text(encoding='utf-8') for p in files if p.suffix in {'.md', '.html'}}
    ids = {p: anchors(p, text) for p, text in docs.items()}
    outgoing = {}
    failures = []
    checked = 0
    for source, text in docs.items():
        outgoing[source] = set()
        for token in destinations(source, text):
            resolved = resolve(source, token)
            if resolved is None:
                continue
            checked += 1
            target, fragment = resolved
            outgoing[source].add(target)
            if not target.exists():
                failures.append(f'{source.relative_to(ROOT)}: missing target: {token}')
            elif fragment and target in ids and fragment not in ids[target]:
                # GitHub also accepts user-content-prefixed explicit anchors.
                if fragment.removeprefix('user-content-') not in ids[target]:
                    failures.append(f'{source.relative_to(ROOT)}: missing anchor: {token}')
        if source.suffix == '.md' and re.search(r'!\[[^\]]*\]\(\s*<?https?://', prose(text)):
            failures.append(f'{source.relative_to(ROOT)}: remote Markdown image')
    for path in files:
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in IMAGE_SUFFIXES and rel.parts[0] != 'images':
            failures.append(f'{rel}: image outside top-level images/')
    # Discover learning topics so adding another system does not require editing
    # a fixed allowlist. Hidden configuration and support directories are excluded.
    topics = sorted({
        p.relative_to(ROOT).parts[0] for p in docs
        if p.suffix == '.md' and len(p.relative_to(ROOT).parts) > 1
        and p.relative_to(ROOT).parts[0] not in SUPPORT_DIRECTORIES
        and not p.relative_to(ROOT).parts[0].startswith('.')
    })
    learning = [p for p in docs if p.suffix == '.md' and p.relative_to(ROOT).parts[0] in topics]
    for topic in topics:
        index = ROOT / topic / 'README.md'
        if index not in outgoing.get(ROOT / 'README.md', set()):
            failures.append(f'{topic}: missing homepage entry')
    directories = set()
    for path in learning:
        if 'source-study' in path.relative_to(ROOT).parts:
            continue
        parent = path.parent
        while parent.parent != ROOT:
            directories.add(parent)
            parent = parent.parent
    for directory in sorted(directories):
        index = directory.parent / 'README.md'
        if directory / 'README.md' not in outgoing.get(index, set()):
            failures.append(f'{directory.relative_to(ROOT)}: missing parent navigation entry')
    for path in learning:
        if path.name == 'README.md':
            continue
        if 'source-study' in path.relative_to(ROOT).parts:
            # Existing ordered course uses one central chapter index.
            index = ROOT / 'sglang/source-study/README.md'
            if path.parent.name == 'architecture':
                index = path.parent / 'README.md'
        else:
            index = path.parent / 'README.md'
            if path.parent == ROOT / path.relative_to(ROOT).parts[0]:
                failures.append(f'{path.relative_to(ROOT)}: article is flat under topic root')
        if path not in outgoing.get(index, set()):
            failures.append(f'{path.relative_to(ROOT)}: missing article entry in {index.relative_to(ROOT)}')
    for failure in failures:
        print(f'ERROR: {failure}', file=sys.stderr)
    print(f'Checked {len(docs)} Markdown/HTML files, {checked} local link occurrences, '
          f'{len(learning)} learning documents, image placement and navigation coverage; '
          f'{len(failures)} errors.')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
