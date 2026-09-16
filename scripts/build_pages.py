#!/usr/bin/env python3
"""Build GitHub Pages with content-versioned local CSS and JavaScript.

HTML keeps its public paths; changing an asset changes its filename, so a browser
with a fresh old cache entry cannot combine the new HTML with an old asset.
The source pages remain directly previewable without a build.
"""
import hashlib
import os
import re
import shutil
from html import escape, unescape
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pages"
OUTPUT = ROOT / ".pages-dist"
REFERENCE = re.compile(r"(?P<prefix>\b(?:src|href)\s*=\s*)(?P<quote>[\"'])(?P<url>[^\"']+)(?P=quote)")


def build():
    # This directory is reserved for generated deployment output.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    shutil.copytree(SOURCE, OUTPUT)
    assets = {}
    release_hash = hashlib.sha256()
    for source in sorted(p for p in SOURCE.rglob("*") if p.is_file()):
        release_hash.update(str(source.relative_to(SOURCE)).encode() + b"\0")
        release_hash.update(source.read_bytes() + b"\0")
    release = release_hash.hexdigest()[:16]

    def version_reference(match, page):
        url = urlsplit(unescape(match["url"]))
        if url.scheme or url.netloc or not url.path:
            return match[0]
        source = (page.parent / unquote(url.path)).resolve()
        relative = source.relative_to(SOURCE)
        if not url.path.endswith((".css", ".js")):
            # Links from the fresh homepage must not reopen a cached old lesson.
            target = source / "index.html" if source.is_dir() else source
            if not match["prefix"].startswith("href") or target.suffix != ".html" or not target.is_file():
                return match[0]
            query = [(k, v) for k, v in parse_qsl(url.query, keep_blank_values=True) if k != "v"]
            query.append(("v", release))
            target = urlunsplit(("", "", url.path, urlencode(query), url.fragment))
            return match["prefix"] + match["quote"] + escape(target, quote=True) + match["quote"]
        if source not in assets:
            content = source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()[:16]
            versioned = relative.with_name(f"{source.stem}.{digest}{source.suffix}")
            (OUTPUT / versioned).write_bytes(content)
            assets[source] = versioned
        path = os.path.relpath(OUTPUT / assets[source], OUTPUT / page.relative_to(SOURCE).parent)
        target = urlunsplit(("", "", quote(Path(path).as_posix(), safe="/"), url.query, url.fragment))
        return match["prefix"] + match["quote"] + escape(target, quote=True) + match["quote"]

    pages = sorted(SOURCE.rglob("*.html"))
    for page in pages:
        rendered = REFERENCE.sub(lambda m: version_reference(m, page), page.read_text(encoding='utf-8'))
        (OUTPUT / page.relative_to(SOURCE)).write_text(rendered, encoding='utf-8')
    print(f"Built {len(pages)} pages with {len(assets)} content-versioned assets in {OUTPUT}; release={release}")


if __name__ == "__main__":
    build()
