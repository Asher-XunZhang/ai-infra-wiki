#!/usr/bin/env python3
"""Build GitHub Pages with content-versioned local CSS and JavaScript.

HTML keeps its public paths; changing an asset changes its filename, so a browser
with a fresh old cache entry cannot combine the new HTML with an old asset.
The source pages remain directly previewable without a build.
"""
import hashlib
import json
import os
import re
import shutil
from html import escape, unescape
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from build_learning_navigation import build as check_learning_navigation
from build_foundations import build as check_foundations

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pages"
OUTPUT = ROOT / ".pages-dist"
PUBLIC_IMAGES = ROOT / "images"
PUBLIC_IMAGE_TOPICS = ("attention-variants",)
REFERENCE = re.compile(r"(?P<prefix>\b(?:src|href)\s*=\s*)(?P<quote>[\"'])(?P<url>[^\"']+)(?P=quote)")
PUBLIC_ROOT = re.compile(r'data-public-root="[^"]*"')


def build():
    check_learning_navigation(check=True)
    check_foundations(check=True)
    # This directory is reserved for generated deployment output.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    shutil.copytree(SOURCE, OUTPUT)
    # Pages are built one level shallower than their sources, so the repository
    # image root becomes a public asset directory in the deployment output.
    # Only topics referenced by the built pages are copied.
    for topic in PUBLIC_IMAGE_TOPICS:
        shutil.copytree(PUBLIC_IMAGES / topic, OUTPUT / "images" / topic)
    assets = {}
    release_hash = hashlib.sha256(Path(__file__).read_bytes())
    for source in sorted(p for p in SOURCE.rglob("*") if p.is_file()):
        release_hash.update(str(source.relative_to(SOURCE)).encode() + b"\0")
        release_hash.update(source.read_bytes() + b"\0")
    for topic in PUBLIC_IMAGE_TOPICS:
        for source in sorted(p for p in (PUBLIC_IMAGES / topic).rglob("*") if p.is_file()):
            release_hash.update(str(source.relative_to(ROOT)).encode() + b"\0")
            release_hash.update(source.read_bytes() + b"\0")
    release = release_hash.hexdigest()[:16]

    def version_reference(match, page):
        url = urlsplit(unescape(match["url"]))
        if url.scheme or url.netloc or not url.path:
            return match[0]
        source = (page.parent / unquote(url.path)).resolve()
        relative = source.relative_to(SOURCE)
        if not url.path.endswith((".css", ".js")):
            # Reading URLs are stable; only CSS/JS filenames carry fingerprints.
            target = source / "index.html" if source.is_dir() else source
            if not match["prefix"].startswith("href") or target.suffix != ".html" or not target.is_file():
                return match[0]
            query = [(k, v) for k, v in parse_qsl(url.query, keep_blank_values=True) if k != "v"]
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
        depth = len(page.relative_to(SOURCE).parent.parts)
        root = '../' * depth or './'
        # Keep the tiny update runtime in HTML: a cached document must still
        # offer recovery after its old fingerprinted assets leave the deployment.
        # It runs before lesson controllers register their reading checkpoints.
        updater = (SOURCE / 'site-update.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
        bootstrap = (f'<meta name="site-release" content="{release}">'
                     f'<meta name="site-root" content="{root}">'
                     f'<script data-site-update-runtime>{updater}</script>')
        source = page.read_text(encoding='utf-8')
        source, count = re.subn(r'(<meta charset="utf-8"\s*/?>)', lambda m: m[0] + bootstrap, source, count=1, flags=re.I)
        if count != 1:
            raise ValueError(f'{page}: expected charset before update bootstrap')
        rendered = REFERENCE.sub(lambda m: version_reference(m, page), source)
        rendered = PUBLIC_ROOT.sub(f'data-public-root="{"../" * depth}"', rendered)
        (OUTPUT / page.relative_to(SOURCE)).write_text(rendered, encoding='utf-8')
    (OUTPUT / 'site-release.json').write_text(json.dumps({'schema': 1, 'release': release}) + '\n', encoding='utf-8')
    print(f"Built {len(pages)} pages with {len(assets)} content-versioned assets in {OUTPUT}; release={release}")


if __name__ == "__main__":
    build()
