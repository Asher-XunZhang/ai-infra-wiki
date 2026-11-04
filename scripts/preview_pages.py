#!/usr/bin/env python3
"""Preview the versioned Pages build locally: python3 scripts/preview_pages.py.

Serving pages/ directly can mix fresh HTML with old cached JavaScript/CSS.
Use the same content-hashed assets as publication, and disable preview caching.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-build", action="store_true", help="Serve the existing .pages-dist build")
    args = parser.parse_args()
    if not args.no_build:
        from build_pages import build
        build()
    output = Path(__file__).resolve().parents[1] / ".pages-dist"
    if not (output / "index.html").is_file():
        parser.error("No Pages build found; run without --no-build first")
    handler = partial(PreviewHandler, directory=str(output))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Pages preview: http://127.0.0.1:{server.server_port}/ (versioned assets, no-store)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
