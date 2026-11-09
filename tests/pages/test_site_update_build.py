"""Exercise release metadata, stable reading links and independent asset hashes."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_pages as build


class ReleaseBuildTest(unittest.TestCase):
    def test_stable_links_and_content_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, output = root / 'pages', root / 'dist'
            source.mkdir()
            (source / 'site-update.js').write_text('/* updater */')
            (source / 'site.css').write_text('body{color:green}')
            (source / 'app.js').write_text('const version=1;')
            (source / 'lesson.html').write_text('<html><head><meta charset="utf-8"></head><body>Lesson</body></html>')
            (source / 'index.html').write_text('''<html><head><meta charset="utf-8"><link href="site.css" rel="stylesheet"><script src="app.js" defer></script></head><body>
<a href="lesson.html?v=old&amp;step=4&amp;color=loop&amp;learn=first#pp">Read</a>
<a href="https://example.com/guide?v=original#part">External</a></body></html>''')
            with patch.multiple(build, ROOT=root, SOURCE=source, OUTPUT=output, PUBLIC_IMAGE_TOPICS=(), check_learning_navigation=lambda **_: None, check_foundations=lambda **_: None):
                build.build()
                first = (output / 'index.html').read_text()
                manifest = json.loads((output / 'site-release.json').read_text())
                self.assertIn('lesson.html?step=4&amp;color=loop&amp;learn=first#pp', first)
                self.assertIn('https://example.com/guide?v=original#part', first)
                self.assertIn(f'name="site-release" content="{manifest["release"]}"', first)
                self.assertEqual(manifest['schema'], 1)
                self.assertLess(first.index('data-site-update-runtime'), first.index('app.'))
                self.assertIn('<script data-site-update-runtime>/* updater */</script>', first)
                refs = re.findall(r'(?:src|href)="([^"?]+\.[0-9a-f]{16}\.(?:css|js))"', first)
                self.assertEqual(len(refs), 2)
                for ref in refs:
                    self.assertEqual(Path(ref).name.split('.')[-2], hashlib.sha256((output / ref).read_bytes()).hexdigest()[:16])
                css = next(ref for ref in refs if ref.endswith('.css'))
                (source / 'app.js').write_text('const version=2;')
                build.build()
                second = (output / 'index.html').read_text()
                self.assertIn(css, second)
                self.assertNotEqual(manifest['release'], json.loads((output / 'site-release.json').read_text())['release'])
                self.assertNotIn(next(ref for ref in refs if '/app.' in '/' + ref), second)


if __name__ == '__main__':
    unittest.main()
