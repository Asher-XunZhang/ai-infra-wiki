"""Exercise migration entries through the full documentation checker."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "learning_docs", REPO / "scripts/check_learning_docs.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class MigrationChecks(unittest.TestCase):
    def check(self, extra):
        files = {
            "README.md": "[主题](主题/README.md)\n",
            "主题/README.md": "[领域](领域/README.md)\n",
            "主题/领域/README.md": "[主文](main.md)\n",
            "主题/领域/main.md": '# Main\n\n<a id="entry"></a>\n正文。\n',
            **extra,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            output, errors = io.StringIO(), io.StringIO()
            names = "\0".join(files).encode()
            with (
                patch.object(CHECKER, "ROOT", root),
                patch.object(CHECKER.subprocess, "check_output", return_value=names),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                failed = CHECKER.main()
            return failed, errors.getvalue()

    def test_short_entry_needs_no_second_navigation_listing(self):
        failed, errors = self.check({
            "主题/领域/old.md": "# Old\n\n"
            "<!-- article-redirect: main.md#entry -->\n\n"
            "[已迁入主文](main.md#entry)\n",
        })
        self.assertFalse(failed, errors)

    def test_marker_anchor_is_checked_even_if_the_reader_link_is_valid(self):
        failed, errors = self.check({
            "主题/领域/old.md": "# Old\n\n"
            "<!-- article-redirect: main.md#missing -->\n\n"
            "[主文](main.md#entry)\n",
        })
        self.assertTrue(failed)
        self.assertIn("missing migration anchor", errors)

    def test_two_retired_urls_cannot_point_at_each_other(self):
        failed, errors = self.check({
            "主题/领域/old.md": "<!-- article-redirect: other.md -->\n[正文](other.md)\n",
            "主题/领域/other.md": "<!-- article-redirect: old.md -->\n[正文](old.md)\n",
        })
        self.assertTrue(failed)
        self.assertIn("migration chain or cycle", errors)

    def test_marker_cannot_hide_an_unlisted_long_article(self):
        failed, errors = self.check({
            "主题/领域/old.md": "<!-- article-redirect: main.md -->\n[主文](main.md)\n"
            + "独立正文。\n" * 25,
        })
        self.assertTrue(failed)
        self.assertIn("still contains a long article", errors)

    def test_normal_articles_still_require_navigation(self):
        failed, errors = self.check({"主题/领域/unlisted.md": "# Article\n正文。\n"})
        self.assertTrue(failed)
        self.assertIn("missing article entry", errors)


if __name__ == "__main__":
    unittest.main()
