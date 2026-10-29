"""Basic behavioral tests for the local tool server; no audio device required."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fduplex_mcp as server


class LocalToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.patcher1 = patch.object(server, "READ_ROOTS", [self.root])
        self.patcher2 = patch.object(server, "WRITE_ROOTS", [self.root])
        self.patcher3 = patch.object(server, "WORKSPACE", self.root)
        self.patcher4 = patch.object(server, "DB", self.root / "index.db")
        for item in (self.patcher1, self.patcher2, self.patcher3, self.patcher4):
            item.start()
            self.addCleanup(item.stop)

    def test_write_read_patch(self):
        target = self.root / "note.txt"
        server.write_file(str(target), "hello\nworld\n")
        self.assertIn("hello", server.read_file(str(target)))
        diff = "--- a/note.txt\n+++ b/note.txt\n@@ -1,2 +1,2 @@\n hello\n-world\n+fduplex\n"
        server.apply_patch(str(target), diff)
        self.assertEqual(target.read_text(), "hello\nfduplex\n")

    def test_patch_rejects_incorrect_context_without_modification(self):
        target = self.root / "a.txt"
        target.write_text("a\n")
        with self.assertRaises(ValueError):
            server.apply_patch(str(target), "@@ -1 +1 @@\n-b\n+c\n")
        self.assertEqual(target.read_text(), "a\n")

    def test_outside_write_root_rejected(self):
        with self.assertRaises(ValueError):
            server.write_file("/tmp/outside-fduplex-workspace.txt", "secret")

    def test_index_and_search(self):
        target = self.root / "research.txt"
        target.write_text("contact center efficiency\n")
        self.assertGreater(server.index_files()["indexed"], 0)
        found = server.search_files("efficiency")
        self.assertEqual(found[0]["path"], str(target))

    def test_terminal_disallows_shell(self):
        with self.assertRaises(ValueError):
            asyncio.run(server.run_terminal("sh -c whoami"))

    def test_symlink_escape(self):
        link = self.root / "escape"
        link.symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            server.read_file(str(link))


if __name__ == "__main__":
    unittest.main()
