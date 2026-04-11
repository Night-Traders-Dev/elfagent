import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.doc_tools import extract_pdf_tables, parse_pdf
from tools.patch_tools import apply_patch, create_diff
from tools.rag_tools import index_directory, semantic_search_raw
from tools.structured_tools import parse_structured_file
from tools.vision_tools import describe_image


class FakeEmbeddingModel:
    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "memory" in lowered or "retrieval" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeTable:
    def extract(self):
        return [["name", "value"], ["mode", "debug"]]


class FakeTableFinder:
    tables = [FakeTable()]


class FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self, mode):
        return self._text

    def find_tables(self):
        return FakeTableFinder()


class FakeDocument:
    def __init__(self, pages):
        self._pages = pages

    def __len__(self):
        return len(self._pages)

    def load_page(self, index):
        return self._pages[index]

    def close(self):
        return None


class LocalToolTests(unittest.TestCase):
    def test_index_directory_and_semantic_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "boot.md").write_text("Bootloader memory retrieval and stack setup notes.\n")
            (root / "ui.md").write_text("Dashboard theme and button spacing details.\n")
            index_path = root / "rag.json"

            with patch("tools.rag_tools._get_embedding_model", return_value=FakeEmbeddingModel()):
                result = index_directory(str(root), index_path=str(index_path))
                self.assertIn("Indexed", result)
                matches = semantic_search_raw("memory retrieval", index_path=str(index_path))

            self.assertTrue(matches)
            self.assertTrue(matches[0]["path"].endswith("boot.md"))

    def test_apply_patch_updates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "note.txt"
            target.write_text("old line\n", encoding="utf-8")
            diff = create_diff("old line\n", "new line\n", before_label="note.txt", after_label="note.txt")
            output = apply_patch(diff, working_directory=tmpdir)
            self.assertIn("[exit 0]", output)
            self.assertEqual(target.read_text(encoding="utf-8"), "new line\n")

    def test_parse_structured_file_normalizes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"mode": "debug", "enabled": True}), encoding="utf-8")
            parsed = parse_structured_file(str(path))
            self.assertIn('"mode": "debug"', parsed)
            self.assertIn('"enabled": true', parsed)

    def test_parse_pdf_and_extract_tables_use_pymupdf_adapter(self):
        fake_fitz = type(
            "FakeFitz",
            (),
            {
                "open": staticmethod(
                    lambda path: FakeDocument([FakePage("Page one text"), FakePage("Page two text")])
                )
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "datasheet.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with patch.dict(sys.modules, {"fitz": fake_fitz}):
                parsed = parse_pdf(str(pdf_path))
                tables = extract_pdf_tables(str(pdf_path))

        self.assertIn("--- Page 1 ---", parsed)
        self.assertIn("Page two text", parsed)
        self.assertIn("mode | debug", tables)

    def test_describe_image_uses_ollama_vision_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "board.png"
            image_path.write_bytes(b"not-a-real-png")

            with patch(
                "tools.vision_tools.requests.post",
                return_value=FakeResponse(payload={"response": "Connector J1 and a reset button are visible."}),
            ) as mocked_post:
                description = describe_image(str(image_path))

        self.assertIn("Connector J1", description)
        self.assertEqual(mocked_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
