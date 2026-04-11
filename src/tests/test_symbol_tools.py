import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.symbol_tools import (
    find_symbol_references,
    list_asm_symbols,
    list_c_symbols,
    read_binary_symbols,
)


class FakeCompletedProcess:
    def __init__(self, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr


class SymbolToolTests(unittest.TestCase):
    def test_list_c_symbols_finds_functions_and_macros(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "boot.c"
            path.write_text(
                "#define STACK_TOP 0x2000\n"
                "typedef struct BootState BootState;\n"
                "void setup_stack(void);\n"
                "static void init_boot(void) {\n"
                "}\n",
                encoding="utf-8",
            )
            output = list_c_symbols(str(path))

        self.assertIn("macro: STACK_TOP", output)
        self.assertIn("declaration: setup_stack", output)
        self.assertIn("function: init_boot", output)

    def test_list_asm_symbols_finds_labels_and_globals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "startup.S"
            path.write_text(
                ".globl _start\n"
                ".type _start, @function\n"
                "_start:\n"
                "  j reset_handler\n"
                "reset_handler:\n",
                encoding="utf-8",
            )
            output = list_asm_symbols(str(path))

        self.assertIn("global: _start", output)
        self.assertIn("label: reset_handler", output)

    def test_find_symbol_references_reports_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "boot.c").write_text("void boot(void) { reset_handler(); }\n", encoding="utf-8")
            (root / "startup.S").write_text("reset_handler:\n", encoding="utf-8")
            output = find_symbol_references("reset_handler", str(root))

        self.assertIn("boot.c:1", output)
        self.assertIn("startup.S:1", output)

    def test_read_binary_symbols_wraps_nm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "firmware.elf"
            binary.write_bytes(b"\x7fELF")
            with patch(
                "tools.symbol_tools.subprocess.run",
                return_value=FakeCompletedProcess(stdout="00000000 T reset_handler\n"),
            ):
                output = read_binary_symbols(str(binary))

        self.assertIn("nm output", output)
        self.assertIn("reset_handler", output)


if __name__ == "__main__":
    unittest.main()
