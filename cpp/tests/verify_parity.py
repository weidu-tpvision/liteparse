"""Verify parity between C++ ocria_cli and Python LiteParse (ocria-nano)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from liteparse import LiteParse

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(
    os.environ.get("LITEPARSE_MODEL_DIR", str(REPO_ROOT / "tests" / "models"))
) / "ocria-nano"
CLI_BIN = REPO_ROOT / "cpp" / "build" / "ocria_cli"
DATA_DIR = REPO_ROOT / "integration_tests_data"


def run_cpp(image_path: Path, long_side: int = 1600, threads: int = 4, repeat: int = 3) -> dict:
    cmd = [
        str(CLI_BIN),
        str(MODEL_DIR),
        str(image_path),
        "--long-side", str(long_side),
        "--threads", str(threads),
        "--repeat", str(repeat),
        "--json",
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(res.stdout)


def main() -> None:
    os.environ["LITEPARSE_MODEL_DIR"] = str(MODEL_DIR.parent)

    # 1. Direct PNG test (receipt.png)
    receipt_png = DATA_DIR / "receipt.png"
    cpp_png = run_cpp(receipt_png, long_side=1600, threads=4, repeat=3)
    print(f"=== receipt.png ({cpp_png['width']}x{cpp_png['height']}) ===")
    print("C++ timings (warm, 4 threads):", cpp_png["timings_ms"])
    for line in cpp_png["lines"]:
        print(f"  C++ [{line['confidence']:.2f}] {line['text']}")

    # 2. Rendered 250-DPI PNG of receipt_scan.pdf + sample.pdf (page 1)
    parser = LiteParse(ocr_enabled=True, ocr_backend="ocria-nano", dpi=250, quiet=True)
    with tempfile.TemporaryDirectory() as tmp:
        for pdf_name in ["receipt_scan.pdf", "sample.pdf"]:
            pdf_path = DATA_DIR / pdf_name
            shot = parser.screenshot(str(pdf_path), page_numbers=[1])[0]
            img_path = Path(tmp) / f"{pdf_name}.png"
            img_path.write_bytes(shot.image_bytes)

            print(f"\n=== {pdf_name} (250 DPI -> {shot.width}x{shot.height}) ===")
            for ls in (1600, 1280, 960):
                cpp_res = run_cpp(img_path, long_side=ls, threads=4, repeat=3)
                t = cpp_res["timings_ms"]
                print(
                    f"  C++ long_side={ls:4d} | lines={len(cpp_res['lines']):2d} "
                    f"| det={t['det']:5.1f}ms crop={t['crop']:4.1f}ms rec={t['rec']:5.1f}ms total={t['total']:5.1f}ms"
                )

            if pdf_name == "receipt_scan.pdf":
                py_res = parser.parse(str(pdf_path))
                cpp_1600 = run_cpp(img_path, long_side=1600, threads=4, repeat=1)
                py_tokens = py_res.text.split()
                cpp_tokens = " ".join(l["text"] for l in cpp_1600["lines"]).split()
                assert py_tokens == cpp_tokens, (
                    f"Parity mismatch on {pdf_name}!\n  Python: {py_tokens}\n  C++:    {cpp_tokens}"
                )
                print(f"  [PASS] Exact token parity on {pdf_name} ({len(cpp_tokens)} tokens identical)")


if __name__ == "__main__":
    main()
