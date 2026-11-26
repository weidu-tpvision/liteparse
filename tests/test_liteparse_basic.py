"""Local smoke and feature tests for LiteParse using the repo's sample fixtures."""

from pathlib import Path
import pytest
from liteparse import LiteParse

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "integration_tests_data"
DEMO_DIR = REPO_ROOT / "demo" / "docs"


def test_parse_native_pdf_text():
    parser = LiteParse(ocr_enabled=False, quiet=True)
    result = parser.parse(str(DATA_DIR / "sample.pdf"))

    assert result.total_pages >= 1
    assert len(result.pages) == result.total_pages
    assert len(result.text.strip()) > 0
    assert all(len(p.text_items) > 0 for p in result.pages)


def test_parse_native_pdf_markdown_and_blocks():
    parser = LiteParse(
        ocr_enabled=False,
        output_format="markdown",
        extract_blocks=True,
        quiet=True,
    )
    result = parser.parse(str(DATA_DIR / "sample.pdf"))

    assert result.total_pages >= 1
    assert len(result.text.strip()) > 0
    assert result.pages[0].blocks is not None
    assert len(result.pages[0].blocks) > 0


def test_parse_from_bytes():
    pdf_bytes = (DATA_DIR / "sample.pdf").read_bytes()
    parser = LiteParse(ocr_enabled=False, quiet=True)
    result = parser.parse(pdf_bytes)

    assert result.total_pages >= 1
    assert len(result.text.strip()) > 0


def test_is_complex_native_vs_scan():
    parser = LiteParse(quiet=True)

    native_stats = parser.is_complex(str(DATA_DIR / "sample.pdf"))
    assert len(native_stats) >= 1
    assert not native_stats[0].needs_ocr

    scan_stats = parser.is_complex(str(DATA_DIR / "receipt_scan.pdf"))
    assert len(scan_stats) >= 1
    assert scan_stats[0].needs_ocr
    assert len(scan_stats[0].reasons) > 0


@pytest.mark.parametrize(
    "filename",
    ["sample_rotated_90cw.pdf", "sample_rotated_180.pdf"],
)
def test_parse_rotated_pdfs(filename: str):
    parser = LiteParse(ocr_enabled=False, quiet=True)
    result = parser.parse(str(DATA_DIR / filename))

    assert result.total_pages >= 1
    assert len(result.text.strip()) > 0


def test_extract_acroform_fields():
    parser = LiteParse(ocr_enabled=False, extract_form_fields=True, quiet=True)
    result = parser.parse(str(DATA_DIR / "filled_acroform.pdf"))

    assert result.total_pages >= 1
    fields = [f for p in result.pages for f in (p.form_fields or [])]
    assert len(fields) > 0


def test_screenshot_rendering():
    parser = LiteParse(dpi=72, quiet=True)
    shots = parser.screenshot(str(DATA_DIR / "sample.pdf"), page_numbers=[1])

    assert len(shots) == 1
    shot = shots[0]
    assert shot.page_num == 1
    assert shot.width > 0 and shot.height > 0
    assert shot.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_parse_target_pages_on_large_pdf():
    parser = LiteParse(
        ocr_enabled=False,
        target_pages="1-2",
        output_format="markdown",
        quiet=True,
    )
    result = parser.parse(str(DEMO_DIR / "apple-10k-2024.pdf"))

    assert result.total_pages > 2
    assert len(result.pages) == 2
    assert len(result.text.strip()) > 0


def test_ocr_on_scanned_pdf():
    parser = LiteParse(ocr_enabled=True, ocr_backend="tesseract", quiet=True)
    result = parser.parse(str(DATA_DIR / "receipt_scan.pdf"))

    assert result.total_pages == 1
    assert len(result.text.strip()) > 0


def test_worker_pool_mode():
    with LiteParse(
        ocr_enabled=False,
        pool_size=2,
        parse_timeout=10.0,
        quiet=True,
    ) as parser:
        parser.warm_up()
        r1 = parser.parse(str(DATA_DIR / "sample.pdf"))
        r2 = parser.parse(str(DATA_DIR / "filled_acroform.pdf"))
        assert len(r1.text.strip()) > 0
        assert len(r2.text.strip()) > 0
