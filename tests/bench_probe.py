"""Quick latency & memory probe for LiteParse (useful baseline for TV/embedded targets)."""

from pathlib import Path
import resource
import time
from liteparse import LiteParse

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "integration_tests_data"
DEMO_DIR = REPO_ROOT / "demo" / "docs"


def rss_mb() -> float:
    # ru_maxrss is in KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(label: str, path: Path, **kwargs) -> None:
    parser = LiteParse(quiet=True, **kwargs)
    t0 = time.perf_counter()
    res = parser.parse(str(path))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    pages = len(res.pages)
    per_page = dt_ms / pages if pages else 0.0
    print(
        f"{label:35s} | pages={pages:3d}/{res.total_pages:<3d} "
        f"| total={dt_ms:7.1f} ms | per_page={per_page:6.2f} ms "
        f"| chars={len(res.text):7d} | peak_rss={rss_mb():6.1f} MB"
    )


def main() -> None:
    print(f"Initial peak RSS: {rss_mb():.1f} MB\n")
    bench("sample.pdf (text, no OCR)", DATA_DIR / "sample.pdf", ocr_enabled=False, output_format="text")
    bench("sample.pdf (markdown, no OCR)", DATA_DIR / "sample.pdf", ocr_enabled=False, output_format="markdown")
    bench("apple-10k p1-10 (text, no OCR)", DEMO_DIR / "apple-10k-2024.pdf", ocr_enabled=False, target_pages="1-10", output_format="text")
    bench("apple-10k p1-10 (md, no OCR)", DEMO_DIR / "apple-10k-2024.pdf", ocr_enabled=False, target_pages="1-10", output_format="markdown")
    bench("receipt_scan.pdf (tesseract OCR)", DATA_DIR / "receipt_scan.pdf", ocr_enabled=True, ocr_backend="tesseract", output_format="text")


if __name__ == "__main__":
    main()
