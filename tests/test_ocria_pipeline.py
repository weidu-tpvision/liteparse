"""Reference Python implementations & tests for ocria-nano pre/postprocessing.

Mirrors crates/liteparse/src/ocr/ocria/{det,rec,page}.rs so you can test the
exact tensor math in Python or port it directly to a TV/embedded runtime.

Run unit tests (offline, no model download needed):
    tests/.venv/bin/pytest -v tests/test_ocria_pipeline.py

Run live ocria-nano OCR check (only if you already pulled `lit models pull ocria-nano`):
    tests/.venv/bin/python tests/test_ocria_pipeline.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest
from liteparse import LiteParse, models

# ── Constants from crates/liteparse/src/ocr/ocria/{det,rec}.rs ─────────────
DET_LONG_SIDE = 1600
DET_SHRINK = 0.3
REC_H = 32
REC_MIN_W = 8
REC_MAX_W = 1280
REC_BUCKET_W = 128
ROUTER_W = 512


def norm_u8(v: int) -> float:
    """Symmetric normalization used by det, router, and rec_* models."""
    return v / 127.5 - 1.0


def det_target_size(big_h: int, big_w: int, long_side: int = DET_LONG_SIDE) -> Tuple[int, int, float]:
    """Compute (h, w, scale) for det.onnx — multiples of 32, never upscaled."""
    s = min(1.0, long_side / max(big_h, big_w))
    h = max(32, round(big_h * s / 32.0) * 32)
    w = max(32, round(big_w * s / 32.0) * 32)
    return h, w, s


def unclip_kernel_size(kernel_w: float, kernel_h: float, shrink: float = DET_SHRINK) -> Tuple[float, float]:
    """Expand a min-area rect from shrunk text-line kernel back to full line box."""
    line_h = kernel_h / (1.0 - 2.0 * shrink)
    d = shrink * line_h
    return kernel_w + 2.0 * d, kernel_h + 2.0 * d


def rec_padded_width(widths: Sequence[int], bucket_w: int = REC_BUCKET_W) -> int:
    """Batch width padded up to the next multiple of 128 px."""
    max_w = max(widths) if widths else 1
    return math.ceil(max_w / bucket_w) * bucket_w


def router_tile_row(row_u8: Sequence[int], target_w: int = ROUTER_W) -> List[float]:
    """Tile a single grayscale scanline across ROUTER_W=512 and normalize."""
    w = len(row_u8)
    return [norm_u8(row_u8[xx % w]) for xx in range(target_w)]


def ctc_greedy_decode(
    ids: Sequence[int],
    confs: Sequence[float],
    charset: Sequence[str],
    line_w: int,
    stride: int,
) -> Tuple[str, float]:
    """Exact greedy CTC collapse from crates/liteparse/src/ocr/ocria/rec.rs."""
    t = min(line_w // stride, len(ids))
    chars: List[str] = []
    kept_confs: List[float] = []
    prev = 0
    for k in range(t):
        c = ids[k]
        if c != 0 and c != prev:
            if 0 <= c - 1 < len(charset):
                chars.append(charset[c - 1])
            kept_confs.append(confs[k])
        prev = c
    mean_conf = sum(kept_confs) / len(kept_confs) if kept_confs else 0.0
    return "".join(chars), mean_conf


# ── Unit Tests ─────────────────────────────────────────────────────────────

def test_normalization_bounds():
    assert norm_u8(0) == pytest.approx(-1.0)
    assert norm_u8(255) == pytest.approx(1.0)
    assert norm_u8(128) == pytest.approx(128 / 127.5 - 1.0)


def test_det_target_size_multiples_of_32_and_no_upscale():
    # Large 250-DPI A4 page (~2067 x 2923) scales down so long side <= 1600
    h, w, s = det_target_size(2923, 2067)
    assert h % 32 == 0 and w % 32 == 0
    assert max(h, w) == 1600
    assert s < 1.0

    # Small image (640 x 480) is never upscaled
    h2, w2, s2 = det_target_size(480, 640)
    assert s2 == 1.0
    assert (h2, w2) == (480, 640)


def test_unclip_kernel_size():
    # With SHRINK=0.3: line_h = kernel_h / 0.4, d = 0.3 * line_h = 0.75 * kernel_h
    w, h = unclip_kernel_size(kernel_w=100.0, kernel_h=12.0)
    assert h == pytest.approx(30.0)
    assert w == pytest.approx(118.0)


def test_rec_width_bucketing_and_router_tiling():
    assert rec_padded_width([10, 95, 128]) == 128
    assert rec_padded_width([129]) == 256
    tiled = router_tile_row([0, 255, 0, 255], target_w=8)
    assert len(tiled) == 8
    assert tiled[0] == pytest.approx(-1.0)
    assert tiled[1] == pytest.approx(1.0)
    assert tiled[4] == pytest.approx(-1.0)


def test_ctc_greedy_decode_respects_blank_duplicates_and_line_width():
    charset = ["H", "i", "!"]
    # Tokens: H, H (dup), blank, i, blank, ! (beyond active width t=4 -> ignored)
    ids = [1, 1, 0, 2, 0, 3]
    confs = [0.95, 0.90, 0.99, 0.85, 0.99, 0.99]
    text, mean_conf = ctc_greedy_decode(ids, confs, charset, line_w=16, stride=4)
    assert text == "Hi"
    assert mean_conf == pytest.approx((0.95 + 0.85) / 2.0)


def test_models_list_api_works():
    installed = models.list()
    assert isinstance(installed, list)


if __name__ == "__main__":
    installed = models.list()
    print("Installed models:", installed)
    has_nano = any(m.name == "ocria-nano" and m.complete for m in installed)
    if not has_nano:
        print("ocria-nano is not pulled yet. Run: tests/.venv/bin/lit models pull ocria-nano")
    else:
        sample = Path(__file__).resolve().parent.parent / "integration_tests_data" / "receipt_scan.pdf"
        parser = LiteParse(ocr_enabled=True, ocr_backend="ocria-nano", quiet=True)
        res = parser.parse(str(sample))
        print(f"ocria-nano output ({len(res.text)} chars):\n{res.text}")
