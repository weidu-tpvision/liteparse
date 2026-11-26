#!/usr/bin/env python3
"""Fetch header-only build dependencies for cpp/ into cpp/third_party/.

- Official ONNX Runtime 1.31.1 C/C++ headers (extracted via HTTP Range requests
  from Microsoft's NuGet package — ~1 MB downloaded, not the full 160 MB archive).
- stb_image.h (for the ocria_cli image loader).
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
import urllib.request
import zlib

NUGET_URL = (
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/"
    "1.31.1/microsoft.ml.onnxruntime.1.31.1.nupkg"
)
STB_URL = "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"

CPP_DIR = Path(__file__).resolve().parent.parent
THIRD_PARTY = CPP_DIR / "third_party"
ORT_INC = THIRD_PARTY / "onnxruntime"


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_ort_headers() -> None:
    ORT_INC.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(NUGET_URL, method="HEAD")
    with urllib.request.urlopen(req) as r:
        size = int(r.headers["Content-Length"])

    tail = fetch_range(NUGET_URL, max(0, size - 65536), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd == -1:
        raise RuntimeError("ZIP EOCD record not found in NuGet package")
    _, _, _, _, cd_size, cd_offset = struct.unpack_from("<HHHHII", tail, eocd + 4)
    cd = fetch_range(NUGET_URL, cd_offset, cd_offset + cd_size - 1)

    pos = 0
    count = 0
    while pos < len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            raise RuntimeError("Corrupt ZIP central directory")
        (
            comp_method,
            _,
            _,
            _,
            comp_size,
            _,
            name_len,
            extra_len,
            cmt_len,
            _,
            _,
            _,
            lfh_off,
        ) = struct.unpack_from("<HHHIIIHHHHHII", cd, pos + 10)
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8")
        pos += 46 + name_len + extra_len + cmt_len
        if name.startswith("build/native/include/") and name.endswith(".h"):
            blob = fetch_range(NUGET_URL, lfh_off, lfh_off + 30 + name_len + 1024 + comp_size)
            nlen, elen = struct.unpack_from("<HH", blob, 26)
            raw = blob[30 + nlen + elen : 30 + nlen + elen + comp_size]
            data = zlib.decompress(raw, -15) if comp_method == 8 else raw
            (ORT_INC / os.path.basename(name)).write_bytes(data)
            count += 1
    print(f"Extracted {count} ONNX Runtime 1.31.1 headers into {ORT_INC}")


def fetch_stb_image() -> None:
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    dst = THIRD_PARTY / "stb_image.h"
    with urllib.request.urlopen(STB_URL) as r:
        dst.write_bytes(r.read())
    print(f"Downloaded stb_image.h into {dst}")


if __name__ == "__main__":
    fetch_ort_headers()
    fetch_stb_image()
