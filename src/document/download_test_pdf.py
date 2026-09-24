"""Download a test legal PDF from the official India Code portal."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

PDF_URL = "https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf"
OUTPUT = Path("data/documents/Indian_Contract_Act_1872.pdf")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    request = Request(PDF_URL, headers={"User-Agent": "LawSuit-LLM-Educational/1.0"})
    with urlopen(request, timeout=60) as response:
        data = response.read()

    if not data.startswith(b"%PDF"):
        raise RuntimeError("Downloaded content is not a PDF.")

    OUTPUT.write_bytes(data)
    print("=" * 64)
    print("LawSuit LLM — Download Test Legal PDF")
    print("=" * 64)
    print(f"Source: {PDF_URL}")
    print(f"Output: {OUTPUT}")
    print(f"Size:  {len(data):,} bytes")
    print("Download completed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
