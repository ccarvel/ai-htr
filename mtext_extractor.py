#!/usr/bin/env python
"""
OCR a local PDF *or* image with Mistral-AI's `mistral-ocr-latest`.

Examples
--------
# simple (auto-named output)
python mtext_extractor.py invoice.pdf

# image file, custom output name + extract pictures
python mtext_extractor.py slide.png -o slide_text.md --images pics/
"""
import argparse
import base64
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral
from tqdm import tqdm  # just for the spinner

# --------------------------------------------------------------------------- #
# 1 CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="OCR a PDF or image with Mistral OCR",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("file", type=Path,
                    help="PDF / JPEG / PNG / WEBP / GIF / TIFF image")
parser.add_argument("-o", "--out", type=Path,
                    help="Markdown output file (auto-named if omitted)")
parser.add_argument("--images", type=Path,
                    help="Folder to dump any extracted figures/photos")

args = parser.parse_args()

# --------------------------------------------------------------------------- #
# 2 Env + client
# --------------------------------------------------------------------------- #
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("Set MISTRAL_API_KEY in env or .env")

client = Mistral(api_key=api_key)

# --------------------------------------------------------------------------- #
# 3 Mime-type lookup & base-64 data URI
# --------------------------------------------------------------------------- #
MIME_MAP = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif":  "image/gif",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
}
suffix = args.file.suffix.lower()
if suffix not in MIME_MAP:
    raise ValueError(f"Unsupported extension '{suffix}'")

with open(args.file, "rb") as f:
    data_b64 = base64.b64encode(f.read()).decode()

data_uri = f"data:{MIME_MAP[suffix]};base64,{data_b64}"

is_pdf = suffix == ".pdf"
payload_key = "document_url" if is_pdf else "image_url"
payload = {"type": payload_key, payload_key: data_uri}

# --------------------------------------------------------------------------- #
# 4 Default output path → <stem>_YYYY-MM-DD-HHMMSS.md
# --------------------------------------------------------------------------- #
if args.out is None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    auto_name = f"{args.file.stem}_extracted_{stamp}.md"
    args.out = args.file.parent / auto_name
# if args.out is None:
#     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     auto_name = f"{args.file.stem}_{stamp}.txt"   # ← change .md → .txt
#     args.out = args.file.parent / auto_name
# --------------------------------------------------------------------------- #
# 5 OCR request
# --------------------------------------------------------------------------- #
print("Running OCR …")
resp = client.ocr.process(
    model="mistral-ocr-latest",
    document=payload,              # works for both PDFs & images
    include_image_base64=True
)
pages = resp.pages
print(f" recognised {len(pages)} page(s)")

# --------------------------------------------------------------------------- #
# 6 Write Markdown
# --------------------------------------------------------------------------- #
with open(args.out, "w", encoding="utf-8") as md:
    for idx, page in enumerate(pages):
    # for page in pages:
        md.write(f"\n\n<!----- Page {idx+1} ----->\n\n")
        md.write(page.markdown) # or page.text if you prefer
        # md.write("\n\n")                 # blank line between pages
print(f"→ Markdown saved to {args.out}")
# --------------------------------------------------------------------------- #
# 7 Optional image extraction
# --------------------------------------------------------------------------- #
if args.images:
    args.images.mkdir(parents=True, exist_ok=True)
    count = 0
    for p_idx, page in enumerate(pages):
        for i_idx, img in enumerate(page.images or []):
            header, b64data = img.data.split(",", 1)
            ext = header.split("/")[1].split(";")[0]
            out_path = args.images / f"page{p_idx+1:03d}_img{i_idx+1}.{ext}"
            with open(out_path, "wb") as out:
                out.write(base64.b64decode(b64data))
            count += 1
    print(f"→ {count} inline image(s) written to {args.images}")

print("✅ Done.")