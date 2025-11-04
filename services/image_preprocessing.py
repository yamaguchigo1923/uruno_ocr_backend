from __future__ import annotations

import io
import os
from typing import Tuple
from datetime import datetime

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - dependency error
    raise ImportError("Pillow is required for image_preprocessing. Install with 'pip install pillow'") from exc

# Optional PDF backends: prefer PyMuPDF (fitz), fallback to pdf2image (requires poppler)
HAS_FITZ = False
HAS_PDF2IMAGE = False
try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except Exception:
    try:
        from pdf2image import convert_from_bytes

        HAS_PDF2IMAGE = True
    except Exception:
        HAS_PDF2IMAGE = False

def crop_image_bytes(
    image_bytes: bytes,
    top_pct: float = 0.0,
    bottom_pct: float = 0.0,
    left_pct: float = 0.0,
    right_pct: float = 0.0,
    out_format: str = "PNG",
) -> Tuple[bytes, Tuple[int, int], Tuple[int, int]]:
    """
    Crop an image provided as bytes by fractional percentages from each edge.

    Parameters:
      image_bytes: input image bytes (PNG/JPEG/...)
      top_pct/bottom_pct/left_pct/right_pct: fractions between 0.0 and 1.0
        indicating how much to cut from each side. e.g. top_pct=0.1 cuts
        the top 10% of the image height.
      out_format: output image format name (PNG by default)

    Returns: (out_bytes, (orig_w, orig_h), (new_w, new_h))
    """
    if not (0 <= top_pct < 1 and 0 <= bottom_pct < 1 and 0 <= left_pct < 1 and 0 <= right_pct < 1):
        raise ValueError("Percentages must be between 0.0 and 1.0")

    with Image.open(io.BytesIO(image_bytes)) as im:
        orig_w, orig_h = im.size
        left_px = int(round(orig_w * left_pct))
        top_px = int(round(orig_h * top_pct))
        right_px = int(round(orig_w * (1.0 - right_pct)))
        bottom_px = int(round(orig_h * (1.0 - bottom_pct)))

        # Clamp
        left_px = max(0, min(left_px, orig_w - 1))
        top_px = max(0, min(top_px, orig_h - 1))
        right_px = max(left_px + 1, min(right_px, orig_w))
        bottom_px = max(top_px + 1, min(bottom_px, orig_h))

        cropped = im.crop((left_px, top_px, right_px, bottom_px))
        out_buf = io.BytesIO()
        save_params = {}
        # For JPEG keep quality reasonable
        if out_format.upper() == "JPEG":
            save_params["quality"] = 90
        cropped.save(out_buf, format=out_format, **save_params)
        out_bytes = out_buf.getvalue()
        new_w, new_h = cropped.size
        return out_bytes, (orig_w, orig_h), (new_w, new_h)


def process_image_file(
    input_path: str,
    output_image_path: str,
    top_pct: float = 0.0,
    bottom_pct: float = 0.0,
    left_pct: float = 0.0,
    right_pct: float = 0.0,
    out_format: str = "PNG",
) -> dict:
    """
    Read image from input_path, crop according to percentages and write
    the processed image to output_image_path. Returns metadata dict.
    """
    with open(input_path, "rb") as f:
        img_bytes = f.read()
    out_bytes, orig_size, new_size = crop_image_bytes(
        img_bytes, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct, out_format=out_format
    )
    out_dir = os.path.dirname(output_image_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_image_path, "wb") as f:
        f.write(out_bytes)
    meta = {
        "input_path": input_path,
        "output_image_path": output_image_path,
        "orig_width": orig_size[0],
        "orig_height": orig_size[1],
        "new_width": new_size[0],
        "new_height": new_size[1],
        "top_pct": top_pct,
        "bottom_pct": bottom_pct,
        "left_pct": left_pct,
        "right_pct": right_pct,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return meta


def _pdf_bytes_to_images(pdf_bytes: bytes) -> list[bytes]:
    """
    Convert PDF bytes to a list of PNG bytes, one per page.
    Uses PyMuPDF if available, otherwise pdf2image if available.
    """
    images: list[bytes] = []
    if HAS_FITZ:
        # Use PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            images.append(pix.tobytes("png"))
        doc.close()
        return images
    if HAS_PDF2IMAGE:
        # pdf2image -> returns PIL Images
        pil_imgs = convert_from_bytes(pdf_bytes)
        for im in pil_imgs:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            images.append(buf.getvalue())
        return images
    raise RuntimeError(
        "No PDF backend available. Install PyMuPDF (pip install pymupdf) or pdf2image (pip install pdf2image) and poppler."
    )


def process_file(
    input_path: str,
    output_dir: str,
    top_pct: float = 0.0,
    bottom_pct: float = 0.0,
    left_pct: float = 0.0,
    right_pct: float = 0.0,
    out_format: str = "PNG",
) -> list[dict]:
    """
    Process an input file which may be an image or a PDF. For images, produces
    a single cropped image. For PDFs, processes each page and writes one
    cropped image per page into output_dir. Returns list of metadata dicts.
    """
    input_path = str(input_path)
    suffix = os.path.splitext(input_path)[1].lower()
    metas: list[dict] = []
    if suffix in {".pdf"}:
        with open(input_path, "rb") as f:
            pdf_bytes = f.read()
        pages = _pdf_bytes_to_images(pdf_bytes)
        for i, page_bytes in enumerate(pages, start=1):
            out_image_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_page{i}_cropped.png"
            out_image_path = os.path.join(output_dir, out_image_name)
            out_bytes, orig_size, new_size = crop_image_bytes(
                page_bytes, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct, out_format=out_format
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(out_image_path, "wb") as wf:
                wf.write(out_bytes)
            metas.append(
                {
                    "input_path": input_path,
                    "page": i,
                    "output_image_path": out_image_path,
                    "orig_width": orig_size[0],
                    "orig_height": orig_size[1],
                    "new_width": new_size[0],
                    "new_height": new_size[1],
                    "top_pct": top_pct,
                    "bottom_pct": bottom_pct,
                    "left_pct": left_pct,
                    "right_pct": right_pct,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
        return metas
    else:
        # treat as image
        stem = os.path.splitext(os.path.basename(input_path))[0]
        out_image_name = f"{stem}_cropped.png"
        out_image_path = os.path.join(output_dir, out_image_name)
        meta = process_image_file(
            input_path,
            out_image_path,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            left_pct=left_pct,
            right_pct=right_pct,
            out_format=out_format,
        )
        # ensure consistent shape as list of metas
        meta["page"] = 1
        metas.append(meta)
        return metas
