#!/usr/bin/env python3
"""Demo OCR + 3 chiến lược chunking cho Buổi 5

Sử dụng:
  python ocr_chunk_demo.py --dry-run
  python ocr_chunk_demo.py --write --use-llamaparse

Ghi chú:
- Theo SPEC, llama-cloud (LlamaParse) chỉ được dùng làm fallback OCR khi bật
  bằng `--use-llamaparse`; mặc định dry-run sẽ không gọi API.
- TUYỆT ĐỐI không in hoặc lưu giá trị API key. Key được đọc từ `src/.env` hoặc
  biến môi trường `LLAMA_CLOUD_API_KEY` nếu cần.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict

import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


BASE = Path(__file__).resolve().parents[1]
DEMO_DIR = BASE / "datademo"
SRC_DIR = BASE / "src"
OUTPUT_RAW = BASE / "output" / "raw"
OUTPUT_CHUNKS = BASE / "output" / "chunks"


@dataclass
class PageRecord:
    document_id: str
    page: int
    text: str
    metadata: Dict


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _parse_llamaparse_result(result: Any) -> Tuple[str, Dict[int, str]]:
    """Extract full text and per-page text from LlamaParse result."""
    if isinstance(result, dict):
        markdown_full = result.get("markdown_full")
        text_full = result.get("text_full")
        text_obj = result.get("text")
        markdown_obj = result.get("markdown")
    else:
        markdown_full = getattr(result, "markdown_full", None)
        text_full = getattr(result, "text_full", None)
        text_obj = getattr(result, "text", None)
        markdown_obj = getattr(result, "markdown", None)

    page_texts: Dict[int, str] = {}

    def _collect_pages(obj: Any, text_key: str):
        pages = None
        if obj is None:
            return
        if isinstance(obj, dict):
            pages = obj.get("pages")
        else:
            pages = getattr(obj, "pages", None)
        if not pages:
            return

        for page in pages:
            if isinstance(page, dict):
                page_number = page.get("page_number")
                page_text = page.get(text_key, "")
            else:
                page_number = getattr(page, "page_number", None)
                page_text = getattr(page, text_key, "")
            if page_number is None:
                continue
            try:
                page_index = int(page_number)
            except (TypeError, ValueError):
                continue
            page_texts[page_index] = page_text or ""

    _collect_pages(text_obj, "text")
    if not page_texts:
        _collect_pages(markdown_obj, "markdown")

    for p in page_texts:
        if isinstance(page_texts[p], str) and page_texts[p].strip() == "NO_CONTENT_HERE":
            page_texts[p] = ""

    text = None
    if markdown_full:
        text = markdown_full
    elif text_full:
        text = text_full
    elif page_texts:
        text = "\n\n".join(page_texts[p] for p in sorted(page_texts))
    elif isinstance(text_obj, str) and text_obj.strip():
        text = text_obj
    elif isinstance(markdown_obj, str) and markdown_obj.strip():
        text = markdown_obj

    if isinstance(text, str) and text.strip() == "NO_CONTENT_HERE":
        text = ""

    return normalize_nfc(text or ""), page_texts

def _extract_llamaparse_status(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    status = result.get("status")
    if status:
        return status
    job = result.get("job")
    if isinstance(job, dict):
        return job.get("status")
    return None


def _extract_llamaparse_error(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    error = result.get("error_message")
    if error:
        return error
    job = result.get("job")
    if isinstance(job, dict):
        return job.get("error_message")
    return None

def find_pdfs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() == ".pdf"]


def safe_extract_text_pymupdf(pdf_path: Path) -> Tuple[List[Optional[str]], List[str]]:
    """Return list of page texts (None when cannot extract) and list of warnings."""
    pages: List[Optional[str]] = []
    warnings: List[str] = []
    if fitz is None:
        warnings.append("PyMuPDF (fitz) not available; skipping text layer extraction")
        return [], warnings

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        warnings.append(f"Cannot open PDF with PyMuPDF: {e}")
        return [], warnings

    for i in range(len(doc)):
        text = ""
        try:
            page = doc.load_page(i)
            text = page.get_text("text") or ""
        except Exception as e:
            warnings.append(f"Error extracting text on page {i+1}: {e}")
            pages.append(None)
            continue

        # Heuristic: if text is too short or contains many non-printable chars, mark None
        if not text.strip():
            pages.append(None)
        else:
            pages.append(text)

    return pages, warnings


def render_page_to_image(pdf_path: Path, page_number: int) -> Optional[bytes]:
    if fitz is None:
        return None
    try:
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=200)
        return pix.tobytes("png")
    except Exception:
        return None


def get_llamacloud_api_key() -> Optional[str]:
    key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMA_PARSE_API_KEY")
    if key:
        return key
    dotenv_path = SRC_DIR / ".env"
    if dotenv_path.exists():
        try:
            from dotenv import dotenv_values

            env_values = dotenv_values(dotenv_path)
            return env_values.get("LLAMA_CLOUD_API_KEY") or env_values.get("LLAMA_PARSE_API_KEY")
        except Exception:
            pass
    return None


def _call_llamaparse_direct(file_bytes: bytes, filename: str, content_type: str, api_key: str) -> Dict[str, Any]:
    base_url = os.environ.get("LLAMA_CLOUD_BASE_URL") or "https://api.cloud.llamaindex.ai"
    upload_url = f"{base_url.rstrip('/')}/api/v2/parse/upload"
    headers = {"Authorization": f"Bearer {api_key}"}
    configuration = {
        "tier": "agentic",
        "version": "latest",
    }

    with httpx.Client(trust_env=False, timeout=httpx.Timeout(600.0)) as client:
        response = client.post(
            upload_url,
            headers=headers,
            files={"file": (filename, file_bytes, content_type)},
            data={"configuration": json.dumps(configuration)},
        )
        if response.status_code != 200:
            raise RuntimeError(f"LlamaParse direct upload failed: {response.status_code} {response.text}")
        upload_result = response.json()

        job_id = upload_result.get("id")
        if not job_id:
            raise RuntimeError("LlamaParse direct upload returned no job ID")

        get_url = f"{base_url.rstrip('/')}/api/v2/parse/{job_id}"
        timeout_secs = 600
        poll_interval = 1.0
        elapsed = 0.0

        while elapsed < timeout_secs:
            status_response = client.get(
                get_url,
                headers=headers,
                params={"expand": "text,markdown,text_full,markdown_full"},
            )
            if status_response.status_code != 200:
                raise RuntimeError(f"LlamaParse direct status check failed: {status_response.status_code} {status_response.text}")
            status_data = status_response.json()
            status = _extract_llamaparse_status(status_data)
            if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(poll_interval)
            elapsed += poll_interval

        if status != "COMPLETED":
            error_message = _extract_llamaparse_error(status_data) or status_data
            raise RuntimeError(f"LlamaParse direct parse did not complete: status={status}, error_message={error_message}")

    return status_data


def call_llamaparse(pdf_path: Path, page_number: Optional[int] = None) -> Dict[str, Any]:
    api_key = get_llamacloud_api_key()
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set in environment or src/.env")

    if page_number is None:
        file_bytes = pdf_path.read_bytes()
        filename = pdf_path.name
        content_type = "application/pdf"
    else:
        file_bytes = render_page_to_image(pdf_path, page_number)
        if file_bytes is None:
            raise RuntimeError(f"Unable to render page {page_number + 1} to image for LlamaParse fallback")
        filename = f"{pdf_path.stem}_page_{page_number + 1}.png"
        content_type = "image/png"

    result = _call_llamaparse_direct(file_bytes, filename, content_type, api_key)
    full_text, page_texts = _parse_llamaparse_result(result)
    if not full_text and not page_texts:
        raise RuntimeError("LlamaParse returned no usable OCR text")

    return {"full_text": full_text, "page_texts": page_texts}


def needs_ocr_text(text: Optional[str]) -> bool:
    if text is None:
        return True
    clean = text.strip()
    if not clean or clean == "NO_CONTENT_HERE":
        return True
    # heuristic: replacement character or high density of non-printables
    if "\ufffd" in text or sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r") > 5:
        return True
    # heuristic: garbled Vietnamese font encoding patterns from broken PDF font maps
    garbled_patterns = [
        r"\b(tâu|huyén|dAu|vn d|giãi quyt|thuo'ng|Thira cha\"|Phïi Iïic|thut ngü|Diêu|cüa|Vit Nam)\b",
        r"[a-zA-Z]+\?[a-zA-Z]+",
    ]
    for pat in garbled_patterns:
        if re.search(pat, text):
            return True
    # very short text
    if len(clean) < 20:
        return True
    return False


def save_json(path: Path, obj: Dict, write: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def chunk_fixed(text: str, source: str, page_start: int, page_end: int, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
    chunks = []
    i = 0
    chunk_id = 0
    L = len(text)
    while i < L:
        start = i
        end = min(i + chunk_size, L)
        chunk_text = text[start:end]
        chunk_id += 1
        chunks.append({
            "chunk_id": f"fixed-{chunk_id}",
            "strategy": "fixed",
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
        })
        i = end - overlap
        if i <= start:
            i = end
    return chunks


def chunk_semantic(text: str, source: str, page_start: int, page_end: int, max_chunk_size: int = 2000) -> List[Dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    chunk_id = 0
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chunk_size:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            chunk_id += 1
            chunks.append({
                "chunk_id": f"semantic-{chunk_id}",
                "strategy": "semantic",
                "source": source,
                "page_start": page_start,
                "page_end": page_end,
                "text": current,
            })
            current = p
    if current:
        chunk_id += 1
        chunks.append({
            "chunk_id": f"semantic-{chunk_id}",
            "strategy": "semantic",
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "text": current,
        })
    return chunks


HEADING_PATTERNS = [
    re.compile(r"^\s*Chương\b", re.I),
    re.compile(r"^\s*Mục\b", re.I),
    re.compile(r"^\s*Điều\b", re.I),
    re.compile(r"^\s*(EXW|FCA|CPT|CIP|DAP|DPU|DDP|FAS|FOB|CFR|CIF)\b", re.I),
    re.compile(r"^\s*(ĐIỀU KIỆN|CHÚ GIẢI HƯỚNG DẪN|BẢNG CÁC THUẬT NGỮ|LỜI NÓI ĐẦU)\b", re.I),
]


def detect_hierarchical_splits(text: str, source: str, page_start: int, page_end: int) -> Tuple[List[Dict], List[str]]:
    """Return list of hierarchical chunks and warnings."""
    warnings = []
    lines = text.splitlines()
    indices = []
    for idx, line in enumerate(lines):
        for pat in HEADING_PATTERNS:
            if pat.search(line):
                indices.append(idx)
                break

    if not indices:
        warnings.append("No hierarchical headings detected; fallback to semantic strategy.")
        semantic_chunks = chunk_semantic(text, source, page_start, page_end)
        fallback_chunks = []
        for idx, sc in enumerate(semantic_chunks, start=1):
            fc = dict(sc)
            fc["chunk_id"] = f"hier-fallback-{idx}"
            fc["strategy"] = "hierarchical"
            fc["metadata_structure"] = {"heading": "N/A", "fallback": "semantic"}
            fallback_chunks.append(fc)
        return fallback_chunks, warnings

    chunks = []
    indices = sorted(set(indices))
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(lines)
        chunk_text = "\n".join(lines[start:end]).strip()
        # basic metadata_structure: heading line
        heading_line = lines[start].strip() if start < len(lines) else ""
        chunks.append({
            "chunk_id": f"hier-{i+1}",
            "strategy": "hierarchical",
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
            "metadata_structure": {"heading": heading_line},
        })
    return chunks, warnings


def summarize_stats(chunks: List[Dict]) -> Dict:
    lengths = [len(c["text"]) for c in chunks] if chunks else [0]
    return {"count": len(chunks), "min": min(lengths), "max": max(lengths), "avg": sum(lengths) / max(1, len(lengths))}


def process_pdf(pdf_path: Path, write: bool, use_llamaparse: bool, doc_stem: Optional[str] = None) -> Dict:
    stem = doc_stem or pdf_path.stem
    result = {"document": pdf_path.name, "pages": [], "warnings": []}
    pages_texts, warnings = safe_extract_text_pymupdf(pdf_path)
    result["warnings"].extend(warnings)

    if pages_texts == [] and fitz is not None:
        result["warnings"].append("PyMuPDF returned no pages; file may be corrupted.")

    page_count = len(pages_texts)
    page_needs_ocr = [needs_ocr_text(t) for t in pages_texts]
    llamaparse_pages: Dict[int, str] = {}

    api_key = None
    if use_llamaparse:
        api_key = get_llamacloud_api_key()
        if not api_key:
            result["warnings"].append("LLAMA_CLOUD_API_KEY is not set; cannot use LlamaParse fallback")

    if use_llamaparse and api_key:
        if not pages_texts:
            try:
                whole_doc_result = call_llamaparse(pdf_path)
                llamaparse_pages = whole_doc_result.get("page_texts", {}) or {}
                page_count = max(page_count, max(llamaparse_pages.keys(), default=page_count))
            except Exception as e:
                result["warnings"].append(f"LlamaParse whole-document fallback failed: {e}")

        for page_index, need in enumerate(page_needs_ocr):
            if not need:
                continue
            if page_index + 1 in llamaparse_pages:
                continue
            try:
                page_result = call_llamaparse(pdf_path, page_number=page_index)
                page_text = page_result.get("full_text") if page_result.get("full_text") is not None else next(iter(page_result.get("page_texts", {}).values()), "")
                if page_text is not None:
                    llamaparse_pages[page_index + 1] = page_text
                else:
                    result["warnings"].append(f"Page {page_index + 1}: LlamaParse returned no OCR text")
            except Exception as e:
                result["warnings"].append(f"Page {page_index + 1}: LlamaParse error: {e}")

    if page_count == 0:
        page_count = 1
        page_needs_ocr = [True]

    if len(page_needs_ocr) < page_count:
        page_needs_ocr.extend([True] * (page_count - len(page_needs_ocr)))

    for i in range(page_count):
        raw_text = pages_texts[i] if i < len(pages_texts) else None
        need_ocr = page_needs_ocr[i] if i < len(page_needs_ocr) else True
        page_meta = {"source": str(pdf_path), "page": i + 1, "ocr_used": "pymupdf_text", "language": "vi"}
        if need_ocr:
            if use_llamaparse and api_key:
                page_text = llamaparse_pages.get(i + 1)
                if page_text is not None:
                    text = page_text
                    page_meta["ocr_used"] = "llamaparse_image"
                elif raw_text:
                    text = raw_text
                    page_meta["ocr_used"] = "pymupdf_text"
                    result["warnings"].append(f"Page {i+1}: LlamaParse fallback returned no page-level OCR text; preserving PyMuPDF text")
                else:
                    text = ""
                    page_meta["ocr_used"] = "pymupdf_text"
                    result["warnings"].append(f"Page {i+1}: no OCR text available after LlamaParse fallback; leaving text empty")
            else:
                text = raw_text or ""
                if not text and use_llamaparse:
                    result["warnings"].append(f"Page {i+1}: LlamaParse unavailable or API key missing; leaving text empty")
            result["warnings"].append(f"Page {i+1}: used {page_meta['ocr_used']}")
        else:
            text = raw_text or ""

        text = normalize_nfc(text)
        page_record = PageRecord(document_id=pdf_path.name, page=i + 1, text=text, metadata=page_meta)
        result["pages"].append(asdict(page_record))

        out_path = OUTPUT_RAW / stem / f"page_{i+1}.json"
        if write:
            save_json(out_path, asdict(page_record), write=True)

    return result


def run(args):
    if not DEMO_DIR.exists():
        logging.error("datademo folder not found: %s", DEMO_DIR)
        return 2

    pdfs = find_pdfs(DEMO_DIR)
    if not pdfs:
        logging.error("Không tìm thấy file PDF trong %s", DEMO_DIR)
        return 2

    overall = {"documents": []}
    for pdf in pdfs:
        use_llamaparse = args.use_llamaparse or (get_llamacloud_api_key() is not None)
        logging.info("Processing %s (dry-run=%s, use_llamaparse=%s)", pdf.name, args.dry_run, use_llamaparse)
        res = process_pdf(pdf, write=not args.dry_run and args.write, use_llamaparse=use_llamaparse)
        overall["documents"].append(res)

        # Combine all page texts for chunking demo
        full_text = "\n\n".join(p["text"] for p in res["pages"])

        # Fixed
        page_start = 1
        page_end = len(res["pages"]) if res["pages"] else 1
        fixed_chunks = chunk_fixed(full_text, pdf.name, page_start, page_end, chunk_size=args.fixed_size, overlap=args.overlap)
        semantic_chunks = chunk_semantic(full_text, pdf.name, page_start, page_end, max_chunk_size=args.semantic_max)
        hierarchical_chunks, h_warnings = detect_hierarchical_splits(full_text, pdf.name, page_start, page_end)

        if h_warnings:
            res.setdefault("warnings", []).extend(h_warnings)

        # Save chunk outputs (dry-run respects args.dry_run)
        if not args.dry_run and args.write:
            (OUTPUT_CHUNKS / "fixed").mkdir(parents=True, exist_ok=True)
            (OUTPUT_CHUNKS / "semantic").mkdir(parents=True, exist_ok=True)
            (OUTPUT_CHUNKS / "hierarchical").mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_CHUNKS / "fixed" / f"{pdf.stem}_fixed.json", "w", encoding="utf-8") as f:
                json.dump(fixed_chunks, f, ensure_ascii=False, indent=2)
            with open(OUTPUT_CHUNKS / "semantic" / f"{pdf.stem}_semantic.json", "w", encoding="utf-8") as f:
                json.dump(semantic_chunks, f, ensure_ascii=False, indent=2)
            with open(OUTPUT_CHUNKS / "hierarchical" / f"{pdf.stem}_hier.json", "w", encoding="utf-8") as f:
                json.dump(hierarchical_chunks, f, ensure_ascii=False, indent=2)

        stats = {
            "fixed": summarize_stats(fixed_chunks),
            "semantic": summarize_stats(semantic_chunks),
            "hierarchical": summarize_stats(hierarchical_chunks),
        }

        logging.info("Stats for %s: %s", pdf.name, stats)

    # summary write
    if not args.dry_run and args.write:
        save_json(OUTPUT_CHUNKS / "summary.json", overall, write=True)

    print("\nDONE. Dry-run=" + str(args.dry_run))
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=False, help="Chỉ preview; không ghi output")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Chạy thật và cho phép ghi output nếu kết hợp với --write")
    p.add_argument("--write", action="store_true", help="Cho phép ghi output (kết hợp với --no-dry-run)")
    p.add_argument("--use-llamaparse", action="store_true", help="Bật llama-cloud LlamaParse fallback (disabled by default)")
    p.add_argument("--fixed-size", type=int, default=2000)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--semantic-max", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args))
