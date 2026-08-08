import importlib.util
import json
import re
import statistics
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# UI strings (Vietnamese)
TITLE = "OCR chứng từ"
OWNER = "Trung tâm Tài trợ thương mại (TFC)"

# Paths
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]  # RAG folder
LOGO_PATH = BASE / "logo-agribank.png"
OUTPUT_DIR = HERE / "output" / "chunks"

# Colors
BGCOLOR = "#ffffff"
BORDER_COLOR = "#7B1113"  # boocđô / burgundy
ACCENT = "#7B1113"

st.set_page_config(page_title=TITLE, layout="wide")

# Simple CSS for color theme
st.markdown(
    f"""
<style>
body {{ background-color: {BGCOLOR}; }}
.header {{ background-color: {ACCENT}; color: white; padding: 12px 16px; border-radius:6px; }}
.logo {{ height:64px; }}
.card {{ border: 1px solid {BORDER_COLOR}; padding: 12px; border-radius: 8px; margin-bottom: 8px; }}
.chunk-item {{ padding: 8px; border-bottom:1px dashed #eee; }}
</style>
""",
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([0.15, 0.85])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    else:
        st.write("")
with col2:
    st.markdown(f"<div class=\"header\"><h2>{TITLE}</h2><div>{OWNER}</div></div>", unsafe_allow_html=True)
    st.markdown("---")

st.sidebar.title("Tùy chọn")
st.sidebar.markdown("Chọn file đã xử lý và chiến lược chunk để trực quan hoá.")

# Utility: scan output folder for json chunk files
def scan_output(folder: Path) -> Dict[str, Dict[str, Path]]:
    """Return mapping: {document_stem: {strategy: path_to_json}}

    Strategy normalized to 'fixed','semantic','hierarchical'.
    """
    docs: Dict[str, Dict[str, Path]] = {}
    if not folder.exists():
        return docs
    for p in folder.rglob("*.json"):
        name = p.name
        # try to infer strategy from parent folder or filename suffix
        strategy = None
        parent = p.parent.name.lower()
        if parent in ("fixed", "semantic", "hierarchical"):
            strategy = parent
        else:
            # filename patterns like foo_fixed.json
            lower = name.lower()
            if "_fixed" in lower:
                strategy = "fixed"
            elif "_semantic" in lower:
                strategy = "semantic"
            elif "_hier" in lower or "_hierarchical" in lower:
                strategy = "hierarchical"
            else:
                # fallback: all-other -> put under 'fixed'
                strategy = "fixed"

        # derive document stem by removing trailing strategy suffix if present
        stem = p.stem
        for suf in ("_fixed", "_semantic", "_hier", "_hierarchical"):
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
                break

        docs.setdefault(stem, {})[strategy] = p
    return docs


def load_chunks(path: Path) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # expected: list of chunk dicts
        if isinstance(data, list):
            return data
        # maybe a dict wrapping list
        if isinstance(data, dict):
            # common keys: 'chunks', 'items', or direct mapping
            for k in ("chunks", "items", "data"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            # otherwise try to find first list value
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []
    except Exception:
        return []


def compute_stats(chunks: List[Dict]) -> Dict[str, Optional[float]]:
    lengths = [len(c.get("text", "")) for c in chunks if isinstance(c.get("text", ""), str)]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0}
    return {"count": len(lengths), "min": min(lengths), "max": max(lengths), "avg": statistics.mean(lengths)}


def safe_import_ocr_module() -> Optional[Any]:
    module_path = HERE / "src" / "ocr_chunk_demo.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("buoi_05.ocr_chunk_demo", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_stem(name: str) -> str:
    stem = unicodedata.normalize("NFC", name)
    stem = re.sub(r"[^\w\-_. ]+", "_", stem)
    stem = stem.strip().replace(" ", "_")
    return stem[:80] or "uploaded_doc"


def chunk_text_document(module: Any, stem: str, text: str) -> Dict[str, Any]:
    text = text.strip()
    page_start = 1
    page_end = 1
    fixed_chunks = module.chunk_fixed(text, stem, page_start, page_end)
    semantic_chunks = module.chunk_semantic(text, stem, page_start, page_end)
    hierarchical_chunks, h_warnings = module.detect_hierarchical_splits(text, stem, page_start, page_end)
    output = {
        "stem": stem,
        "fixed": fixed_chunks,
        "semantic": semantic_chunks,
        "hierarchical": hierarchical_chunks,
        "warnings": h_warnings,
    }
    return output


def save_chunk_outputs(stem: str, chunks_by_strategy: Dict[str, List[Dict]]) -> None:
    for strategy, chunk_list in chunks_by_strategy.items():
        folder = OUTPUT_DIR / strategy
        folder.mkdir(parents=True, exist_ok=True)
        safe_write_json(folder / f"{stem}_{strategy}.json", chunk_list)


# file uploader and processing controls
ocr_module = safe_import_ocr_module()
if ocr_module is None:
    st.sidebar.error("Không thể nạp module xử lý OCR/chunk từ src/ocr_chunk_demo.py. Kiểm tra file và quyền truy cập.")

st.sidebar.markdown("---")
st.sidebar.header("Xử lý tài liệu mới")
uploaded_file = st.sidebar.file_uploader("Tải lên PDF hoặc TXT:", type=["pdf", "txt"])
pasted_text = st.sidebar.text_area("Hoặc dán văn bản vào đây:", height=180)
use_llamaparse = st.sidebar.checkbox("Cho phép LlamaParse fallback OCR", value=False)
should_process = st.sidebar.button("Chạy OCR/chunk")

last_status: Optional[str] = None
last_warnings: List[str] = []
last_created_stem: Optional[str] = None

if should_process:
    if not ocr_module:
        st.sidebar.error("Không thể xử lý tài liệu vì module OCR/chunk chưa nạp được.")
    else:
        if uploaded_file is None and not pasted_text.strip():
            st.sidebar.warning("Vui lòng tải lên file PDF/TXT hoặc dán văn bản trước khi xử lý.")
        else:
            stem = None
            if uploaded_file is not None and uploaded_file.name.lower().endswith(".pdf"):
                file_bytes = uploaded_file.read()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = Path(tmp.name)
                stem = normalize_stem(Path(uploaded_file.name).stem)
                result = ocr_module.process_pdf(tmp_path, write=True, use_llamaparse=use_llamaparse, doc_stem=stem)
                raw_text = "\n\n".join(page.get("text", "") for page in result.get("pages", []))
                chunks_output = chunk_text_document(ocr_module, stem, raw_text)
                save_chunk_outputs(stem, {"fixed": chunks_output["fixed"], "semantic": chunks_output["semantic"], "hierarchical": chunks_output["hierarchical"]})
                last_warnings = result.get("warnings", []) + chunks_output.get("warnings", [])
                last_status = f"Đã xử lý PDF và ghi output cho tài liệu '{stem}'."
            elif uploaded_file is not None and uploaded_file.name.lower().endswith(".txt"):
                text = uploaded_file.read().decode("utf-8", errors="replace")
                stem = normalize_stem(Path(uploaded_file.name).stem)
                chunks_output = chunk_text_document(ocr_module, stem, text)
                save_chunk_outputs(stem, {"fixed": chunks_output["fixed"], "semantic": chunks_output["semantic"], "hierarchical": chunks_output["hierarchical"]})
                last_warnings = chunks_output.get("warnings", [])
                last_status = f"Đã xử lý TXT và ghi output cho tài liệu '{stem}'."
            elif pasted_text.strip():
                stem = normalize_stem("pasted_text")
                chunks_output = chunk_text_document(ocr_module, stem, pasted_text)
                save_chunk_outputs(stem, {"fixed": chunks_output["fixed"], "semantic": chunks_output["semantic"], "hierarchical": chunks_output["hierarchical"]})
                last_warnings = chunks_output.get("warnings", [])
                last_status = "Đã xử lý văn bản dán và ghi output chunk."
            else:
                st.sidebar.warning("Định dạng file không được hỗ trợ. Vui lòng chọn PDF hoặc TXT.")

            if stem:
                st.sidebar.success(last_status)
                if last_warnings:
                    st.sidebar.warning("Warnings: " + "; ".join(last_warnings))
                st.session_state["scan_docs"] = scan_output(OUTPUT_DIR)
                last_created_stem = stem

# scan once (but add refresh option)
if "scan_docs" not in st.session_state:
    st.session_state["scan_docs"] = scan_output(OUTPUT_DIR)

if st.sidebar.button("Làm mới (Scan lại output)"):
    st.session_state["scan_docs"] = scan_output(OUTPUT_DIR)

docs = st.session_state["scan_docs"]

if not docs:
    st.info("Chưa có dữ liệu chunk sẵn có. Vui lòng tải lên tài liệu mới và chạy OCR/chunk.")
    if not should_process:
        st.stop()

# choose document
doc_choices = sorted(docs.keys())
selected_doc = st.sidebar.selectbox("Chọn tài liệu đã xử lý:", doc_choices)

# choose strategy
available_strats = list(docs.get(selected_doc, {}).keys())
strategy = st.sidebar.radio("Chiến lược chunk:", options=available_strats)

# load chunks
path_for_strategy = docs[selected_doc].get(strategy)
chunks = load_chunks(path_for_strategy) if path_for_strategy else []

# summary metrics
stats = compute_stats(chunks)

st.markdown(f"### Tài liệu: **{selected_doc}**")
st.markdown(f"**Chiến lược:** {strategy.capitalize()}")
col_metrics = st.columns(4)
col_metrics[0].metric("Tổng số chunk", stats["count"])
col_metrics[1].metric("Độ dài nhỏ nhất", stats["min"])
col_metrics[2].metric("Độ dài lớn nhất", stats["max"])
col_metrics[3].metric("Độ dài trung bình", f"{stats['avg']:.1f}")

st.markdown("---")

# list chunks and selection
if chunks:
    id_to_chunk = {c.get("chunk_id", f"chunk-{i}"): c for i, c in enumerate(chunks, start=1)}
    chunk_ids = list(id_to_chunk.keys())
    selected_chunk_id = st.selectbox("Chọn chunk để xem chi tiết:", chunk_ids)
    selected_chunk = id_to_chunk[selected_chunk_id]

    # show chunk details
    st.subheader("Chi tiết chunk")
    st.markdown("**Thông tin chính**")
    st.write({
        "chunk_id": selected_chunk.get("chunk_id"),
        "strategy": selected_chunk.get("strategy"),
        "source": selected_chunk.get("source"),
        "page_start": selected_chunk.get("page_start"),
        "page_end": selected_chunk.get("page_end"),
    })

    if selected_chunk.get("metadata_structure"):
        st.markdown("**Metadata cấu trúc**")
        st.json(selected_chunk.get("metadata_structure"))

    st.markdown("**Nội dung text**")
    st.text_area("", value=selected_chunk.get("text", ""), height=300)

    st.markdown("---")

    # show list of all chunks in a compact way
    st.subheader("Danh sách chunk (tóm tắt)")
    for c in chunks:
        cid = c.get("chunk_id")
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.markdown(f"**{cid}**")
            st.write(f"[{c.get('page_start')}→{c.get('page_end')}]")
        with col2:
            preview = c.get("text", "")[:300].replace("\n", " ")
            st.write(preview + ("..." if len(c.get("text", "")) > 300 else ""))
else:
    st.info("Tài liệu đã chọn không có chunk ở chiến lược này hoặc file không đọc được.")

st.markdown("---")
st.caption("Lưu ý: App chỉ đọc output đã có. Không gọi LLM, không thực hiện OCR, không tạo embedding hay vector DB.")
