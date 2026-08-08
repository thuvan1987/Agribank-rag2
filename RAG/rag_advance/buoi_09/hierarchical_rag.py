"""
Buổi 09: Module Multi-Query & Parent-Child Hierarchical RAG (Hierarchy Registry & Parent Store Builder).

Module này chịu trách nhiệm:
1. Validate cấu hình tham số cho Multi-Query và Parent-Child RAG.
2. Xây dựng Hierarchy Registry ánh xạ Child-to-Parent từ dữ liệu Chunks pháp lý.
3. ChiaParent Document theo ranh giới Child Chunk không vượt PARENT_MAX_CHARS.
4. Ghi Store atomic vào storage/hierarchy/ và cung cấp lệnh CLI/Audit.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag

# Load .env từ vị trí tuyệt đối của BASE_DIR
load_dotenv(dotenv_path=ENV_PATH)


def get_hierarchical_config(custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Đọc, ép kiểu và xác thực toàn bộ thông số cấu hình cho Buổi 09.
    Hỗ trợ override trực tiếp qua dict custom_config phục vụ testing.
    """
    default_cfg = {
        "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip(),
        "embedding_dim": int(os.getenv("GEMINI_EMBEDDING_DIM", "768").strip()),
        "generation_model": os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite").strip(),
        "rag_max_distance": float(os.getenv("RAG_MAX_DISTANCE", "0.45").strip()),
        "bm25_candidates": int(os.getenv("BM25_CANDIDATES", "20").strip()),
        "semantic_candidates": int(os.getenv("SEMANTIC_CANDIDATES", "20").strip()),
        "rrf_k": int(os.getenv("RRF_K", "60").strip()),
        "rrf_bm25_weight": float(os.getenv("RRF_BM25_WEIGHT", "1.0").strip()),
        "rrf_semantic_weight": float(os.getenv("RRF_SEMANTIC_WEIGHT", "1.0").strip()),
        "rerank_candidates": int(os.getenv("RERANK_CANDIDATES", "20").strip()),
        "final_top_k": int(os.getenv("FINAL_TOP_K", "5").strip()),
        "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
        "reranker_max_length": int(os.getenv("RERANKER_MAX_LENGTH", "512").strip()),
        "rerank_batch_size": int(os.getenv("RERANK_BATCH_SIZE", "4").strip()),
        "rerank_min_score": float(os.getenv("RERANK_MIN_SCORE", "0.50").strip()),
        "rerank_device": os.getenv("RERANK_DEVICE", "auto").strip().lower(),
        # Cấu hình mới Buổi 09
        "multi_query_count": int(os.getenv("MULTI_QUERY_COUNT", "3").strip()),
        "multi_query_max_chars": int(os.getenv("MULTI_QUERY_MAX_CHARS", "300").strip()),
        "multi_query_temperature": float(os.getenv("MULTI_QUERY_TEMPERATURE", "0.2").strip()),
        "multi_query_original_weight": float(os.getenv("MULTI_QUERY_ORIGINAL_WEIGHT", "1.5").strip()),
        "multi_query_variant_weight": float(os.getenv("MULTI_QUERY_VARIANT_WEIGHT", "1.0").strip()),
        "multi_query_rrf_k": int(os.getenv("MULTI_QUERY_RRF_K", "60").strip()),
        "per_query_candidates": int(os.getenv("PER_QUERY_CANDIDATES", "12").strip()),
        "parent_max_chars": int(os.getenv("PARENT_MAX_CHARS", "6000").strip()),
        "parent_score_child_limit": int(os.getenv("PARENT_SCORE_CHILD_LIMIT", "3").strip()),
        "parent_rrf_k": int(os.getenv("PARENT_RRF_K", "60").strip()),
        "parent_candidates": int(os.getenv("PARENT_CANDIDATES", "10").strip()),
        "final_parent_top_k": int(os.getenv("FINAL_PARENT_TOP_K", "3").strip()),
        "total_context_max_chars": int(os.getenv("TOTAL_CONTEXT_MAX_CHARS", "16000").strip()),
    }

    cfg = dict(default_cfg)
    if custom_config is not None:
        cfg.update(custom_config)

    # Model names validation
    for m_key in ["embedding_model", "generation_model", "reranker_model"]:
        if not cfg.get(m_key):
            raise ValueError(f"Cấu hình '{m_key}' không được để rỗng.")

    # Multi-query validation
    mq_count = cfg.get("multi_query_count", 0)
    if not (1 <= mq_count <= 5):
        raise ValueError(f"MULTI_QUERY_COUNT ({mq_count}) phải là số nguyên từ 1 đến 5.")

    mq_chars = cfg.get("multi_query_max_chars", 0)
    if not (50 <= mq_chars <= 1000):
        raise ValueError(f"MULTI_QUERY_MAX_CHARS ({mq_chars}) phải nằm trong khoảng [50, 1000].")

    mq_temp = cfg.get("multi_query_temperature", -1.0)
    if not (0.0 <= mq_temp <= 1.0):
        raise ValueError(f"MULTI_QUERY_TEMPERATURE ({mq_temp}) phải nằm trong khoảng [0.0, 1.0].")

    w_orig = cfg.get("multi_query_original_weight", 0.0)
    w_var = cfg.get("multi_query_variant_weight", 0.0)
    if w_orig < 0.0 or w_var < 0.0:
        raise ValueError("Trọng số Multi-query (MULTI_QUERY_ORIGINAL_WEIGHT, MULTI_QUERY_VARIANT_WEIGHT) phải không âm (>= 0).")
    if w_orig == 0.0 and w_var == 0.0:
        raise ValueError("Trọng số Multi-query không được đồng thời bằng 0.0.")

    if cfg.get("multi_query_rrf_k", 0) <= 0:
        raise ValueError(f"MULTI_QUERY_RRF_K ({cfg.get('multi_query_rrf_k')}) phải là số nguyên dương > 0.")

    # Parent validation
    p_max_chars = cfg.get("parent_max_chars", 0)
    if not (1000 <= p_max_chars <= 20000):
        raise ValueError(f"PARENT_MAX_CHARS ({p_max_chars}) phải nằm trong khoảng [1000, 20000].")

    p_child_limit = cfg.get("parent_score_child_limit", 0)
    if not (1 <= p_child_limit <= 20):
        raise ValueError(f"PARENT_SCORE_CHILD_LIMIT ({p_child_limit}) phải nằm trong khoảng [1, 20].")

    p_cands = cfg.get("parent_candidates", 0)
    if not (1 <= p_cands <= 100):
        raise ValueError(f"PARENT_CANDIDATES ({p_cands}) phải là số nguyên dương từ 1 đến 100.")

    final_p_k = cfg.get("final_parent_top_k", 0)
    if not (1 <= final_p_k <= 100):
        raise ValueError(f"FINAL_PARENT_TOP_K ({final_p_k}) phải nằm trong khoảng [1, 100].")

    if final_p_k > p_cands:
        raise ValueError(f"FINAL_PARENT_TOP_K ({final_p_k}) không được lớn hơn PARENT_CANDIDATES ({p_cands}).")

    tot_ctx = cfg.get("total_context_max_chars", 0)
    if tot_ctx < p_max_chars:
        raise ValueError(f"TOTAL_CONTEXT_MAX_CHARS ({tot_ctx}) không được nhỏ hơn PARENT_MAX_CHARS ({p_max_chars}).")

    cfg["has_api_key"] = bool(cfg.get("api_key"))
    return cfg


def extract_trailing_number(chunk_id: str) -> int:
    """Tách phần số tự nhiên cuối cùng của chunk_id để sắp xếp thứ tự số học (ví dụ 'hier-2' -> 2, 'hier-10' -> 10)."""
    match = re.search(r"(\d+)$", chunk_id.strip())
    if match:
        return int(match.group(1))
    return float("inf")


def parse_legal_structure(record: Dict[str, Any]) -> Tuple[Dict[str, Optional[str]], str, bool, List[str]]:
    """
    Phân giải cấu trúc pháp lý cho một child chunk dựa trên thứ tự ưu tiên:
    1. Metadata structure hợp lệ của record.
    2. Heading cấp cao ở đầu chunk text.
    3. Carry forward từ chunk trước đó trong cùng source (thực hiện ở vòng lặp ngoài).
    4. Document fallback.
    """
    warnings: List[str] = []
    ambiguous = False
    resolution_method = "document_fallback"

    struct_path = {
        "chapter": None,
        "article": None,
        "clause": None,
        "point": None,
    }

    # 1. Metadata Precedence
    raw_st = record.get("metadata_structure", record.get("structure", {}))
    if isinstance(raw_st, str):
        try:
            raw_st = json.loads(raw_st)
        except Exception:
            raw_st = {}
    elif not isinstance(raw_st, dict):
        raw_st = {}

    has_meta_article = any(k in raw_st for k in ["article", "article_number", "dieu"])
    has_meta_chapter = any(k in raw_st for k in ["chapter", "chapter_number", "chuong"])

    if has_meta_article or has_meta_chapter:
        resolution_method = "metadata"
        if has_meta_chapter:
            struct_path["chapter"] = str(raw_st.get("chapter", raw_st.get("chapter_number", raw_st.get("chuong", "")))).strip() or None
        if has_meta_article:
            struct_path["article"] = str(raw_st.get("article", raw_st.get("article_number", raw_st.get("dieu", "")))).strip() or None
        if "clause" in raw_st or "clause_number" in raw_st:
            struct_path["clause"] = str(raw_st.get("clause", raw_st.get("clause_number", ""))).strip() or None
        if "point" in raw_st or "point_number" in raw_st:
            struct_path["point"] = str(raw_st.get("point", raw_st.get("point_number", ""))).strip() or None

        return struct_path, resolution_method, ambiguous, warnings

    # 2. Heading Inferred ở đầu chunk text
    text = str(record.get("text", "")).strip()
    first_line = text.split("\n")[0].strip() if text else ""

    ch_match = re.match(r"^(?:#\s*)?(Chương\s+[0-9IVXLCDM\d]+(?:\:|\.|\s+|$).*)", first_line, re.IGNORECASE)
    art_match = re.match(r"^(?:#\s*)?(Điều\s+\d+(?:\.\d+)*(?:\:|\.|\s+|$).*)", first_line, re.IGNORECASE)

    if art_match or ch_match:
        resolution_method = "heading_inferred"
        if ch_match:
            struct_path["chapter"] = ch_match.group(1).strip()
        if art_match:
            struct_path["article"] = art_match.group(1).strip()
        return struct_path, resolution_method, ambiguous, warnings

    # Cụm Điều giữa câu (inline references) hoặc heading không khớp -> Đánh dấu nếu có xung đột
    if re.search(r"Điều\s+\d+", text, re.IGNORECASE) and not art_match:
        # Inline reference
        pass

    return struct_path, resolution_method, ambiguous, warnings


def build_hierarchy_registry(
    input_path: Optional[Union[str, Path]] = None,
    strategy: str = "hierarchical",
    custom_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Xây dựng Hierarchy Registry và Parent Store từ danh sách Child Chunks.
    - Nhóm theo source và sắp xếp child theo thứ tự số của chunk_id.
    - Ánh xạ Child-to-Parent theo 4 cấp độ ưu tiên (metadata -> heading_inferred -> carried_forward -> document_fallback).
    - Gom nhóm Parent Document theo ranh giới Child không vượt quá PARENT_MAX_CHARS.
    """
    cfg = get_hierarchical_config(custom_config)
    norm_strat = rag.normalize_strategy(strategy)

    if norm_strat != "hierarchical":
        raise ValueError(f"Buổi 09 chỉ chấp nhận strategy 'hierarchical', nhận giá trị '{strategy}'.")

    if input_path is None:
        input_path = rag.DEFAULT_CHUNKS_DIR

    chunks, stats = rag.load_chunks(input_path=input_path, strategy=norm_strat)

    # 1. Nhóm theo source
    chunks_by_source: Dict[str, List[Dict[str, Any]]] = {}
    seen_ids = set()

    for c in chunks:
        cid = str(c["chunk_id"]).strip()
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk_id '{cid}' phát hiện khi xây dựng hierarchy.")
        seen_ids.add(cid)

        src = str(c["source"]).strip()
        chunks_by_source.setdefault(src, []).append(c)

    processed_children: List[Dict[str, Any]] = []
    parent_groups: Dict[str, List[Dict[str, Any]]] = {}

    total_ambiguous_children = 0
    total_oversized_children = 0

    # 2. Xử lý phân giải thứ tự và hierarchy resolution theo từng source
    for src in sorted(list(chunks_by_source.keys())):
        src_chunks = chunks_by_source[src]
        # Sắp xếp child theo số sequence của chunk_id
        src_chunks.sort(key=lambda item: (extract_trailing_number(str(item["chunk_id"])), str(item["chunk_id"])))

        last_chapter: Optional[str] = None
        last_article: Optional[str] = None

        for child_item in src_chunks:
            cid = str(child_item["chunk_id"]).strip()
            text = str(child_item["text"]).strip()
            p_start = int(child_item["page_start"])
            p_end = int(child_item["page_end"])

            struct_path, method, ambiguous, warnings = parse_legal_structure(child_item)

            # Check priority 3: Carry forward
            if method == "document_fallback":
                if last_article or last_chapter:
                    method = "carried_forward"
                    struct_path["chapter"] = last_chapter
                    struct_path["article"] = last_article
            else:
                # Update carry forward state
                if struct_path["chapter"]:
                    last_chapter = struct_path["chapter"]
                if struct_path["article"]:
                    last_article = struct_path["article"]

            # Conflict check: nếu text chứa nhiều cụm Điều khác biệt so với article_key
            inline_articles = re.findall(r"Điều\s+\d+(?:\.\d+)*", text)
            if len(set(inline_articles)) > 1 and method != "metadata":
                ambiguous = True
                warnings.append(f"Văn bản chứa nhiều viện dẫn Điều luật xung đột ({', '.join(set(inline_articles))}).")

            if ambiguous:
                total_ambiguous_children += 1

            # Xác định Article Key để gom nhóm Parent
            if struct_path["article"]:
                art_key = f"article:{struct_path['article']}"
            elif struct_path["chapter"]:
                art_key = f"chapter:{struct_path['chapter']}"
            else:
                art_key = f"doc_fallback:{src}"

            group_key = f"{src}::{art_key}"

            child_record = {
                "child_id": cid,
                "parent_id": None, # Sẽ gán sau khi chia parent window
                "source": src,
                "page_start": p_start,
                "page_end": p_end,
                "text": text,
                "structural_path": struct_path,
                "resolution_method": method,
                "ambiguous": ambiguous,
                "warnings": warnings,
                "_group_key": group_key,
                "_article_key": art_key
            }
            processed_children.append(child_record)
            parent_groups.setdefault(group_key, []).append(child_record)

    # 3. Gom nhóm và chia Parent Windows theo PARENT_MAX_CHARS
    parent_documents: List[Dict[str, Any]] = []
    p_max_chars = cfg["parent_max_chars"]

    for group_key in sorted(list(parent_groups.keys())):
        group_children = parent_groups[group_key]
        src = group_children[0]["source"]
        art_key = group_children[0]["_article_key"]

        windows: List[List[Dict[str, Any]]] = []
        curr_win: List[Dict[str, Any]] = []
        curr_len = 0

        for c_rec in group_children:
            c_len = len(c_rec["text"])
            if c_len > p_max_chars:
                total_oversized_children += 1
                c_rec["warnings"].append(f"oversized_single_child: Chiều dài child ({c_len}) vượt quá PARENT_MAX_CHARS ({p_max_chars}).")

            if curr_win and (curr_len + c_len > p_max_chars):
                windows.append(curr_win)
                curr_win = [c_rec]
                curr_len = c_len
            else:
                curr_win.append(c_rec)
                curr_len += c_len

        if curr_win:
            windows.append(curr_win)

        # Xây dựng Parent Object cho từng window
        for win_idx, win_children in enumerate(windows, start=1):
            stable_seed = f"{src}::{art_key}::win_{win_idx}"
            parent_id = f"parent_{hashlib.md5(stable_seed.encode('utf-8')).hexdigest()[:12]}"

            win_cids = [c["child_id"] for c in win_children]
            win_text = "\n\n".join([c["text"] for c in win_children])
            min_p = min(c["page_start"] for c in win_children)
            max_p = max(c["page_end"] for c in win_children)
            amb_cnt = sum(1 for c in win_children if c["ambiguous"])

            parent_warns = []
            if any("oversized_single_child" in w for c in win_children for w in c["warnings"]):
                parent_warns.append("oversized_single_child")
            if amb_cnt > 0:
                parent_warns.append(f"Chứa {amb_cnt} child chunks có thứ hạng/cấu trúc không chắc chắn (ambiguous).")

            # Gán parent_id ngược lại cho từng child
            for c in win_children:
                c["parent_id"] = parent_id

            # Xác định Heading cho Parent
            head_val = win_children[0]["structural_path"]["article"] or win_children[0]["structural_path"]["chapter"] or f"Tài liệu {src}"

            parent_doc = {
                "parent_id": parent_id,
                "source": src,
                "page_start": min_p,
                "page_end": max_p,
                "article_key": art_key,
                "heading": head_val,
                "window_index": win_idx,
                "child_ids": win_cids,
                "text": win_text,
                "char_count": len(win_text),
                "ambiguous_child_count": amb_cnt,
                "warnings": parent_warns
            }
            parent_documents.append(parent_doc)

    # Dọn dẹp trường tạm trong child records
    for c in processed_children:
        del c["_group_key"]
        del c["_article_key"]

    summary_stats = {
        "total_sources": len(chunks_by_source),
        "total_children": len(processed_children),
        "total_parents": len(parent_documents),
        "ambiguous_children_count": total_ambiguous_children,
        "oversized_children_count": total_oversized_children,
    }

    return processed_children, parent_documents, summary_stats


def save_hierarchy_store(
    children: List[Dict[str, Any]],
    parents: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    target_dir: Optional[Union[str, Path]] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ghi Store chủ động vào storage/hierarchy/ với cơ chế ghi Atomic qua file tạm.
    """
    cfg = get_hierarchical_config(custom_config)
    if target_dir is None:
        store_dir = BASE_DIR / "storage" / "hierarchy"
    else:
        store_dir = Path(target_dir)

    store_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "buoi_09_v1",
        "build_timestamp": datetime.datetime.now().isoformat(),
        "strategy": "hierarchical",
        "config_identity": {
            "parent_max_chars": cfg["parent_max_chars"],
            "parent_score_child_limit": cfg["parent_score_child_limit"],
            "parent_rrf_k": cfg["parent_rrf_k"]
        },
        "counts": summary_stats,
        "warning_counts": {
            "ambiguous_children": summary_stats["ambiguous_children_count"],
            "oversized_children": summary_stats["oversized_children_count"],
        }
    }

    # Ghi atomic qua file tạm cùng thư mục
    files_map = {
        "children.json": children,
        "parents.json": parents,
        "manifest.json": manifest
    }

    for fname, data_obj in files_map.items():
        final_path = store_dir / fname
        tmp_path = store_dir / f"{fname}.tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=2)

        tmp_path.replace(final_path)

    return manifest


def hierarchy_status(target_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Kiểm tra trạng thái Hierarchy Store (READ-ONLY: Không mkdir, không ghi file, không sửa timestamp).
    """
    if target_dir is None:
        store_dir = BASE_DIR / "storage" / "hierarchy"
    else:
        store_dir = Path(target_dir)

    children_path = store_dir / "children.json"
    parents_path = store_dir / "parents.json"
    manifest_path = store_dir / "manifest.json"

    store_exists = children_path.exists() and parents_path.exists() and manifest_path.exists()
    manifest_data = {}

    if store_exists:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            store_exists = False

    return {
        "store_exists": store_exists,
        "store_dir": str(store_dir),
        "manifest": manifest_data
    }


_QUERY_VARIANTS_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_query_variants_cache():
    """Xóa sạch bộ nhớ tạm Cache Multi-Query trong process."""
    global _QUERY_VARIANTS_CACHE
    _QUERY_VARIANTS_CACHE.clear()


def extract_legal_references(text: str) -> set:
    """Trích xuất danh sách các tham chiếu Điều, Khoản, Điểm, Văn bản từ văn bản."""
    patterns = [
        r"Điều\s+\d+(?:\.\d+)*",
        r"Khoản\s+\d+",
        r"Điểm\s+[a-zđ]",
        r"(?:Thông tư|Nghị định)\s+\d+/\d+/[A-Z0-9\-\.]+",
        r"Luật\s+[\w\s]+",
    ]
    refs = set()
    for p in patterns:
        for m in re.findall(p, text, re.IGNORECASE):
            refs.add(m.strip())
    return refs


def normalize_dedup_key(text: str) -> str:
    """Chuẩn hóa chuỗi ký tự phục vụ deduplicate (NFC + lower + collapse whitespace/punctuation)."""
    import unicodedata
    t = unicodedata.normalize("NFC", text).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def generate_query_variants(
    question: str,
    custom_config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Sinh tập câu hỏi tìm kiếm Multi-Query cho câu hỏi pháp lý tiếng Việt.
    - Luôn giữ Q0 (nguyên văn câu hỏi gốc) ở vị trí đầu tiên (Q0, origin='original').
    - Sinh tối đa MULTI_QUERY_COUNT câu hỏi biến thể tiếp cận đa chiều (Q1..Qn, origin='generated').
    - Thực hiện deduplicate, kiểm tra bảo tồn tham chiếu Điều/Khoản và cache trong process.
    - Hỗ trợ dependency injection `query_generator_fn` cho 100% offline testing.
    """
    import unicodedata

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi không được để rỗng.")

    norm_q0 = unicodedata.normalize("NFC", question).strip()
    cfg = get_hierarchical_config(custom_config)

    model_name = cfg["generation_model"]
    mq_count = cfg["multi_query_count"]
    mq_max_chars = cfg["multi_query_max_chars"]
    mq_temp = cfg["multi_query_temperature"]

    # Cache key theo question hash + config + model
    cache_key = hashlib.sha256(f"{norm_q0}::{model_name}::{mq_count}::{mq_temp}".encode("utf-8")).hexdigest()

    if cache_key in _QUERY_VARIANTS_CACHE:
        cached_res = dict(_QUERY_VARIANTS_CACHE[cache_key])
        cached_res["queries"] = [dict(q) for q in cached_res["queries"]]
        cached_res["cache_hit"] = True
        return cached_res

    start_time = time.time()
    status = "ready"
    warnings: List[str] = []
    raw_json_str: Optional[str] = None

    if query_generator_fn is not None:
        try:
            raw_json_str = query_generator_fn(norm_q0, cfg)
        except Exception as e:
            status = "query_generation_unavailable"
            warnings.append(f"Generator ngắt/lỗi: {e}")
    else:
        if not cfg.get("api_key"):
            status = "query_generation_unavailable"
            warnings.append("Thiếu GEMINI_API_KEY để gọi Multi-query expansion.")
        else:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=cfg["api_key"])
                prompt = f"""Bạn là trợ lý AI chuyên gia phân tích tìm kiếm văn bản pháp luật ngân hàng.
Nhiệm vụ: Hãy tạo ra đúng {mq_count} cách diễn đạt/câu hỏi tra cứu khác nhau cho câu hỏi pháp lý sau.

Câu hỏi gốc: "{norm_q0}"

Quy tắc bắt buộc:
1. Chỉ sinh ra đúng {mq_count} câu hỏi biến thể trong JSON format.
2. Mỗi biến thể gồm "text" (câu hỏi) và "focus" ("exact_legal_terms", "paraphrase", hoặc "missing_aspect").
3. GIỮ NGUYÊN các số Điều, Khoản, Điểm hoặc tên văn bản nếu có trong câu hỏi gốc. Không được tự bịa ra số Điều/Khoản mới không có trong câu hỏi gốc.
4. KHÔNG trả lời câu hỏi, KHÔNG đưa kết luận hay trích dẫn ngoài câu hỏi.

JSON Schema trả về:
{{"queries": [{{"text": "...", "focus": "..."}}]}}
"""
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=mq_temp,
                        response_mime_type="application/json"
                    )
                )
                raw_json_str = resp.text
            except Exception as e:
                status = "query_generation_unavailable"
                warnings.append(f"Lỗi gọi Gemini API sinh Multi-query: {e}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    q0_record = {
        "query_id": "Q0",
        "text": norm_q0,
        "origin": "original",
        "focus": "original_intent"
    }
    queries_list = [q0_record]
    dropped_duplicates = 0

    if status == "ready" and raw_json_str:
        try:
            parsed = json.loads(raw_json_str) if isinstance(raw_json_str, str) else raw_json_str
            if isinstance(parsed, dict) and "queries" in parsed:
                raw_variants = parsed["queries"]
            elif isinstance(parsed, list):
                raw_variants = parsed
            else:
                raw_variants = []

            orig_refs = extract_legal_references(norm_q0)
            seen_dedup = {normalize_dedup_key(norm_q0)}

            for item in raw_variants:
                if not isinstance(item, dict):
                    continue
                t_val = unicodedata.normalize("NFC", str(item.get("text", ""))).strip()
                f_val = str(item.get("focus", "paraphrase")).strip()

                if not t_val or len(t_val) > mq_max_chars:
                    continue

                # Kiểm tra bịa thêm số Điều/Khoản không có trong Q0
                item_refs = extract_legal_references(t_val)
                invented_refs = item_refs - orig_refs
                if invented_refs:
                    warnings.append(f"Loại bỏ query chứa số Điều/Khoản bịa thêm ({', '.join(invented_refs)}): '{t_val}'")
                    continue

                d_key = normalize_dedup_key(t_val)
                if d_key in seen_dedup:
                    dropped_duplicates += 1
                    continue
                seen_dedup.add(d_key)

                q_idx = len(queries_list)
                queries_list.append({
                    "query_id": f"Q{q_idx}",
                    "text": t_val,
                    "origin": "generated",
                    "focus": f_val if f_val in ("exact_legal_terms", "paraphrase", "missing_aspect") else "paraphrase"
                })

                if len(queries_list) >= mq_count + 1:
                    break

        except Exception as e:
            status = "query_generation_unavailable"
            warnings.append(f"Lỗi parse JSON kết quả sinh Multi-query: {e}")

    result_obj = {
        "original_question": norm_q0,
        "queries": queries_list,
        "model": model_name,
        "generation_latency_ms": latency_ms,
        "status": status,
        "cache_hit": False,
        "dropped_duplicate_count": dropped_duplicates,
        "warnings": warnings
    }

    if status == "ready":
        _QUERY_VARIANTS_CACHE[cache_key] = dict(result_obj)

    return result_obj


def retrieve_multi_query_child_hits(
    question: str,
    strategy: str = "hierarchical",
    custom_config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Thực thi Fan-Out Retrieval cho tập câu hỏi Multi-Query và hợp nhất kết quả qua Cross-Query RRF Fusion.
    - Chạy Hybrid Search (BM25 + Semantic -> Inner RRF) cho từng query Q0..Qn.
    - Lấy tối đa PER_QUERY_CANDIDATES child hits cho mỗi query.
    - Hợp nhất bằng công thức Cross-Query RRF Fusion với trọng số Q0 > Q_variant.
    - Đảm bảo tính toàn vẹn metadata (mismatch fail), deduplicate và sắp xếp deterministic.
    """
    import unicodedata
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi không được để rỗng.")

    norm_q0 = unicodedata.normalize("NFC", question).strip()
    cfg = get_hierarchical_config(custom_config)

    # 1. Sinh câu hỏi biến thể Multi-Query
    query_set = generate_query_variants(
        question=norm_q0,
        custom_config=cfg,
        query_generator_fn=query_generator_fn
    )

    queries = query_set["queries"]
    per_query_candidates: Dict[str, List[Dict[str, Any]]] = {}
    query_retrieval_latencies: Dict[str, float] = {}
    query_result_counts: Dict[str, int] = {}
    query_errors: Dict[str, str] = {}
    failed_queries: List[str] = []

    # 2. Per-Query Hybrid Retrieval (Fan-Out)
    for q_item in queries:
        q_id = q_item["query_id"]
        q_text = q_item["text"]

        t_start = time.perf_counter()
        try:
            if hybrid_retriever_fn is not None:
                h_res = hybrid_retriever_fn(q_text, strategy, cfg)
            else:
                import advanced_rag
                h_res = advanced_rag.hybrid_retrieval(
                    question=q_text,
                    strategy=strategy,
                    custom_config=cfg
                )

            raw_cands = h_res.get("candidates", []) if isinstance(h_res, dict) else h_res
            top_cands = raw_cands[:cfg["per_query_candidates"]]

            # Gán 1-based inner rank nếu chưa có
            for idx, c in enumerate(top_cands, start=1):
                if "fused_rank" not in c:
                    c["fused_rank"] = idx

            per_query_candidates[q_id] = top_cands
            query_result_counts[q_id] = len(top_cands)

        except Exception as e:
            err_str = str(e)
            if "chưa tồn tại" in err_str or "collection" in err_str.lower() or "prepare-semantic" in err_str.lower():
                return {
                    "status": "collection_not_ready",
                    "question": norm_q0,
                    "query_set": query_set,
                    "child_hits": [],
                    "warnings": [f"collection_not_ready: {e}. Vui lòng bấm nút 'Chuẩn bị Vector DB (prepare-semantic)' ở Sidebar trước khi hỏi đáp."],
                    "trace": {
                        "query_count_total": len(queries),
                        "query_count_valid": 0,
                        "query_count_failed": len(queries),
                        "failed_query_ids": [q["query_id"] for q in queries],
                        "query_result_counts": {},
                        "query_retrieval_latencies_ms": {},
                        "cross_rrf_fusion_latency_ms": 0.0,
                        "union_child_count": 0,
                        "overlap_distribution": {}
                    }
                }
            if q_id == "Q0":
                raise RuntimeError(f"Lỗi tìm kiếm Hybrid Search cho câu hỏi gốc Q0: {e}")
            else:
                failed_queries.append(q_id)
                query_errors[q_id] = str(e)
                per_query_candidates[q_id] = []
                query_result_counts[q_id] = 0

        query_retrieval_latencies[q_id] = round((time.perf_counter() - t_start) * 1000.0, 2)

    # 3. Cross-Query RRF Fusion
    t_cq_start = time.perf_counter()
    rrf_k = cfg["multi_query_rrf_k"]
    w_orig = cfg["multi_query_original_weight"]
    w_var = cfg["multi_query_variant_weight"]

    child_map: Dict[str, Dict[str, Any]] = {}
    child_per_query_ranks: Dict[str, Dict[str, int]] = {}
    child_per_query_trace: Dict[str, Dict[str, Any]] = {}
    child_query_ids: Dict[str, List[str]] = {}

    for q_id, cands in per_query_candidates.items():
        for c in cands:
            cid = str(c["chunk_id"]).strip()
            rank_q = int(c.get("fused_rank", cands.index(c) + 1))

            if cid not in child_map:
                child_map[cid] = {
                    "child_id": cid,
                    "text": c["text"],
                    "source": c["source"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"]
                }
                child_per_query_ranks[cid] = {}
                child_per_query_trace[cid] = {}
                child_query_ids[cid] = []
            else:
                # Metadata Mismatch Check
                existing = child_map[cid]
                if (existing["text"] != c["text"] or existing["source"] != c["source"] or
                        existing["page_start"] != c["page_start"] or existing["page_end"] != c["page_end"]):
                    raise ValueError(f"Metadata mismatch phát hiện cho child_id '{cid}' giữa các query.")

            if q_id not in child_per_query_ranks[cid]:
                child_per_query_ranks[cid][q_id] = rank_q
                child_query_ids[cid].append(q_id)
                child_per_query_trace[cid][q_id] = {
                    "inner_rrf_rank": rank_q,
                    "bm25_rank": c.get("bm25_rank"),
                    "semantic_rank": c.get("semantic_rank")
                }

    # Tính toán Multi-Query RRF Score
    merged_child_hits = []
    for cid, base_info in child_map.items():
        q_ranks = child_per_query_ranks[cid]
        q_ids = sorted(child_query_ids[cid], key=lambda x: (0 if x == "Q0" else int(x[1:]) if x[1:].isdigit() else 999))

        mq_rrf_score = 0.0
        for q_id, r in q_ranks.items():
            w = w_orig if q_id == "Q0" else w_var
            mq_rrf_score += w / (rrf_k + r)

        best_rank = min(q_ranks.values())
        supp_count = len(q_ids)

        merged_hit = dict(base_info)
        merged_hit["multi_query_rrf_score"] = round(mq_rrf_score, 6)
        merged_hit["support_query_count"] = supp_count
        merged_hit["support_query_ids"] = q_ids
        merged_hit["per_query_ranks"] = q_ranks
        merged_hit["best_query_rank"] = best_rank
        merged_hit["per_query_trace"] = child_per_query_trace[cid]
        merged_child_hits.append(merged_hit)

    # Sort child hits theo 4 tiêu chí:
    merged_child_hits.sort(
        key=lambda item: (
            -item["multi_query_rrf_score"],
            -item["support_query_count"],
            item["best_query_rank"],
            item["child_id"]
        )
    )

    # Gán multi_query_rank từ 1
    for rank_idx, hit in enumerate(merged_child_hits, start=1):
        hit["multi_query_rank"] = rank_idx

    # Overlap distribution
    overlap_dist: Dict[str, int] = {}
    for hit in merged_child_hits:
        cnt_key = f"{hit['support_query_count']}_query" if hit['support_query_count'] == 1 else f"{hit['support_query_count']}_queries"
        overlap_dist[cnt_key] = overlap_dist.get(cnt_key, 0) + 1

    # Pipeline Status
    if failed_queries:
        overall_status = "multi_query_partial"
    elif query_set["status"] != "ready":
        overall_status = "multi_query_partial"
    else:
        overall_status = "ready"

    cq_fusion_latency = round((time.perf_counter() - t_cq_start) * 1000.0, 2)

    trace = {
        "query_count_requested": cfg["multi_query_count"],
        "query_count_valid": len(queries),
        "query_count_executed": len(per_query_candidates),
        "query_count_failed": len(failed_queries),
        "failed_query_ids": failed_queries,
        "failed_query_errors": query_errors,
        "generation_latency_ms": query_set["generation_latency_ms"],
        "retrieval_latency_ms_per_query": query_retrieval_latencies,
        "result_count_per_query": query_result_counts,
        "union_child_count": len(merged_child_hits),
        "overlap_distribution": overlap_dist,
        "cross_rrf_fusion_latency_ms": cq_fusion_latency
    }

    return {
        "status": overall_status,
        "question": norm_q0,
        "query_set": query_set,
        "child_hits": merged_child_hits,
        "trace": trace
    }


def load_hierarchy_store(target_dir: Optional[Union[str, Path]] = None) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Tải dữ liệu Hierarchy Store (children.json, parents.json, manifest.json) từ đường dẫn đĩa."""
    if target_dir is None:
        store_dir = BASE_DIR / "storage" / "hierarchy"
    else:
        store_dir = Path(target_dir)

    c_file = store_dir / "children.json"
    p_file = store_dir / "parents.json"
    m_file = store_dir / "manifest.json"

    if not (c_file.exists() and p_file.exists() and m_file.exists()):
        return None, None, None

    try:
        with open(c_file, "r", encoding="utf-8") as f:
            children_list = json.load(f)
        with open(p_file, "r", encoding="utf-8") as f:
            parents_list = json.load(f)
        with open(m_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        return children_list, parents_list, manifest_data
    except Exception:
        return None, None, None


def retrieve_parent_candidates(
    question: str,
    mode: str = "multi_parent",
    strategy: str = "hierarchical",
    custom_config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    hierarchy_store_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Thực thi 'Retrieve Child, Return Parent' (Mở rộng bối cảnh Child Hits sang Parent Documents).
    - Kiểm tra tính sẵn sàng của Hierarchy Store (children.json, parents.json, manifest.json).
    - Ánh xạ từng fused Child Hit sang Parent Document tương ứng.
    - Tính Parent RRF Score từ Top PARENT_SCORE_CHILD_LIMIT Child Hits có multi_query_rank tốt nhất.
    - Khống chế ngân sách Context TOTAL_CONTEXT_MAX_CHARS theo ranh giới Parent.
    - Hỗ trợ hai chế độ: 'single_parent' và 'multi_parent'.
    """
    import unicodedata
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi không được để rỗng.")

    norm_q0 = unicodedata.normalize("NFC", question).strip()
    cfg = get_hierarchical_config(custom_config)

    # 1. Precondition Check: Load Hierarchy Store
    children_list, parents_list, manifest_data = load_hierarchy_store(hierarchy_store_dir)
    if children_list is None or parents_list is None or manifest_data is None:
        return {
            "status": "hierarchy_not_ready",
            "question": norm_q0,
            "mode": mode,
            "parent_candidates": [],
            "warnings": ["Hierarchy Store chưa được khởi tạo hoặc dữ liệu bị thiếu. Hãy chạy lệnh 'build-hierarchy' trước."],
            "trace": {}
        }

    t0 = time.perf_counter()

    child_registry = {str(c["child_id"]).strip(): c for c in children_list}
    parent_registry = {str(p["parent_id"]).strip(): p for p in parents_list}

    # 2. Modes Execution: single_parent vs multi_parent
    if mode == "single_parent":
        single_query_set = {
            "original_question": norm_q0,
            "queries": [{"query_id": "Q0", "text": norm_q0, "origin": "original", "focus": "original_intent"}],
            "model": cfg["generation_model"],
            "generation_latency_ms": 0.0,
            "status": "ready",
            "cache_hit": False,
            "dropped_duplicate_count": 0,
            "warnings": []
        }
        t_ret_start = time.perf_counter()
        if hybrid_retriever_fn is not None:
            h_res = hybrid_retriever_fn(norm_q0, strategy, cfg)
        else:
            import advanced_rag
            h_res = advanced_rag.hybrid_retrieval(question=norm_q0, strategy=strategy, custom_config=cfg)

        raw_cands = h_res.get("candidates", []) if isinstance(h_res, dict) else h_res
        top_cands = raw_cands[:cfg["per_query_candidates"]]

        child_hits = []
        for idx, c in enumerate(top_cands, start=1):
            child_hits.append({
                "child_id": str(c["chunk_id"]).strip(),
                "text": c["text"],
                "source": c["source"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "multi_query_rrf_score": round(1.0 / (cfg["multi_query_rrf_k"] + idx), 6),
                "multi_query_rank": idx,
                "support_query_count": 1,
                "support_query_ids": ["Q0"],
                "per_query_ranks": {"Q0": idx},
                "best_query_rank": idx,
                "per_query_trace": {"Q0": {"inner_rrf_rank": idx}}
            })

        multi_res = {
            "status": "ready",
            "question": norm_q0,
            "query_set": single_query_set,
            "child_hits": child_hits,
            "trace": {
                "generation_latency_ms": 0.0,
                "retrieval_latency_ms_per_query": {"Q0": round((time.perf_counter() - t_ret_start) * 1000.0, 2)},
                "result_count_per_query": {"Q0": len(child_hits)},
                "union_child_count": len(child_hits),
                "overlap_distribution": {"1_query": len(child_hits)}
            }
        }
    else:
        multi_res = retrieve_multi_query_child_hits(
            question=norm_q0,
            strategy=strategy,
            custom_config=cfg,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn
        )

    child_hits = multi_res["child_hits"]
    status = multi_res["status"]

    # 3. Child to Parent Mapping & Grouping
    parent_groups: Dict[str, List[Dict[str, Any]]] = {}
    child_to_parent_map_table: List[Dict[str, str]] = []

    for hit in child_hits:
        cid = hit["child_id"]
        if cid not in child_registry:
            raise KeyError(f"Child ID '{cid}' không tìm thấy trong Children Registry.")

        child_rec = child_registry[cid]
        pid = child_rec["parent_id"]

        if not pid or pid not in parent_registry:
            raise KeyError(f"Parent ID '{pid}' tương ứng với Child '{cid}' không tồn tại trong Parent Store.")

        parent_groups.setdefault(pid, []).append(hit)
        child_to_parent_map_table.append({"child_id": cid, "parent_id": pid})

    # 4. Parent Aggregation
    p_rrf_k = cfg["parent_rrf_k"]
    score_child_limit = cfg["parent_score_child_limit"]
    all_parent_candidates: List[Dict[str, Any]] = []

    for pid in sorted(list(parent_groups.keys())):
        p_hits = parent_groups[pid]
        p_hits.sort(key=lambda item: item["multi_query_rank"])

        supporting_child_ids = [h["child_id"] for h in p_hits]
        scoring_child_hits = p_hits[:score_child_limit]
        scoring_child_ids = [h["child_id"] for h in scoring_child_hits]

        anchor_child_id = p_hits[0]["child_id"]
        best_child_rank = p_hits[0]["multi_query_rank"]

        p_rrf_score = sum(1.0 / (p_rrf_k + h["multi_query_rank"]) for h in scoring_child_hits)

        supp_q_set = set()
        for h in p_hits:
            supp_q_set.update(h["support_query_ids"])
        supp_q_ids = sorted(list(supp_q_set), key=lambda x: (0 if x == "Q0" else int(x[1:]) if x[1:].isdigit() else 999))

        p_obj = dict(parent_registry[pid])

        cand_item = {
            "parent_id": pid,
            "source": p_obj["source"],
            "page_start": p_obj["page_start"],
            "page_end": p_obj["page_end"],
            "structural_path": p_obj.get("structural_path", {}),
            "heading": p_obj.get("heading", ""),
            "text": p_obj["text"],
            "char_count": p_obj["char_count"],
            "parent_rrf_score": round(p_rrf_score, 6),
            "parent_rank": 0,
            "anchor_child_id": anchor_child_id,
            "scoring_child_ids": scoring_child_ids,
            "supporting_child_ids": supporting_child_ids,
            "support_query_ids": supp_q_ids,
            "best_child_rank": best_child_rank,
            "ambiguous": any(h.get("ambiguous", False) for h in p_hits),
            "warnings": list(p_obj.get("warnings", []))
        }
        all_parent_candidates.append(cand_item)

    # Sort parent candidates
    all_parent_candidates.sort(
        key=lambda p: (
            -p["parent_rrf_score"],
            -len(p["support_query_ids"]),
            p["best_child_rank"],
            p["parent_id"]
        )
    )

    parents_before_budget = all_parent_candidates[:cfg["parent_candidates"]]
    for r_idx, p in enumerate(parents_before_budget, start=1):
        p["parent_rank"] = r_idx

    # 5. Context Budgeting (TOTAL_CONTEXT_MAX_CHARS)
    tot_budget = cfg["total_context_max_chars"]
    accepted_candidates: List[Dict[str, Any]] = []
    curr_chars = 0
    dropped_by_budget = 0

    for p in parents_before_budget:
        p_len = p["char_count"]
        if not accepted_candidates and p_len > tot_budget:
            p["warnings"].append("oversized_first_parent: Parent đầu tiên vượt budget nhưng được giữ nguyên để tránh trả context rỗng.")
            accepted_candidates.append(p)
            curr_chars += p_len
            break
        elif curr_chars + p_len <= tot_budget:
            accepted_candidates.append(p)
            curr_chars += p_len
        else:
            dropped_by_budget += 1

    child_chars_total = sum(len(h["text"]) for h in child_hits)
    expanded_parent_chars_total = curr_chars
    context_expansion_factor = round(expanded_parent_chars_total / child_chars_total, 2) if child_chars_total > 0 else 1.0

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    parent_trace = {
        "input_child_hit_count": len(child_hits),
        "unique_parent_count": len(parent_groups),
        "children_per_parent": {pid: len(hits) for pid, hits in parent_groups.items()},
        "child_to_parent_mapping_table": child_to_parent_map_table,
        "parents_dropped_by_candidate_limit": len(all_parent_candidates) - len(parents_before_budget),
        "parents_dropped_by_context_budget": dropped_by_budget,
        "child_chars_total": child_chars_total,
        "expanded_parent_chars_total": expanded_parent_chars_total,
        "context_expansion_factor": context_expansion_factor,
        "mapping_and_aggregation_latency_ms": latency_ms,
        "multi_query_trace": multi_res.get("trace", {})
    }

    return {
        "status": status,
        "question": norm_q0,
        "mode": mode,
        "query_set": multi_res["query_set"],
        "parent_candidates": accepted_candidates,
        "child_hits": child_hits,
        "trace": parent_trace
    }


def rerank_parent_candidates(
    question: str,
    parent_candidates: List[Dict[str, Any]],
    custom_config: Optional[Dict[str, Any]] = None,
    reranker_fn: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Rerank các Parent Candidate bằng Cross-Encoder model dựa trên cặp (original_question, parent_text).
    - Chỉ rerank tối đa PARENT_CANDIDATES (default 10).
    - Tính parent_rerank_raw_score và parent_rerank_score = sigmoid(raw_score).
    - Tính parent_rerank_rank và parent_rank_change = parent_rank - parent_rerank_rank.
    - Sắp xếp theo:
        1. parent_rerank_score DESC
        2. parent_rank (RRF rank) ASC
        3. parent_id ASC
    """
    import math
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi không được để rỗng.")

    if not parent_candidates:
        return []

    cfg = get_hierarchical_config(custom_config)
    max_cands = cfg["parent_candidates"]
    targets = [dict(p) for p in parent_candidates[:max_cands]]
    texts = [p["text"] for p in targets]

    if cfg.get("use_fast_reranker"):
        # Fast RRF Parent Reranker: Chuyển đổi trực tiếp điểm RRF sang logit và sigmoid score
        scores_raw = []
        for p in targets:
            rrf_s = p.get("parent_rrf_score", 0.05)
            # Map score sang range logit dương (ví dụ: rrf 0.03 -> logit 2.0 -> sigmoid 0.88)
            logit_val = math.log(max(rrf_s * 50.0, 1.1))
            scores_raw.append(logit_val)
    elif reranker_fn is not None:
        try:
            scores_raw = reranker_fn(question, texts, cfg)
        except Exception as e:
            raise RuntimeError(f"reranker_unavailable: Lỗi khi gọi Reranker: {e}")
    else:
        try:
            import advanced_rag
            tokenizer, model, device = advanced_rag.load_reranker_model(
                model_name=cfg["reranker_model"],
                device_setting=cfg["rerank_device"]
            )
            import torch
            pairs = [[question, text] for text in texts]
            batch_size = cfg["rerank_batch_size"]
            scores_raw = []

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i: i + batch_size]
                inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=cfg["reranker_max_length"],
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        logits = logits.squeeze(-1)
                    elif logits.ndim == 2 and logits.shape[1] > 1:
                        logits = logits[:, 0]

                    for logit in logits:
                        scores_raw.append(float(logit.item()))
        except Exception as e:
            raise RuntimeError(f"reranker_unavailable: Lỗi khi gọi Cross-Encoder Reranker: {e}")

    reranked_parents = []
    for idx, p in enumerate(targets):
        raw_score = float(scores_raw[idx])
        sig_score = round(1.0 / (1.0 + math.exp(-raw_score)), 6)
        p["parent_rerank_raw_score"] = round(raw_score, 4)
        p["parent_rerank_score"] = sig_score
        reranked_parents.append(p)

    reranked_parents.sort(
        key=lambda item: (
            -item["parent_rerank_score"],
            item["parent_rank"],
            item["parent_id"]
        )
    )

    for r_idx, p in enumerate(reranked_parents, start=1):
        p["parent_rerank_rank"] = r_idx
        p["parent_rank_change"] = p["parent_rank"] - r_idx

    final_top_k = cfg["final_parent_top_k"]
    return reranked_parents[:final_top_k]


def generate_answer_text(
    question: str,
    context_str: str,
    citations: List[Dict[str, Any]],
    custom_config: Optional[Dict[str, Any]] = None
) -> str:
    """Sinh câu trả lời chuyên sâu bằng Gemini API dựa trên ngữ cảnh trích dẫn."""
    cfg = get_hierarchical_config(custom_config)
    if not cfg.get("api_key"):
        return "Thiếu GEMINI_API_KEY để sinh câu trả lời."

    from google import genai

    client = genai.Client(api_key=cfg["api_key"])
    model_name = cfg["generation_model"]

    prompt = f"""Bạn là trợ lý AI chuyên gia tư vấn pháp luật ngân hàng Việt Nam và thương mại quốc tế.
Nhiệm vụ: Dựa TRỰC TIẾP và CHI THUẦN vào ngữ cảnh văn bản được cung cấp bên dưới, hãy trả lời câu hỏi một cách chính xác, đầy đủ và khách quan.

Quy tắc trích dẫn bắt buộc:
1. Khi sử dụng thông tin từ một đoạn ngữ cảnh, bạn PHẢI trích dẫn nhãn tương ứng (ví dụ: [P1], [P2] hoặc [C1], [C2]) ở ngay cuối câu hoặc cuối đoạn thông tin đó.
2. KHÔNG được sử dụng nhãn trích dẫn giả lập không có trong danh sách ngữ cảnh.
3. Nếu ngữ cảnh không chứa đủ thông tin để trả lời, hãy nêu rõ rằng ngữ cảnh hiện tại chưa đủ căn cứ pháp lý để kết luận.

--- BẮT ĐẦU NGỮ CẢNH VĂN BẢN ---
{context_str}
--- KẾT THÚC NGỮ CẢNH VĂN BẢN ---

Câu hỏi của người dùng: "{question}"

Trả lời:"""

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return resp.text.strip() if resp and resp.text else "Không nhận được câu trả lời từ Gemini."


def query_hierarchical_rag(
    question: str,
    mode: str = "multi_parent",
    strategy: str = "hierarchical",
    custom_config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
    generate_answer_fn: Optional[Any] = None,
    hierarchy_store_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Pipeline Hỏi đáp Toàn diện Multi-Query & Parent-Child Hierarchical RAG (Buổi 09).
    Các Mode hỗ trợ:
    - single_flat: Q0 -> Hybrid Search -> Rerank Child Chunks bằng Q0 -> Gate Evidence -> Answer.
    - multi_flat: Q0 + variants -> Hybrid -> MQ-RRF -> Rerank Child Chunks bằng Q0 -> Gate Evidence -> Answer.
    - single_parent: Q0 -> Hybrid -> Child-to-Parent -> Parent Aggregation -> Rerank Parent bằng Q0 -> Gate Evidence -> Answer.
    - multi_parent (Default): Q0 + variants -> Hybrid -> MQ-RRF -> Child-to-Parent -> Parent Aggregation -> Rerank Parent bằng Q0 -> Gate Evidence -> Answer.
    """
    import unicodedata
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi không được để rỗng.")

    valid_modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    if mode not in valid_modes:
        raise ValueError(f"Mode '{mode}' không hợp lệ. Chỉ chấp nhận: {valid_modes}")

    norm_q0 = unicodedata.normalize("NFC", question).strip()
    cfg = get_hierarchical_config(custom_config)

    t_start = time.perf_counter()
    gen_api_calls = 0
    emb_api_calls = 0
    warnings: List[str] = []

    stage_latencies: Dict[str, float] = {}

    # 1. Query Expansion
    t_exp_start = time.perf_counter()
    if mode in ["multi_flat", "multi_parent"]:
        try:
            query_set = generate_query_variants(
                question=norm_q0,
                custom_config=cfg,
                query_generator_fn=query_generator_fn
            )
            if query_set.get("status") in ["query_generation_unavailable", "query_generation_failed"]:
                return {
                    "status": "query_generation_unavailable",
                    "mode": mode,
                    "original_question": norm_q0,
                    "query_set": query_set,
                    "child_hits": [],
                    "parent_candidates": [],
                    "accepted_evidence": [],
                    "answer": "",
                    "citations": [],
                    "warnings": query_set.get("warnings", ["query_generation_unavailable"]),
                    "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": 0, "embedding_calls": 0}}
                }
            if not query_set.get("cache_hit", True) and query_set.get("status") == "ready":
                gen_api_calls += 1
            if query_set.get("warnings"):
                warnings.extend(query_set["warnings"])
        except Exception as e:
            return {
                "status": "query_generation_unavailable",
                "mode": mode,
                "original_question": norm_q0,
                "query_set": {"original_question": norm_q0, "queries": [{"query_id": "Q0", "text": norm_q0, "origin": "original", "focus": "original_intent"}]},
                "child_hits": [],
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": "",
                "citations": [],
                "warnings": [f"query_generation_unavailable: {e}"],
                "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": 0, "embedding_calls": 0}}
            }
    else:
        query_set = {
            "original_question": norm_q0,
            "queries": [{"query_id": "Q0", "text": norm_q0, "origin": "original", "focus": "original_intent"}],
            "model": cfg["generation_model"],
            "generation_latency_ms": 0.0,
            "status": "ready",
            "cache_hit": False,
            "dropped_duplicate_count": 0,
            "warnings": []
        }

    stage_latencies["query_expansion"] = round((time.perf_counter() - t_exp_start) * 1000.0, 2)
    emb_api_calls += len(query_set["queries"])

    # 2. Retrieval & Aggregation Stage
    t_ret_start = time.perf_counter()

    if mode in ["single_parent", "multi_parent"]:
        parent_res = retrieve_parent_candidates(
            question=norm_q0,
            mode=mode,
            strategy=strategy,
            custom_config=cfg,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            hierarchy_store_dir=hierarchy_store_dir
        )

        if parent_res.get("status") in ["hierarchy_not_ready", "collection_not_ready"]:
            return {
                "status": parent_res["status"],
                "mode": mode,
                "original_question": norm_q0,
                "query_set": query_set,
                "child_hits": [],
                "parent_candidates": [],
                "accepted_evidence": [],
                "answer": "",
                "citations": [],
                "warnings": parent_res.get("warnings", []),
                "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": gen_api_calls, "embedding_calls": emb_api_calls}}
            }

        child_hits = parent_res.get("child_hits", [])
        parent_candidates = parent_res.get("parent_candidates", [])
        if parent_res.get("warnings"):
            warnings.extend(parent_res["warnings"])
    else:
        parent_candidates = []
        if mode == "multi_flat":
            multi_child_res = retrieve_multi_query_child_hits(
                question=norm_q0,
                strategy=strategy,
                custom_config=cfg,
                query_generator_fn=query_generator_fn,
                hybrid_retriever_fn=hybrid_retriever_fn
            )
            if multi_child_res.get("status") in ["hierarchy_not_ready", "collection_not_ready"]:
                return {
                    "status": multi_child_res["status"],
                    "mode": mode,
                    "original_question": norm_q0,
                    "query_set": query_set,
                    "child_hits": [],
                    "parent_candidates": [],
                    "accepted_evidence": [],
                    "answer": "",
                    "citations": [],
                    "warnings": multi_child_res.get("warnings", []),
                    "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": gen_api_calls, "embedding_calls": emb_api_calls}}
                }
            child_hits = multi_child_res.get("child_hits", [])
        else:
            try:
                if hybrid_retriever_fn is not None:
                    h_res = hybrid_retriever_fn(norm_q0, strategy, cfg)
                else:
                    import advanced_rag
                    h_res = advanced_rag.hybrid_retrieval(question=norm_q0, strategy=strategy, custom_config=cfg)
            except Exception as e:
                err_str = str(e)
                if "chưa tồn tại" in err_str or "collection" in err_str.lower() or "prepare-semantic" in err_str.lower():
                    return {
                        "status": "collection_not_ready",
                        "mode": mode,
                        "original_question": norm_q0,
                        "query_set": query_set,
                        "child_hits": [],
                        "parent_candidates": [],
                        "accepted_evidence": [],
                        "answer": "",
                        "citations": [],
                        "warnings": [f"collection_not_ready: {e}. Vui lòng bấm nút 'Chuẩn bị Vector DB (prepare-semantic)' ở Sidebar trước khi hỏi đáp."],
                        "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": gen_api_calls, "embedding_calls": emb_api_calls}}
                    }
                raise e

            raw_cands = h_res.get("candidates", []) if isinstance(h_res, dict) else h_res
            top_cands = raw_cands[:cfg["per_query_candidates"]]
            child_hits = []
            for idx, c in enumerate(top_cands, start=1):
                child_hits.append({
                    "child_id": str(c["chunk_id"]).strip(),
                    "text": c["text"],
                    "source": c["source"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "multi_query_rrf_score": round(1.0 / (cfg["rrf_k"] + idx), 6),
                    "multi_query_rank": idx,
                    "support_query_count": 1,
                    "support_query_ids": ["Q0"],
                    "per_query_ranks": {"Q0": idx},
                    "best_query_rank": idx
                })

    stage_latencies["retrieval_and_aggregation"] = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

    # 3. Reranking Stage
    t_rr_start = time.perf_counter()
    if mode in ["single_parent", "multi_parent"]:
        if not parent_candidates:
            accepted_evidence = []
            reranked_parents = []
        else:
            try:
                reranked_parents = rerank_parent_candidates(
                    question=norm_q0,
                    parent_candidates=parent_candidates,
                    custom_config=cfg,
                    reranker_fn=reranker_fn
                )
            except Exception as e:
                return {
                    "status": "reranker_unavailable",
                    "mode": mode,
                    "original_question": norm_q0,
                    "query_set": query_set,
                    "child_hits": child_hits,
                    "parent_candidates": parent_candidates,
                    "accepted_evidence": [],
                    "answer": "",
                    "citations": [],
                    "warnings": [f"reranker_unavailable: {e}"],
                    "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": gen_api_calls, "embedding_calls": emb_api_calls}}
                }

            accepted_evidence = [p for p in reranked_parents if p["parent_rerank_score"] >= cfg["rerank_min_score"]]
    else:
        if not child_hits:
            accepted_evidence = []
            reranked_children = []
        else:
            try:
                import advanced_rag
                raw_reranked = advanced_rag.rerank_candidates(
                    query=norm_q0,
                    candidates=child_hits,
                    top_k=cfg["final_top_k"],
                    custom_config=cfg,
                    reranker_fn=reranker_fn
                )
                reranked_children = raw_reranked
            except Exception as e:
                return {
                    "status": "reranker_unavailable",
                    "mode": mode,
                    "original_question": norm_q0,
                    "query_set": query_set,
                    "child_hits": child_hits,
                    "parent_candidates": [],
                    "accepted_evidence": [],
                    "answer": "",
                    "citations": [],
                    "warnings": [f"reranker_unavailable: {e}"],
                    "trace": {"stage_latencies_ms": {}, "api_call_counts": {"generation_calls": gen_api_calls, "embedding_calls": emb_api_calls}}
                }

            accepted_evidence = [c for c in reranked_children if c.get("rerank_score", 0.0) >= cfg["rerank_min_score"]]

    stage_latencies["reranking"] = round((time.perf_counter() - t_rr_start) * 1000.0, 2)

    # 4. Evidence Gate Check
    if not accepted_evidence:
        return {
            "status": "insufficient_evidence",
            "mode": mode,
            "original_question": norm_q0,
            "query_set": query_set,
            "child_hits": child_hits,
            "parent_candidates": parent_candidates if mode in ["single_parent", "multi_parent"] else [],
            "accepted_evidence": [],
            "answer": "Không có đủ chứng cứ pháp lý đạt ngưỡng để trả lời câu hỏi.",
            "citations": [],
            "warnings": warnings + ["Không có candidate nào vượt qua ngưỡng RERANK_MIN_SCORE."],
            "trace": {
                "stage_latencies_ms": stage_latencies,
                "api_call_counts": {"generation_calls": gen_api_calls, "embedding_calls": emb_api_calls}
            }
        }

    # 5. Build Citations Array & Answer Generation
    citations = []
    if mode in ["single_parent", "multi_parent"]:
        for idx, p in enumerate(accepted_evidence, start=1):
            lbl = f"P{idx}"
            citations.append({
                "evidence_id": lbl,
                "parent_id": p["parent_id"],
                "anchor_child_id": p["anchor_child_id"],
                "supporting_child_ids": p["supporting_child_ids"],
                "source": p["source"],
                "page_start": p["page_start"],
                "page_end": p["page_end"],
                "structural_path": p.get("structural_path", {}),
                "parent_rerank_score": p["parent_rerank_score"],
                "ambiguous": p.get("ambiguous", False),
                "warnings": p.get("warnings", [])
            })
    else:
        for idx, c in enumerate(accepted_evidence, start=1):
            lbl = f"C{idx}"
            citations.append({
                "evidence_id": lbl,
                "chunk_id": c.get("child_id", c.get("chunk_id")),
                "source": c["source"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "rerank_score": c.get("rerank_score", 0.0),
                "ambiguous": c.get("ambiguous", False),
                "warnings": c.get("warnings", [])
            })

    t_gen_start = time.perf_counter()
    if generate_answer_fn is not None:
        answer_text = generate_answer_fn(question=norm_q0, accepted_evidence=accepted_evidence, citations=citations, custom_config=cfg)
        gen_api_calls += 1
    else:
        try:
            import advanced_rag
            prompt_context_lines = []
            for cit in citations:
                eid = cit["evidence_id"]
                matching_p = next((p for p in accepted_evidence if p.get("parent_id") == cit.get("parent_id") or p.get("child_id") == cit.get("chunk_id")), None)
                txt = matching_p["text"] if matching_p else ""
                prompt_context_lines.append(f"[{eid}] (Nguồn: {cit['source']}, Trang {cit['page_start']}-{cit['page_end']}):\n{txt}")

            context_str = "\n\n".join(prompt_context_lines)
            answer_text = generate_answer_text(question=norm_q0, context_str=context_str, citations=citations, custom_config=cfg)
            gen_api_calls += 1
        except Exception as e:
            answer_text = f"[LỖI GENERATION] Không thể sinh câu trả lời từ Gemini: {e}"
            warnings.append(f"answer_generation_error: {e}")

    stage_latencies["generation"] = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
    stage_latencies["total"] = round((time.perf_counter() - t_start) * 1000.0, 2)

    valid_citation_ids = {c["evidence_id"] for c in citations}
    import re
    found_citations = set(re.findall(r"\[(P\d+|C\d+)\]", answer_text))
    invalid_found = found_citations - valid_citation_ids
    if invalid_found:
        warnings.append(f"invalid_citations_detected: Câu trả lời trích dẫn các nhãn không tồn tại trong evidence: {list(invalid_found)}")

    trace = {
        "stage_latencies_ms": stage_latencies,
        "api_call_counts": {
            "generation_calls": gen_api_calls,
            "embedding_calls": emb_api_calls
        },
        "model_config": {
            "embedding_model": cfg["embedding_model"],
            "reranker_model": cfg["reranker_model"],
            "generation_model": cfg["generation_model"]
        },
        "counts": {
            "query_count": len(query_set["queries"]),
            "child_hits_count": len(child_hits),
            "parent_candidates_count": len(parent_candidates),
            "accepted_evidence_count": len(accepted_evidence)
        },
        "warnings_and_errors": warnings
    }

    return {
        "status": "ready" if not warnings else "multi_query_partial",
        "mode": mode,
        "original_question": norm_q0,
        "query_set": query_set,
        "child_hits": child_hits,
        "parent_candidates": parent_candidates if mode in ["single_parent", "multi_parent"] else [],
        "accepted_evidence": accepted_evidence,
        "answer": answer_text,
        "citations": citations,
        "warnings": warnings,
        "trace": trace
    }


def compare_hierarchical_rag(
    question: str,
    strategy: str = "hierarchical",
    custom_config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None,
    hierarchy_store_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Chạy so sánh 4 chế độ Pipeline Retrieval/Reranking (single_flat, multi_flat, single_parent, multi_parent)
    nhưng KHÔNG gọi Gemini Answer Generation.
    """
    import unicodedata
    norm_q0 = unicodedata.normalize("NFC", question).strip()
    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    comparison_results = {}

    dummy_gen = lambda question, accepted_evidence, citations, custom_config: ""

    for m in modes:
        res = query_hierarchical_rag(
            question=norm_q0,
            mode=m,
            strategy=strategy,
            custom_config=custom_config,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            reranker_fn=reranker_fn,
            generate_answer_fn=dummy_gen,
            hierarchy_store_dir=hierarchy_store_dir
        )
        comparison_results[m] = res

    return {
        "question": norm_q0,
        "modes_compared": modes,
        "results": comparison_results
    }


def main():
    import unicodedata

    parser = argparse.ArgumentParser(description="Multi-Query & Parent-Child Hierarchical RAG CLI — Buổi 09")
    subparsers = parser.add_subparsers(dest="command", help="Lệnh thực thi")

    # Command hierarchy-audit
    parser_audit = subparsers.add_parser("hierarchy-audit", help="Audit phân giải cấu trúc Parent-Child từ Chunks")
    parser_audit.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")
    parser_audit.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")

    # Command build-hierarchy
    parser_build = subparsers.add_parser("build-hierarchy", help="Xây dựng Hierarchy Registry và lưu Store")
    parser_build.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")
    parser_build.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")

    # Command hierarchy-status
    subparsers.add_parser("hierarchy-status", help="Kiểm tra trạng thái Hierarchy Store (Read-Only)")

    # Command expand-query
    parser_expand = subparsers.add_parser("expand-query", help="Sinh tập câu hỏi tìm kiếm Multi-Query Expansion")
    parser_expand.add_argument("--question", type=str, required=True, help="Câu hỏi pháp lý cần mở rộng")

    # Command multi-child
    parser_multichild = subparsers.add_parser("multi-child", help="Thực thi Multi-Query Fan-Out Retrieval và Cross-Query RRF Fusion")
    parser_multichild.add_argument("--question", type=str, required=True, help="Câu hỏi pháp lý cần tìm kiếm")
    parser_multichild.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")

    # Command parent-retrieve
    parser_parent = subparsers.add_parser("parent-retrieve", help="Thực thi Retrieve Child, Return Parent Candidate Expansion")
    parser_parent.add_argument("--question", type=str, required=True, help="Câu hỏi pháp lý cần truy xuất")
    parser_parent.add_argument("--mode", type=str, default="multi_parent", choices=["single_parent", "multi_parent"], help="Chế độ retrieval")
    parser_parent.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")

    # Command query
    parser_query = subparsers.add_parser("query", help="Thực thi Pipeline Hỏi đáp RAG Hoàn chỉnh")
    parser_query.add_argument("--question", type=str, required=True, help="Câu hỏi pháp lý")
    parser_query.add_argument("--mode", type=str, default="multi_parent", choices=["single_flat", "multi_flat", "single_parent", "multi_parent"], help="Chế độ hỏi đáp")
    parser_query.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")

    # Command compare
    parser_compare = subparsers.add_parser("compare", help="So sánh 4 chế độ Retrieval & Reranking (Không sinh câu trả lời)")
    parser_compare.add_argument("--question", type=str, required=True, help="Câu hỏi pháp lý")
    parser_compare.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")

    args = parser.parse_args()

    if args.command == "hierarchy-audit":
        try:
            children, parents, stats = build_hierarchy_registry(input_path=args.input_dir, strategy=args.strategy)
            print("\n=== KẾT QUẢ AUDIT HIERARCHY REGISTRY (BUỔI 09) ===")
            print(f"Tổng số Nguồn (Sources):     {stats['total_sources']}")
            print(f"Tổng số Child Chunks:        {stats['total_children']}")
            print(f"Tổng số Parent Documents:    {stats['total_parents']}")
            print(f"Số Child nghi ngờ (Ambiguous): {stats['ambiguous_children_count']}")
            print(f"Số Child vượt kích thước (Oversized): {stats['oversized_children_count']}\n")

            print("--- PHÂN BỔ KÍCH THƯỚC PARENT DOCUMENTS ---")
            p_lens = [p["char_count"] for p in parents]
            if p_lens:
                p_lens.sort()
                print(f"  Min: {p_lens[0]} ký tự | Median: {p_lens[len(p_lens)//2]} ký tự | Max: {p_lens[-1]} ký tự")

            print("\n--- MẪU HIERARCHY RESOLUTION (TOP 5 CHILDREN) ---")
            for c in children[:5]:
                method_badge = f"[{c['resolution_method'].upper()}]"
                amb_badge = "⚠️ AMBIGUOUS" if c["ambiguous"] else ""
                print(f"  • {c['child_id']} -> Parent: {c['parent_id']} {method_badge} {amb_badge}")
                print(f"    Structural Path: {c['structural_path']}")

        except Exception as e:
            print(f"\n[LỖI HIERARCHY-AUDIT] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "build-hierarchy":
        try:
            children, parents, stats = build_hierarchy_registry(input_path=args.input_dir, strategy=args.strategy)
            manifest = save_hierarchy_store(children, parents, stats)
            print("\n=== ĐÃ XÂY DỰNG VÀ LƯU HIERARCHY STORE THÀNH CÔNG ===")
            print(f"Store Path:               {BASE_DIR / 'storage' / 'hierarchy'}")
            print(f"Build Timestamp:          {manifest['build_timestamp']}")
            print(f"Tổng số Child Records:    {manifest['counts']['total_children']}")
            print(f"Tổng số Parent Docs:      {manifest['counts']['total_parents']}")

        except Exception as e:
            print(f"\n[LỖI BUILD-HIERARCHY] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "hierarchy-status":
        try:
            st = hierarchy_status()
            print("\n=== TRẠNG THÁI HIERARCHY STORE (READ-ONLY) ===")
            print(f"Store Path:     {st['store_dir']}")
            print(f"Store Tồn tại:  {'🟢 Đã khởi tạo' if st['store_exists'] else '🔴 Chưa khởi tạo'}")
            if st["store_exists"]:
                m = st["manifest"]
                print(f"Build Timestamp:{m.get('build_timestamp', 'N/A')}")
                print(f"Total Children: {m.get('counts', {}).get('total_children', 0)}")
                print(f"Total Parents:  {m.get('counts', {}).get('total_parents', 0)}")
        except Exception as e:
            print(f"\n[LỖI HIERARCHY-STATUS] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "expand-query":
        try:
            res = generate_query_variants(question=args.question)
            print("\n=== KẾT QUẢ MULTI-QUERY EXPANSION (BUỔI 09) ===")
            print(f"Câu hỏi gốc:         '{res['original_question']}'")
            print(f"Trạng thái:          {res['status'].upper()}")
            print(f"Model generation:    {res['model']}")
            print(f"Thời gian xử lý:     {res['generation_latency_ms']} ms")
            print(f"Cache Hit:           {'Có ⚡' if res['cache_hit'] else 'Không 🔄'}")
            print(f"Duplicates loại bỏ:  {res['dropped_duplicate_count']}")
            if res.get("warnings"):
                print(f"Cảnh báo ({len(res['warnings'])}):")
                for w in res["warnings"]:
                    print(f"  ⚠️ {w}")

            print(f"\nDanh sách Queries ({len(res['queries'])}):")
            for q in res["queries"]:
                badge = "[GỐC Q0]" if q["origin"] == "original" else f"[{q['focus'].upper()}]"
                print(f"  • {q['query_id']} {badge}: \"{q['text']}\"")

        except Exception as e:
            print(f"\n[LỖI EXPAND-QUERY] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "multi-child":
        try:
            res = retrieve_multi_query_child_hits(question=args.question, strategy=args.strategy)
            tr = res["trace"]
            print("\n=== KẾT QUẢ CROSS-QUERY CHILD RETRIEVAL (BUỔI 09) ===")
            print(f"Câu hỏi:             '{res['question']}'")
            print(f"Trạng thái:          {res['status'].upper()}")
            print(f"Số Query thực thi:   {tr['query_count_valid']} valid, {tr['query_count_failed']} failed")
            print(f"Tổng Union Child Hits: {tr['union_child_count']}")
            print(f"Phân bổ Overlap:      {tr['overlap_distribution']}")
            print(f"Cross RRF Latency:   {tr['cross_rrf_fusion_latency_ms']} ms")

            print("\n--- BẢNG CHI TIẾT MERGED CHILD HITS (TOP 10) ---")
            print(f"{'Rank':<5} | {'Child ID':<12} | {'MQ-RRF Score':<12} | {'Supp Count':<10} | {'Queries':<15} | {'Best Rank':<10}")
            print("-" * 75)
            for hit in res["child_hits"][:10]:
                q_str = ", ".join(hit["support_query_ids"])
                print(f"{hit['multi_query_rank']:<5} | {hit['child_id']:<12} | {hit['multi_query_rrf_score']:<12.6f} | {hit['support_query_count']:<10} | {q_str:<15} | {hit['best_query_rank']:<10}")

        except Exception as e:
            print(f"\n[LỖI MULTI-CHILD] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "parent-retrieve":
        try:
            res = retrieve_parent_candidates(question=args.question, mode=args.mode, strategy=args.strategy)
            if res["status"] == "hierarchy_not_ready":
                print("\n[CẢNH BÁO HIERARCHY STORE CHƯA SẴN SÀNG]")
                print(f"Status: {res['status']}")
                for w in res.get("warnings", []):
                    print(f"  ⚠️ {w}")
                sys.exit(1)

            tr = res["trace"]
            print(f"\n=== KẾT QUẢ PARENT RETRIEVAL (BUỔI 09 — {args.mode.upper()}) ===")
            print(f"Câu hỏi:             '{res['question']}'")
            print(f"Trạng thái:          {res['status'].upper()}")
            print(f"Số Parent Candidates: {len(res['parent_candidates'])}")
            print(f"Context Expansion:   {tr['expanded_parent_chars_total']} chars (Factor x{tr['context_expansion_factor']})")
            print(f"Latency Aggregation:  {tr['mapping_and_aggregation_latency_ms']} ms")

            print("\n--- MAPPING TREE (PARENT DOCUMENT ──> SUPPORTING CHILDREN) ---")
            for p in res["parent_candidates"]:
                print(f"\n• [Rank {p['parent_rank']}] Parent ID: {p['parent_id']} (Score: {p['parent_rrf_score']:.6f})")
                print(f"  Source: {p['source']} | Pages: {p['page_start']}-{p['page_end']} | Heading: {p['heading'][:50]}")
                print(f"  Supporting Child Chunks ({len(p['supporting_child_ids'])}):")
                for cid in p["supporting_child_ids"]:
                    # Tìm trace của child này
                    is_anchor = " [ANCHOR]" if cid == p["anchor_child_id"] else ""
                    is_scoring = " [SCORING]" if cid in p["scoring_child_ids"] else ""
                    print(f"  └── Child: {cid}{is_anchor}{is_scoring}")

        except Exception as e:
            print(f"\n[LỖI PARENT-RETRIEVE] {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()


