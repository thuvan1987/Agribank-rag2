"""
Buổi 09: Advanced RAG Pipeline Baseline (Baseline snapshot sao chép độc lập từ Buổi 08).

Nguồn baseline: rag_advance/buoi_08/advanced_rag.py
Mục đích: Cung cấp BM25, Semantic Candidate Retrieval, RRF Fusion và Cross-Encoder Reranker độc lập cho Buổi 09.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Thêm BASE_DIR vào sys.path để import rag.py local của Buổi 08
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag

# Load .env dựa trên vị trí tuyệt đối của BASE_DIR
load_dotenv(dotenv_path=ENV_PATH)

# Global process singleton cache cho Reranker Model
_RERANKER_CACHE: Dict[str, Any] = {
    "model_name": None,
    "device_setting": None,
    "tokenizer": None,
    "model": None,
    "device": None,
}


def get_advanced_config(custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Đọc, ép kiểu và xác thực toàn bộ thông số cấu hình cho Advanced RAG Pipeline.
    Hỗ trợ override trực tiếp qua dict custom_config phục vụ testing.
    """
    if custom_config is not None:
        cfg = dict(custom_config)
    else:
        cfg = {
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
        }

    for m_key in ["embedding_model", "generation_model", "reranker_model"]:
        if not cfg.get(m_key):
            raise ValueError(f"Cấu hình '{m_key}' không được để rỗng.")

    for c_key in ["bm25_candidates", "semantic_candidates", "rerank_candidates", "final_top_k"]:
        val = cfg.get(c_key, 0)
        if not isinstance(val, int) or not (1 <= val <= 100):
            raise ValueError(f"Cấu hình '{c_key}' ({val}) phải là số nguyên dương từ 1 đến 100.")

    if cfg["final_top_k"] > cfg["rerank_candidates"]:
        raise ValueError(
            f"FINAL_TOP_K ({cfg['final_top_k']}) không được lớn hơn RERANK_CANDIDATES ({cfg['rerank_candidates']})."
        )

    if cfg.get("rrf_k", 0) <= 0:
        raise ValueError(f"RRF_K ({cfg.get('rrf_k')}) phải là số nguyên dương > 0.")

    w_bm25 = cfg.get("rrf_bm25_weight", 0.0)
    w_sem = cfg.get("rrf_semantic_weight", 0.0)
    if w_bm25 < 0.0 or w_sem < 0.0:
        raise ValueError("Trọng số RRF (RRF_BM25_WEIGHT, RRF_SEMANTIC_WEIGHT) phải là số thực không âm (>= 0).")
    if w_bm25 == 0.0 and w_sem == 0.0:
        raise ValueError("Trọng số RRF không được đồng thời bằng 0.0.")

    max_len = cfg.get("reranker_max_length", 0)
    if not (64 <= max_len <= 4096):
        raise ValueError(f"RERANKER_MAX_LENGTH ({max_len}) phải nằm trong khoảng từ 64 đến 4096.")

    b_size = cfg.get("rerank_batch_size", 0)
    if not (1 <= b_size <= 64):
        raise ValueError(f"RERANK_BATCH_SIZE ({b_size}) phải nằm trong khoảng từ 1 đến 64.")

    min_score = cfg.get("rerank_min_score", -1.0)
    if not (0.0 <= min_score <= 1.0):
        raise ValueError(f"RERANK_MIN_SCORE ({min_score}) phải nằm trong khoảng [0.0, 1.0].")

    device = cfg.get("rerank_device", "")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"RERANK_DEVICE '{device}' không hợp lệ. Chỉ chấp nhận 'auto', 'cpu', hoặc 'cuda'.")

    cfg["has_api_key"] = bool(cfg.get("api_key"))
    return cfg


def tokenize_vi_legal(text: str) -> List[str]:
    """Tokenizer cho văn bản pháp lý tiếng Việt."""
    if not isinstance(text, str):
        raise TypeError("Input text phải là kiểu string.")
    if not text.strip():
        raise ValueError("Input text không được để rỗng hoặc chỉ chứa khoảng trắng.")

    norm_text = unicodedata.normalize("NFC", text).casefold()
    tokens = re.findall(r"\w+", norm_text)
    clean_tokens = [t for t in tokens if t.strip()]

    if not clean_tokens:
        raise ValueError("Input text sau khi tokenize không tạo ra token hợp lệ nào.")

    return clean_tokens


def bm25_retrieval(query: str, chunks: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
    """Truy xuất Lexical Candidate bằng BM25Okapi."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Câu hỏi (query) không được để rỗng.")

    query_tokens = tokenize_vi_legal(query)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Danh sách chunks không được để rỗng.")

    corpus_tokens: List[List[str]] = []
    for idx, c in enumerate(chunks):
        text = c.get("text", "")
        if not isinstance(text, str) or not text.strip():
            tokens = []
        else:
            try:
                tokens = tokenize_vi_legal(text)
            except ValueError:
                tokens = []
        corpus_tokens.append(tokens)

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    scored_entries = []
    for idx, score in enumerate(scores):
        chunk_copy = dict(chunks[idx])
        chunk_id = str(chunk_copy.get("chunk_id", ""))
        scored_entries.append({
            "chunk": chunk_copy,
            "score": float(score),
            "chunk_id": chunk_id
        })

    scored_entries.sort(key=lambda item: (-item["score"], item["chunk_id"]))

    effective_k = min(max(1, top_k), len(chunks))
    top_candidates: List[Dict[str, Any]] = []

    for rank, entry in enumerate(scored_entries[:effective_k], start=1):
        candidate = entry["chunk"]
        candidate["bm25_rank"] = rank
        candidate["bm25_score"] = entry["score"]
        top_candidates.append(candidate)

    return top_candidates


def check_reranker_cache_exists(model_name: str) -> bool:
    """Kiểm tra xem model Cross-Encoder Reranker đã có trong cache Hugging Face chưa."""
    repo_folder = "models--" + model_name.replace("/", "--")
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    target_dir = os.path.join(hf_home, repo_folder)
    return os.path.isdir(target_dir)


def check_advanced_status(
    strategy: str = "hierarchical",
    input_dir: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Kiểm tra trạng thái hệ thống Advanced RAG (Read-only)."""
    cfg = get_advanced_config(custom_config)
    norm_strat = rag.normalize_strategy(strategy)

    if input_dir is None:
        input_dir = str(rag.DEFAULT_CHUNKS_DIR)

    try:
        chunks, _ = rag.load_chunks(input_path=input_dir, strategy=norm_strat)
        corpus_size = len(chunks)
        bm25_ready = True
    except Exception:
        corpus_size = 0
        bm25_ready = False

    col_name = rag.get_collection_name(
        strategy=norm_strat,
        embedding_model=cfg["embedding_model"],
        embedding_dim=cfg["embedding_dim"]
    )

    storage_dir = rag.STORAGE_DIR
    client = rag.get_chroma_client(storage_dir=storage_dir)

    collection_exists = False
    record_count = 0

    try:
        cols = client.list_collections()
        col_names = [c.name for c in cols]
        if col_name in col_names:
            col = client.get_collection(name=col_name)
            collection_exists = True
            record_count = col.count()
    except Exception:
        pass

    reranker_cache = check_reranker_cache_exists(cfg["reranker_model"])

    return {
        "strategy": norm_strat,
        "corpus_size": corpus_size,
        "semantic_collection_name": col_name,
        "collection_exists": collection_exists,
        "record_count": record_count,
        "embedding_model": cfg["embedding_model"],
        "embedding_dim": cfg["embedding_dim"],
        "bm25_ready": bm25_ready,
        "reranker_model": cfg["reranker_model"],
        "reranker_cache_exists": reranker_cache,
        "has_api_key": cfg["has_api_key"],
        "storage_dir": str(storage_dir)
    }


def prepare_semantic(
    strategy: str = "hierarchical",
    input_dir: Optional[str] = None,
    reset: bool = False,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Tạo embeddings thật và index vào ChromaDB Persistent Collection."""
    cfg = get_advanced_config(custom_config)
    if not cfg["has_api_key"]:
        raise ValueError("GEMINI_API_KEY không được để rỗng trong .env khi chạy prepare-semantic. Không dùng vector giả.")

    if input_dir is None:
        input_dir = str(rag.DEFAULT_CHUNKS_DIR)

    res = rag.index_chunks(input_path=input_dir, strategy=strategy, reset=reset, custom_config=cfg)
    return res


def semantic_retrieval(
    query: str,
    strategy: str = "hierarchical",
    top_k: int = 20,
    custom_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Truy xuất Semantic Candidate stage từ ChromaDB Collection."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Câu hỏi (query) không được để rỗng.")

    cfg = get_advanced_config(custom_config)
    if not cfg["has_api_key"]:
        raise ValueError("GEMINI_API_KEY không được để rỗng trong .env. Không sử dụng vector giả.")

    norm_strat = rag.normalize_strategy(strategy)
    col_name = rag.get_collection_name(
        strategy=norm_strat,
        embedding_model=cfg["embedding_model"],
        embedding_dim=cfg["embedding_dim"]
    )

    storage_dir = rag.STORAGE_DIR
    client = rag.get_chroma_client(storage_dir=storage_dir)

    try:
        cols = [c.name for c in client.list_collections()]
        if col_name not in cols:
            raise ValueError(f"Collection '{col_name}' chưa tồn tại trong storage '{storage_dir}'. Hãy chạy prepare-semantic trước.")
        collection = client.get_collection(name=col_name)
    except Exception as e:
        raise ValueError(f"Lỗi khi truy cập Chroma Collection '{col_name}': {e}")

    rag.verify_collection_metadata(
        collection=collection,
        target_strategy=norm_strat,
        config=cfg
    )

    total_count = collection.count()
    if total_count == 0:
        raise ValueError(f"Collection '{col_name}' không có bản ghi nào. Hãy chạy prepare-semantic trước.")

    query_embeddings = rag.generate_embeddings(
        texts=[query],
        task_type="RETRIEVAL_QUERY",
        config=cfg
    )

    effective_k = min(max(1, top_k), total_count)

    res = collection.query(
        query_embeddings=query_embeddings,
        n_results=effective_k,
        include=["documents", "metadatas", "distances"]
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    candidates: List[Dict[str, Any]] = []
    for rank, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        meta_dict = dict(meta) if meta else {}
        meta_struct_raw = meta_dict.get("metadata_structure", "{}")
        if isinstance(meta_struct_raw, str):
            try:
                meta_struct = json.loads(meta_struct_raw)
            except Exception:
                meta_struct = {}
        else:
            meta_struct = meta_struct_raw

        candidate = {
            "chunk_id": str(doc_id),
            "text": text,
            "source": str(meta_dict.get("source", "")),
            "page_start": int(meta_dict.get("page_start", 1)),
            "page_end": int(meta_dict.get("page_end", 1)),
            "semantic_rank": rank,
            "semantic_distance": float(dist),
            "strategy": norm_strat,
            "metadata_structure": meta_struct
        }
        candidates.append(candidate)

    return candidates


def rrf_fusion(
    bm25_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int = 60,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0
) -> List[Dict[str, Any]]:
    """Hợp nhất danh sách ứng viên từ BM25 và Semantic theo thuật toán RRF."""
    if not isinstance(bm25_results, list) or not isinstance(semantic_results, list):
        raise TypeError("bm25_results và semantic_results phải là danh sách (list).")

    if k <= 0:
        raise ValueError(f"Tham số RRF_K ({k}) phải là số nguyên dương > 0.")

    if bm25_weight < 0.0 or semantic_weight < 0.0:
        raise ValueError("Trọng số RRF (RRF_BM25_WEIGHT, RRF_SEMANTIC_WEIGHT) phải là số thực không âm (>= 0).")

    if bm25_weight == 0.0 and semantic_weight == 0.0:
        raise ValueError("Trọng số RRF không được đồng thời bằng 0.0.")

    merged: Dict[str, Dict[str, Any]] = {}

    for item in bm25_results:
        cid = str(item.get("chunk_id", "")).strip()
        if not cid:
            continue
        text = str(item.get("text", ""))
        source = str(item.get("source", ""))
        p_start = int(item.get("page_start", 1))
        p_end = int(item.get("page_end", 1))
        b_rank = item.get("bm25_rank")
        b_score = item.get("bm25_score")

        merged[cid] = {
            "chunk_id": cid,
            "text": text,
            "source": source,
            "page_start": p_start,
            "page_end": p_end,
            "bm25_rank": int(b_rank) if b_rank is not None else None,
            "bm25_score": float(b_score) if b_score is not None else None,
            "semantic_rank": None,
            "semantic_distance": None,
            "matched_by": ["bm25"],
            "strategy": str(item.get("strategy", "")),
            "metadata_structure": item.get("metadata_structure", {})
        }

    for item in semantic_results:
        cid = str(item.get("chunk_id", "")).strip()
        if not cid:
            continue
        text = str(item.get("text", ""))
        source = str(item.get("source", ""))
        p_start = int(item.get("page_start", 1))
        p_end = int(item.get("page_end", 1))
        s_rank = item.get("semantic_rank")
        s_dist = item.get("semantic_distance")

        if cid in merged:
            existing = merged[cid]
            if existing["text"] != text:
                raise ValueError(f"Lỗi metadata mismatch cho chunk_id '{cid}': text của BM25 và Semantic không khớp.")
            if existing["source"] != source:
                raise ValueError(f"Lỗi metadata mismatch cho chunk_id '{cid}': source '{existing['source']}' vs '{source}' không khớp.")
            if existing["page_start"] != p_start or existing["page_end"] != p_end:
                raise ValueError(f"Lỗi metadata mismatch cho chunk_id '{cid}': phạm vi trang không khớp.")

            existing["semantic_rank"] = int(s_rank) if s_rank is not None else None
            existing["semantic_distance"] = float(s_dist) if s_dist is not None else None
            if "semantic" not in existing["matched_by"]:
                existing["matched_by"].append("semantic")
        else:
            merged[cid] = {
                "chunk_id": cid,
                "text": text,
                "source": source,
                "page_start": p_start,
                "page_end": p_end,
                "bm25_rank": None,
                "bm25_score": None,
                "semantic_rank": int(s_rank) if s_rank is not None else None,
                "semantic_distance": float(s_dist) if s_dist is not None else None,
                "matched_by": ["semantic"],
                "strategy": str(item.get("strategy", "")),
                "metadata_structure": item.get("metadata_structure", {})
            }

    fused_entries = []
    for cid, cand in merged.items():
        score = 0.0
        b_rank = cand["bm25_rank"]
        s_rank = cand["semantic_rank"]

        if b_rank is not None:
            score += bm25_weight / (k + b_rank)
        if s_rank is not None:
            score += semantic_weight / (k + s_rank)

        cand["rrf_score"] = float(score)

        b_val = b_rank if b_rank is not None else float("inf")
        s_val = s_rank if s_rank is not None else float("inf")
        best_rank = min(b_val, s_val)

        cand["_sort_key"] = (-score, best_rank, s_val, b_val, cid)
        fused_entries.append(cand)

    fused_entries.sort(key=lambda item: item["_sort_key"])

    results: List[Dict[str, Any]] = []
    for rank, cand in enumerate(fused_entries, start=1):
        del cand["_sort_key"]
        cand["fused_rank"] = rank
        results.append(cand)

    return results


def hybrid_retrieval(
    question: str,
    strategy: str = "hierarchical",
    input_dir: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pipeline Hybrid Search kết hợp BM25 + Semantic via Reciprocal Rank Fusion (RRF)."""
    t0 = time.perf_counter()

    cfg = get_advanced_config(custom_config)
    norm_strat = rag.normalize_strategy(strategy)

    if input_dir is None:
        input_dir = str(rag.DEFAULT_CHUNKS_DIR)

    t_bm25_start = time.perf_counter()
    chunks, _ = rag.load_chunks(input_path=input_dir, strategy=norm_strat)
    bm25_cands = bm25_retrieval(query=question, chunks=chunks, top_k=cfg["bm25_candidates"])
    t_bm25 = (time.perf_counter() - t_bm25_start) * 1000.0

    t_sem_start = time.perf_counter()
    sem_cands = semantic_retrieval(query=question, strategy=norm_strat, top_k=cfg["semantic_candidates"], custom_config=cfg)
    t_sem = (time.perf_counter() - t_sem_start) * 1000.0

    t_fusion_start = time.perf_counter()
    fused_cands = rrf_fusion(
        bm25_results=bm25_cands,
        semantic_results=sem_cands,
        k=cfg["rrf_k"],
        bm25_weight=cfg["rrf_bm25_weight"],
        semantic_weight=cfg["rrf_semantic_weight"]
    )
    t_fusion = (time.perf_counter() - t_fusion_start) * 1000.0

    t_total = (time.perf_counter() - t0) * 1000.0

    bm25_ids = {c["chunk_id"] for c in bm25_cands}
    sem_ids = {c["chunk_id"] for c in sem_cands}
    overlap_count = len(bm25_ids & sem_ids)
    union_count = len(bm25_ids | sem_ids)

    trace = {
        "bm25_candidate_count": len(bm25_cands),
        "semantic_candidate_count": len(sem_cands),
        "union_count": union_count,
        "overlap_count": overlap_count,
        "fused_count": len(fused_cands),
        "config": {
            "rrf_k": cfg["rrf_k"],
            "rrf_bm25_weight": cfg["rrf_bm25_weight"],
            "rrf_semantic_weight": cfg["rrf_semantic_weight"]
        },
        "latency_ms": {
            "bm25": round(t_bm25, 2),
            "semantic": round(t_sem, 2),
            "fusion": round(t_fusion, 2),
            "total": round(t_total, 2)
        }
    }

    return {
        "candidates": fused_cands,
        "trace": trace
    }


def load_reranker_model(model_name: str = "BAAI/bge-reranker-v2-m3", device_setting: str = "auto") -> Tuple[Any, Any, torch.device]:
    """Lazy-load Tokenizer và Model Cross-Encoder Reranker."""
    global _RERANKER_CACHE

    if (
        _RERANKER_CACHE["tokenizer"] is not None
        and _RERANKER_CACHE["model_name"] == model_name
        and _RERANKER_CACHE["device_setting"] == device_setting
    ):
        return _RERANKER_CACHE["tokenizer"], _RERANKER_CACHE["model"], _RERANKER_CACHE["device"]

    if device_setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("RERANK_DEVICE được cấu hình là 'cuda' nhưng hệ thống không có GPU CUDA khả dụng.")
        device = torch.device("cuda")
    elif device_setting == "cpu":
        device = torch.device("cpu")
    elif device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"RERANK_DEVICE '{device_setting}' không hợp lệ. Chỉ chấp nhận 'auto', 'cpu', hoặc 'cuda'.")

    hf_cache_dir = (BASE_DIR / "storage" / "huggingface").resolve()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RERANKER NOTICE] Đang khởi tạo Reranker Model '{model_name}' trên thiết bị {device}.")
    print(f"[RERANKER NOTICE] Thư mục cache local: '{hf_cache_dir}'")
    print("[RERANKER NOTICE] Model có kích thước tương đối lớn, cần Internet/RAM/dung lượng đĩa nếu chưa có sẵn.")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(hf_cache_dir),
            trust_remote_code=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(hf_cache_dir),
            trust_remote_code=False
        )
        model.to(device)
        model.eval()

        _RERANKER_CACHE["model_name"] = model_name
        _RERANKER_CACHE["device_setting"] = device_setting
        _RERANKER_CACHE["tokenizer"] = tokenizer
        _RERANKER_CACHE["model"] = model
        _RERANKER_CACHE["device"] = device

        return tokenizer, model, device
    except Exception as e:
        raise RuntimeError(f"reranker_unavailable: Lỗi không thể nạp model Reranker '{model_name}': {e}")


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    custom_config: Optional[Dict[str, Any]] = None,
    reranker_fn: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Chấm điểm và xếp hạng lại candidate bằng Cross-Encoder Reranker."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Câu hỏi (query) không được để rỗng.")

    if not isinstance(candidates, list) or not candidates:
        return []

    cfg = get_advanced_config(custom_config)
    t0 = time.perf_counter()

    max_rerank = min(cfg["rerank_candidates"], len(candidates))
    target_candidates = [dict(c) for c in candidates[:max_rerank]]

    if reranker_fn is not None:
        scores_res = reranker_fn(query, [c["text"] for c in target_candidates], cfg)
    else:
        tokenizer, model, device = load_reranker_model(
            model_name=cfg["reranker_model"],
            device_setting=cfg["rerank_device"]
        )
        pairs = [[query, c["text"]] for c in target_candidates]
        batch_size = cfg["rerank_batch_size"]
        scores_res = []

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
                    raw_val = float(logit.item())
                    sig_val = 1.0 / (1.0 + math.exp(-raw_val))
                    scores_res.append((raw_val, sig_val))

    t_lat = (time.perf_counter() - t0) * 1000.0

    for idx, c in enumerate(target_candidates):
        score_item = scores_res[idx]
        if isinstance(score_item, tuple):
            raw_s, sig_s = score_item
        else:
            raw_s = float(score_item)
            sig_s = 1.0 / (1.0 + math.exp(-raw_s))

        c["rerank_raw_score"] = float(raw_s)
        c["rerank_score"] = float(sig_s)
        c["reranker_model"] = cfg["reranker_model"]
        c["rerank_latency_ms"] = round(t_lat, 2)

        if "fused_rank" not in c:
            c["fused_rank"] = idx + 1

        c["_sort_key"] = (-float(sig_s), int(c["fused_rank"]), str(c.get("chunk_id", "")))

    target_candidates.sort(key=lambda item: item["_sort_key"])

    final_k = min(top_k, len(target_candidates))
    results: List[Dict[str, Any]] = []

    for rank, c in enumerate(target_candidates[:final_k], start=1):
        del c["_sort_key"]
        c["rerank_rank"] = rank
        c["rank_change"] = int(c["fused_rank"]) - rank
        results.append(c)

    return results


def build_evidence_schema(cand: Dict[str, Any], accepted: bool) -> Dict[str, Any]:
    """Tạo dict evidence chuẩn schema Buổi 08 với các trường N/A là None."""
    return {
        "chunk_id": str(cand.get("chunk_id", "")),
        "text": str(cand.get("text", "")),
        "source": str(cand.get("source", "")),
        "page_start": int(cand.get("page_start", 1)),
        "page_end": int(cand.get("page_end", 1)),
        "bm25_rank": cand.get("bm25_rank"),
        "bm25_score": cand.get("bm25_score"),
        "semantic_rank": cand.get("semantic_rank"),
        "semantic_distance": cand.get("semantic_distance"),
        "rrf_score": cand.get("rrf_score"),
        "fused_rank": cand.get("fused_rank"),
        "rerank_raw_score": cand.get("rerank_raw_score"),
        "rerank_score": cand.get("rerank_score"),
        "rerank_rank": cand.get("rerank_rank"),
        "rank_change": cand.get("rank_change"),
        "accepted": accepted,
        "metadata_structure": cand.get("metadata_structure", {})
    }


def query_advanced_rag(
    question: str,
    mode: str = "hybrid_rerank",
    strategy: str = "hierarchical",
    input_dir: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    reranker_fn: Optional[Any] = None,
    llm_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Pipeline hỏi đáp Advanced RAG hỗ trợ 4 chế độ: bm25, semantic, hybrid, hybrid_rerank.
    Bao gồm Gating, Generation với Grounding & Citation mapping, và Pipeline Trace.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi (question) không được để rỗng.")

    allowed_modes = {"bm25", "semantic", "hybrid", "hybrid_rerank"}
    if mode not in allowed_modes:
        raise ValueError(f"Mode '{mode}' không hợp lệ. Chỉ chấp nhận một trong các mode {sorted(list(allowed_modes))}.")

    t0 = time.perf_counter()
    cfg = get_advanced_config(custom_config)
    norm_strat = rag.normalize_strategy(strategy)

    if input_dir is None:
        input_dir = str(rag.DEFAULT_CHUNKS_DIR)

    warnings: List[str] = []
    candidates: List[Dict[str, Any]] = []

    t_bm25, t_sem, t_fusion, t_rerank, t_gen = 0.0, 0.0, 0.0, 0.0, 0.0
    bm25_count, sem_count, overlap_count, union_count, reranked_count = 0, 0, 0, 0, 0

    # 1. RETRIEVAL & RERANK BY MODE
    if mode == "bm25":
        t_b_start = time.perf_counter()
        chunks, _ = rag.load_chunks(input_path=input_dir, strategy=norm_strat)
        cands = bm25_retrieval(query=question, chunks=chunks, top_k=cfg["bm25_candidates"])
        t_bm25 = (time.perf_counter() - t_b_start) * 1000.0
        bm25_count = len(cands)
        union_count = len(cands)
        candidates = cands

    elif mode == "semantic":
        t_s_start = time.perf_counter()
        cands = semantic_retrieval(query=question, strategy=norm_strat, top_k=cfg["semantic_candidates"], custom_config=cfg)
        t_sem = (time.perf_counter() - t_s_start) * 1000.0
        sem_count = len(cands)
        union_count = len(cands)
        candidates = cands

    elif mode in ("hybrid", "hybrid_rerank"):
        hyb_res = hybrid_retrieval(question=question, strategy=norm_strat, input_dir=input_dir, custom_config=cfg)
        cands = hyb_res["candidates"]
        tr = hyb_res["trace"]

        bm25_count = tr["bm25_candidate_count"]
        sem_count = tr["semantic_candidate_count"]
        overlap_count = tr["overlap_count"]
        union_count = tr["union_count"]
        t_bm25 = tr["latency_ms"]["bm25"]
        t_sem = tr["latency_ms"]["semantic"]
        t_fusion = tr["latency_ms"]["fusion"]

        if mode == "hybrid":
            candidates = cands[:cfg["final_top_k"]]
        else: # hybrid_rerank
            t_rr_start = time.perf_counter()
            try:
                reranked_cands = rerank_candidates(
                    query=question,
                    candidates=cands,
                    top_k=cfg["final_top_k"],
                    custom_config=cfg,
                    reranker_fn=reranker_fn
                )
                t_rerank = (time.perf_counter() - t_rr_start) * 1000.0
                reranked_count = len(cands[:min(cfg["rerank_candidates"], len(cands))])
                candidates = reranked_cands
            except Exception as e:
                # reranker_unavailable status
                t_total = (time.perf_counter() - t0) * 1000.0
                return {
                    "status": "reranker_unavailable",
                    "mode": mode,
                    "question": question,
                    "answer": "",
                    "evidence": [build_evidence_schema(c, False) for c in cands[:cfg["final_top_k"]]],
                    "citations": [],
                    "warnings": [f"Lỗi nạp hoặc thực thi Cross-Encoder Reranker: {e}"],
                    "trace": {
                        "bm25_candidates": bm25_count,
                        "semantic_candidates": sem_count,
                        "overlap": overlap_count,
                        "union": union_count,
                        "reranked": 0,
                        "accepted": 0,
                        "generation_called": False,
                        "latency_ms": {
                            "bm25": round(t_bm25, 2),
                            "semantic": round(t_sem, 2),
                            "fusion": round(t_fusion, 2),
                            "rerank": round(t_rerank, 2),
                            "generation": 0.0,
                            "total": round(t_total, 2)
                        }
                    }
                }

    # 2. CONFIDENCE GATING BY MODE
    evidence_list: List[Dict[str, Any]] = []
    accepted_evidence: List[Dict[str, Any]] = []

    for c in candidates:
        accepted = False
        if mode == "hybrid_rerank":
            r_score = c.get("rerank_score")
            if r_score is not None and r_score >= cfg["rerank_min_score"]:
                accepted = True
        elif mode == "semantic":
            s_dist = c.get("semantic_distance")
            if s_dist is not None and s_dist <= cfg["rag_max_distance"]:
                accepted = True
        elif mode in ("bm25", "hybrid"):
            # Chẩn đoán mode: Chấp nhận nếu có semantic_distance <= max_distance hoặc nếu trong mode chẩn đoán không có semantic distance thì lấy top 3
            s_dist = c.get("semantic_distance")
            if s_dist is not None:
                if s_dist <= cfg["rag_max_distance"]:
                    accepted = True
            else:
                accepted = True

        ev_obj = build_evidence_schema(c, accepted)
        evidence_list.append(ev_obj)
        if accepted:
            accepted_evidence.append(ev_obj)

    accepted_count = len(accepted_evidence)

    # Nếu không có evidence nào đạt ngưỡng gate -> insufficient_evidence (Không gọi generation)
    if accepted_count == 0:
        t_total = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "insufficient_evidence",
            "mode": mode,
            "question": question,
            "answer": "Không tìm thấy thông tin đủ độ tin cậy trong tài liệu để trả lời câu hỏi.",
            "evidence": evidence_list,
            "citations": [],
            "warnings": warnings,
            "trace": {
                "bm25_candidates": bm25_count,
                "semantic_candidates": sem_count,
                "overlap": overlap_count,
                "union": union_count,
                "reranked": reranked_count,
                "accepted": 0,
                "generation_called": False,
                "latency_ms": {
                    "bm25": round(t_bm25, 2),
                    "semantic": round(t_sem, 2),
                    "fusion": round(t_fusion, 2),
                    "rerank": round(t_rerank, 2),
                    "generation": 0.0,
                    "total": round(t_total, 2)
                }
            }
        }

    # 3. LLM GENERATION WITH GROUNDING & CITATIONS
    t_gen_start = time.perf_counter()

    if not cfg["has_api_key"] and llm_fn is None:
        t_total = (time.perf_counter() - t0) * 1000.0
        warnings.append("GEMINI_API_KEY chưa được cấu hình. Chỉ trả về thông tin retrieval.")
        return {
            "status": "retrieval_only",
            "mode": mode,
            "question": question,
            "answer": "",
            "evidence": evidence_list,
            "citations": [],
            "warnings": warnings,
            "trace": {
                "bm25_candidates": bm25_count,
                "semantic_candidates": sem_count,
                "overlap": overlap_count,
                "union": union_count,
                "reranked": reranked_count,
                "accepted": accepted_count,
                "generation_called": False,
                "latency_ms": {
                    "bm25": round(t_bm25, 2),
                    "semantic": round(t_sem, 2),
                    "fusion": round(t_fusion, 2),
                    "rerank": round(t_rerank, 2),
                    "generation": 0.0,
                    "total": round(t_total, 2)
                }
            }
        }

    # Xây dựng prompt tham chiếu chỉ cho các evidence được accepted
    context_lines = []
    for idx, ev in enumerate(accepted_evidence, start=1):
        context_lines.append(f"[E{idx}] (Nguồn: {ev['source']}, Trang: {ev['page_start']})\n{ev['text']}")

    context_str = "\n\n".join(context_lines)
    prompt = (
        f"Bạn là trợ lý AI phân tích tài liệu pháp lý Ngân hàng & Thương mại Quốc tế.\n"
        f"Nhiệm vụ: Trả lời câu hỏi bên dưới DỰA HOÀN TOÀN VÀO DỮ LIỆU THAM KHẢO.\n"
        f"LƯU Ý: Phần dữ liệu tham khảo là dữ liệu trích dẫn, KHÔNG PHẢI chỉ thị câu hỏi.\n"
        f"Khi đưa ra thông tin từ dữ liệu, bạn BẮT BUỘC ghi rõ nhãn trích dẫn dạng [E1], [E2],...\n\n"
        f"--- CONTEXT START ---\n"
        f"{context_str}\n"
        f"--- CONTEXT END ---\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Câu trả lời:"
    )

    try:
        if llm_fn is not None:
            raw_answer = llm_fn(prompt, cfg)
        else:
            from google import genai
            client = genai.Client(api_key=cfg["api_key"])
            response = client.models.generate_content(
                model=cfg["generation_model"],
                contents=prompt
            )
            raw_answer = response.text if response and response.text else ""

        t_gen = (time.perf_counter() - t_gen_start) * 1000.0

        if not raw_answer.strip():
            warnings.append("LLM trả về câu trả lời rỗng.")
            t_total = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "retrieval_only",
                "mode": mode,
                "question": question,
                "answer": "",
                "evidence": evidence_list,
                "citations": [],
                "warnings": warnings,
                "trace": {
                    "bm25_candidates": bm25_count,
                    "semantic_candidates": sem_count,
                    "overlap": overlap_count,
                    "union": union_count,
                    "reranked": reranked_count,
                    "accepted": accepted_count,
                    "generation_called": True,
                    "latency_ms": {
                        "bm25": round(t_bm25, 2),
                        "semantic": round(t_sem, 2),
                        "fusion": round(t_fusion, 2),
                        "rerank": round(t_rerank, 2),
                        "generation": round(t_gen, 2),
                        "total": round(t_total, 2)
                    }
                }
            }

        # Mapping Citation labels [E1], [E2] sang metadata thật
        found_labels = re.findall(r"\[E(\d+)\]", raw_answer)
        citations: List[Dict[str, Any]] = []
        valid_indices = set()

        for lbl_num_str in sorted(list(set(found_labels)), key=lambda x: int(x)):
            idx_val = int(lbl_num_str)
            if 1 <= idx_val <= len(accepted_evidence):
                valid_indices.add(idx_val)
                target_ev = accepted_evidence[idx_val - 1]
                p_str = f"tr.{target_ev['page_start']}" if target_ev['page_start'] == target_ev['page_end'] else f"tr.{target_ev['page_start']}-{target_ev['page_end']}"
                citations.append({
                    "evidence_id": f"[E{idx_val}]",
                    "chunk_id": target_ev["chunk_id"],
                    "source": target_ev["source"],
                    "page_start": target_ev["page_start"],
                    "page_end": target_ev["page_end"],
                    "display": f"{target_ev['source']}, {p_str}, chunk: {target_ev['chunk_id']}"
                })
            else:
                warnings.append(f"LLM tạo nhãn trích dẫn giả không tồn tại [E{idx_val}]. Đã tự động loại bỏ.")

        t_total = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "answered",
            "mode": mode,
            "question": question,
            "answer": raw_answer.strip(),
            "evidence": evidence_list,
            "citations": citations,
            "warnings": warnings,
            "trace": {
                "bm25_candidates": bm25_count,
                "semantic_candidates": sem_count,
                "overlap": overlap_count,
                "union": union_count,
                "reranked": reranked_count,
                "accepted": accepted_count,
                "generation_called": True,
                "latency_ms": {
                    "bm25": round(t_bm25, 2),
                    "semantic": round(t_sem, 2),
                    "fusion": round(t_fusion, 2),
                    "rerank": round(t_rerank, 2),
                    "generation": round(t_gen, 2),
                    "total": round(t_total, 2)
                }
            }
        }

    except Exception as e:
        t_gen = (time.perf_counter() - t_gen_start) * 1000.0
        t_total = (time.perf_counter() - t0) * 1000.0
        warnings.append(f"Lỗi gọi Gemini LLM Generation: {e}")
        return {
            "status": "retrieval_only",
            "mode": mode,
            "question": question,
            "answer": "",
            "evidence": evidence_list,
            "citations": [],
            "warnings": warnings,
            "trace": {
                "bm25_candidates": bm25_count,
                "semantic_candidates": sem_count,
                "overlap": overlap_count,
                "union": union_count,
                "reranked": reranked_count,
                "accepted": accepted_count,
                "generation_called": True,
                "latency_ms": {
                    "bm25": round(t_bm25, 2),
                    "semantic": round(t_sem, 2),
                    "fusion": round(t_fusion, 2),
                    "rerank": round(t_rerank, 2),
                    "generation": round(t_gen, 2),
                    "total": round(t_total, 2)
                }
            }
        }


def compare_retrieval_modes(
    question: str,
    strategy: str = "hierarchical",
    input_dir: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    reranker_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    So sánh kết quả của cả 4 chế độ retrieval (bm25, semantic, hybrid, hybrid_rerank) trên cùng một câu hỏi.
    ĐẢM BẢO KHÔNG GỌI LLM GENERATION BẤT KỲ LẦN NÀO.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Câu hỏi (question) không được để rỗng.")

    cfg = get_advanced_config(custom_config)
    norm_strat = rag.normalize_strategy(strategy)

    if input_dir is None:
        input_dir = str(rag.DEFAULT_CHUNKS_DIR)

    mode_results: Dict[str, Any] = {}
    mode_latencies: Dict[str, float] = {}
    all_chunks: Dict[str, Dict[str, Any]] = {}

    # Mode 1: BM25
    t0 = time.perf_counter()
    chunks, _ = rag.load_chunks(input_path=input_dir, strategy=norm_strat)
    bm25_cands = bm25_retrieval(query=question, chunks=chunks, top_k=cfg["bm25_candidates"])
    t_bm25 = (time.perf_counter() - t0) * 1000.0
    mode_results["bm25"] = bm25_cands
    mode_latencies["bm25"] = round(t_bm25, 2)

    # Mode 2: Semantic
    t0 = time.perf_counter()
    try:
        sem_cands = semantic_retrieval(query=question, strategy=norm_strat, top_k=cfg["semantic_candidates"], custom_config=cfg)
        t_sem = (time.perf_counter() - t0) * 1000.0
    except Exception:
        sem_cands = []
        t_sem = 0.0
    mode_results["semantic"] = sem_cands
    mode_latencies["semantic"] = round(t_sem, 2)

    # Mode 3: Hybrid
    t0 = time.perf_counter()
    fused_cands = rrf_fusion(
        bm25_results=bm25_cands,
        semantic_results=sem_cands,
        k=cfg["rrf_k"],
        bm25_weight=cfg["rrf_bm25_weight"],
        semantic_weight=cfg["rrf_semantic_weight"]
    )
    t_hyb = (time.perf_counter() - t0) * 1000.0
    mode_results["hybrid"] = fused_cands[:cfg["final_top_k"]]
    mode_latencies["hybrid"] = round(t_hyb, 2)

    # Mode 4: Hybrid Rerank
    t0 = time.perf_counter()
    try:
        reranked_cands = rerank_candidates(
            query=question,
            candidates=fused_cands,
            top_k=cfg["final_top_k"],
            custom_config=cfg,
            reranker_fn=reranker_fn
        )
        t_rr = (time.perf_counter() - t0) * 1000.0
    except Exception:
        reranked_cands = []
        t_rr = 0.0
    mode_results["hybrid_rerank"] = reranked_cands
    mode_latencies["hybrid_rerank"] = round(t_rr, 2)

    # Xây dựng bảng so sánh tổng hợp các chunk xuất hiện
    for mode_name, c_list in mode_results.items():
        for rank, c in enumerate(c_list, start=1):
            cid = c["chunk_id"]
            if cid not in all_chunks:
                all_chunks[cid] = {
                    "chunk_id": cid,
                    "source": c["source"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "ranks": {"bm25": None, "semantic": None, "hybrid": None, "hybrid_rerank": None},
                    "text": c["text"]
                }
            all_chunks[cid]["ranks"][mode_name] = rank

    comparison_table = list(all_chunks.values())

    return {
        "question": question,
        "strategy": norm_strat,
        "mode_results": mode_results,
        "mode_latencies": mode_latencies,
        "comparison_table": comparison_table
    }


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline CLI — Buổi 08")
    subparsers = parser.add_subparsers(dest="command", help="Lệnh thực thi")

    # Command status
    st_parser = subparsers.add_parser("status", help="Kiểm tra trạng thái hệ thống và Chroma DB")
    st_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    st_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")

    # Command prepare-semantic
    prep_parser = subparsers.add_parser("prepare-semantic", help="Tạo embeddings và index vào ChromaDB")
    prep_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    prep_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")
    prep_parser.add_argument("--reset", action="store_true", help="Xóa và index lại collection")

    # Command bm25
    bm25_parser = subparsers.add_parser("bm25", help="Truy xuất Lexical Search BM25")
    bm25_parser.add_argument("--question", type=str, required=True, help="Nội dung câu hỏi")
    bm25_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    bm25_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")
    bm25_parser.add_argument("--top-k", type=int, default=20, help="Số lượng candidates tối đa")

    # Command semantic
    sem_parser = subparsers.add_parser("semantic", help="Truy xuất Semantic Search bằng Gemini Embeddings")
    sem_parser.add_argument("--question", type=str, required=True, help="Nội dung câu hỏi")
    sem_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    sem_parser.add_argument("--top-k", type=int, default=20, help="Số lượng candidates tối đa")

    # Command hybrid
    hyb_parser = subparsers.add_parser("hybrid", help="Truy xuất Hybrid Search kết hợp BM25 + Semantic qua RRF")
    hyb_parser.add_argument("--question", type=str, required=True, help="Nội dung câu hỏi")
    hyb_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    hyb_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")

    # Command rerank
    rr_parser = subparsers.add_parser("rerank", help="Truy xuất Hybrid RRF + Cross-Encoder Reranker")
    rr_parser.add_argument("--question", type=str, required=True, help="Nội dung câu hỏi")
    rr_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    rr_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")
    rr_parser.add_argument("--top-k", type=int, default=5, help="Số lượng final candidates tối đa")

    # Command query
    q_parser = subparsers.add_parser("query", help="Thực hiện hỏi đáp RAG với Pipeline hoàn chỉnh")
    q_parser.add_argument("--question", type=str, required=True, help="Nội dung câu hỏi")
    q_parser.add_argument("--mode", type=str, default="hybrid_rerank", help="Chế độ (bm25, semantic, hybrid, hybrid_rerank)")
    q_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    q_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")

    # Command compare
    cmp_parser = subparsers.add_parser("compare", help="So sánh cả 4 chế độ retrieval trên cùng câu hỏi (Không gọi LLM)")
    cmp_parser.add_argument("--question", type=str, required=True, help="Nội dung câu hỏi")
    cmp_parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking")
    cmp_parser.add_argument("--input-dir", type=str, default=str(rag.DEFAULT_CHUNKS_DIR), help="Thư mục chunks")

    args = parser.parse_args()

    if args.command == "status":
        try:
            st_res = check_advanced_status(strategy=args.strategy, input_dir=args.input_dir)
            print("\n=== TRẠNG THÁI HỆ THỐNG ADVANCED RAG (BUỔI 08) ===")
            print(f"Chiến lược (Strategy):       {st_res['strategy']}")
            print(f"Tổng chunks trong corpus:     {st_res['corpus_size']}")
            print(f"BM25 Index sẵn sàng:          {'Có' if st_res['bm25_ready'] else 'Chưa'}")
            print(f"Chroma Collection Name:      {st_res['semantic_collection_name']}")
            print(f"Collection Tồn tại:          {'Có' if st_res['collection_exists'] else 'Chưa'}")
            print(f"Số record trong Collection:  {st_res['record_count']}")
            print(f"Embedding Model:             {st_res['embedding_model']} (dim={st_res['embedding_dim']})")
            print(f"Reranker Model:              {st_res['reranker_model']}")
            print(f"Reranker Cache Tồn tại:      {'Có' if st_res['reranker_cache_exists'] else 'Chưa'}")
            print(f"API Key Gemini (.env):       {'Đã cấu hình' if st_res['has_api_key'] else 'Thiếu'}")
            print(f"Storage Path:                {st_res['storage_dir']}")

        except Exception as e:
            print(f"\n[LỖI STATUS] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "prepare-semantic":
        try:
            res = prepare_semantic(strategy=args.strategy, input_dir=args.input_dir, reset=args.reset)
            print("\n=== KẾT QUẢ PREPARE SEMANTIC CHROMA INDEX ===")
            print(f"Collection Name:       {res['collection_name']}")
            print(f"Số chunk vừa index:    {res['indexed_chunks']}")
            print(f"Tổng record collection:{res['total_collection_records']}")
            print(f"Reset Collection:      {'Có' if res['reset_performed'] else 'Không'}")
        except Exception as e:
            print(f"\n[LỖI PREPARE-SEMANTIC] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "bm25":
        try:
            chunks, stats = rag.load_chunks(input_path=args.input_dir, strategy=args.strategy)
            print(f"\n=== BM25 LEXICAL RETRIEVAL RESULTS ===")
            print(f"Câu hỏi:      \"{args.question}\"")
            print(f"Chiến lược:   {args.strategy}")
            print(f"Tổng chunks:  {len(chunks)} (đã đọc {stats['files_read']} files)")
            print(f"Top K:        {args.top_k}\n")

            results = bm25_retrieval(query=args.question, chunks=chunks, top_k=args.top_k)

            print(f"{'RANK':<5} | {'SCORE':<8} | {'SOURCE':<35} | {'PAGE':<8} | {'CHUNK ID':<15} | PREVIEW")
            print("-" * 110)
            for res in results:
                p_str = f"tr.{res['page_start']}" if res['page_start'] == res['page_end'] else f"tr.{res['page_start']}-{res['page_end']}"
                preview = res['text'][:60].replace("\n", " ") + ("..." if len(res['text']) > 60 else "")
                print(f"{res['bm25_rank']:<5} | {res['bm25_score']:<8.4f} | {res['source']:<35} | {p_str:<8} | {res['chunk_id']:<15} | \"{preview}\"")

        except Exception as e:
            print(f"\n[LỖI BM25 RETRIEVAL] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "semantic":
        try:
            results = semantic_retrieval(query=args.question, strategy=args.strategy, top_k=args.top_k)
            print(f"\n=== SEMANTIC CANDIDATE RETRIEVAL RESULTS ===")
            print(f"Câu hỏi:      \"{args.question}\"")
            print(f"Chiến lược:   {args.strategy}")
            print(f"Top K:        {args.top_k}\n")

            print(f"{'RANK':<5} | {'DISTANCE':<8} | {'SOURCE':<35} | {'PAGE':<8} | {'CHUNK ID':<15} | PREVIEW")
            print("-" * 110)
            for res in results:
                p_str = f"tr.{res['page_start']}" if res['page_start'] == res['page_end'] else f"tr.{res['page_start']}-{res['page_end']}"
                preview = res['text'][:60].replace("\n", " ") + ("..." if len(res['text']) > 60 else "")
                print(f"{res['semantic_rank']:<5} | {res['semantic_distance']:<8.4f} | {res['source']:<35} | {p_str:<8} | {res['chunk_id']:<15} | \"{preview}\"")

        except Exception as e:
            print(f"\n[LỖI SEMANTIC RETRIEVAL] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "hybrid":
        try:
            res = hybrid_retrieval(question=args.question, strategy=args.strategy, input_dir=args.input_dir)
            cands = res["candidates"]
            tr = res["trace"]

            print(f"\n=== HYBRID SEARCH RESULTS (BM25 + SEMANTIC VIA RRF) ===")
            print(f"Câu hỏi:      \"{args.question}\"")
            print(f"Chiến lược:   {args.strategy}")
            print(f"Trace Count:  BM25={tr['bm25_candidate_count']}, Semantic={tr['semantic_candidate_count']}, Overlap={tr['overlap_count']}, Union={tr['union_count']}")
            print(f"RRF Config:   k={tr['config']['rrf_k']}, w_bm25={tr['config']['rrf_bm25_weight']}, w_sem={tr['config']['rrf_semantic_weight']}")
            print(f"Latency:      BM25={tr['latency_ms']['bm25']}ms, Semantic={tr['latency_ms']['semantic']}ms, Fusion={tr['latency_ms']['fusion']}ms, Total={tr['latency_ms']['total']}ms\n")

            print(f"{'FUSED':<5} | {'RRF SCORE':<10} | {'MATCHED BY':<15} | {'BM25 RANK/SCORE':<17} | {'SEM RANK/DIST':<17} | {'CHUNK ID':<12} | PREVIEW")
            print("-" * 120)
            for c in cands:
                b_str = f"#{c['bm25_rank']} ({c['bm25_score']:.2f})" if c['bm25_rank'] is not None else "-"
                s_str = f"#{c['semantic_rank']} ({c['semantic_distance']:.4f})" if c['semantic_rank'] is not None else "-"
                m_str = "+".join(c['matched_by'])
                preview = c['text'][:40].replace("\n", " ") + ("..." if len(c['text']) > 40 else "")
                print(f"{c['fused_rank']:<5} | {c['rrf_score']:<10.5f} | {m_str:<15} | {b_str:<17} | {s_str:<17} | {c['chunk_id']:<12} | \"{preview}\"")

        except Exception as e:
            print(f"\n[LỖI HYBRID RETRIEVAL] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "rerank":
        try:
            hyb_res = hybrid_retrieval(question=args.question, strategy=args.strategy, input_dir=args.input_dir)
            fused_cands = hyb_res["candidates"]

            reranked_cands = rerank_candidates(query=args.question, candidates=fused_cands, top_k=args.top_k)

            print(f"\n=== CROSS-ENCODER RERANKER RESULTS ===")
            print(f"Câu hỏi:      \"{args.question}\"")
            print(f"Chiến lược:   {args.strategy}")
            print(f"Model:        {get_advanced_config()['reranker_model']}")
            print(f"Final Top K:  {len(reranked_cands)}\n")

            print(f"{'RERANK':<6} | {'SIGMOID SCORE':<13} | {'RAW LOGIT':<10} | {'FUSED RANK':<10} | {'CHANGE':<8} | {'CHUNK ID':<12} | PREVIEW")
            print("-" * 120)
            for c in reranked_cands:
                chg_str = f"+{c['rank_change']}" if c['rank_change'] > 0 else str(c['rank_change'])
                preview = c['text'][:40].replace("\n", " ") + ("..." if len(c['text']) > 40 else "")
                print(f"{c['rerank_rank']:<6} | {c['rerank_score']:<13.5f} | {c['rerank_raw_score']:<10.4f} | #{c['fused_rank']:<9} | {chg_str:<8} | {c['chunk_id']:<12} | \"{preview}\"")

        except Exception as e:
            print(f"\n[LỖI RERANK RETRIEVAL] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "query":
        try:
            res = query_advanced_rag(
                question=args.question,
                mode=args.mode,
                strategy=args.strategy,
                input_dir=args.input_dir
            )
            print("\n=== KẾT QUẢ HỎI ĐÁP ADVANCED RAG PIPELINE ===")
            print(f"Trạng thái (Status): {res['status']}")
            print(f"Chế độ (Mode):       {res['mode']}")
            print(f"Câu hỏi:             \"{res['question']}\"")
            print(f"\n--- CÂU TRẢ LỜI ---")
            print(res['answer'] if res['answer'] else "(Không có câu trả lời từ LLM)")

            if res['citations']:
                print(f"\n--- DANH SÁCH TRÍCH DẪN ({len(res['citations'])}) ---")
                for c in res['citations']:
                    print(f"  • {c['evidence_id']}: {c['display']}")

            print(f"\n--- DANH SÁCH EVIDENCE TRUY XUẤT ({len(res['evidence'])}) ---")
            for e in res['evidence']:
                acc_str = "[ĐẠT NGƯỠNG]" if e['accepted'] else "[BỊ LOẠI]"
                r_info = f"RerankScore: {e['rerank_score']:.4f}" if e['rerank_score'] is not None else (f"Dist: {e['semantic_distance']:.4f}" if e['semantic_distance'] is not None else f"BM25Score: {e['bm25_score']}")
                print(f"  • {acc_str} {r_info} | {e['source']} (tr.{e['page_start']}) | Chunk: {e['chunk_id']}")

            if res['warnings']:
                print(f"\n--- CẢNH BÁO / WARNINGS ({len(res['warnings'])}) ---")
                for w in res['warnings']:
                    print(f"  ⚠️  {w}")

            tr = res['trace']
            print(f"\n--- PIPELINE TRACE & LATENCY ---")
            print(f"  Candidates: BM25={tr['bm25_candidates']}, Semantic={tr['semantic_candidates']}, Overlap={tr['overlap']}, Union={tr['union']}, Reranked={tr['reranked']}, Accepted={tr['accepted']}")
            print(f"  Latency: BM25={tr['latency_ms']['bm25']}ms, Sem={tr['latency_ms']['semantic']}ms, Fusion={tr['latency_ms']['fusion']}ms, Rerank={tr['latency_ms']['rerank']}ms, Gen={tr['latency_ms']['generation']}ms | Total={tr['latency_ms']['total']}ms")

        except Exception as e:
            print(f"\n[LỖI QUERY ADVANCED RAG] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "compare":
        try:
            cmp_res = compare_retrieval_modes(
                question=args.question,
                strategy=args.strategy,
                input_dir=args.input_dir
            )
            print("\n=== BẢNG SO SÁNH CÁC CHẾ ĐỘ RETRIEVAL (KHÔNG GỌI LLM) ===")
            print(f"Câu hỏi:   \"{cmp_res['question']}\"")
            print(f"Chiến lược: {cmp_res['strategy']}")
            print(f"Latency:   BM25={cmp_res['mode_latencies']['bm25']}ms, Semantic={cmp_res['mode_latencies']['semantic']}ms, Hybrid={cmp_res['mode_latencies']['hybrid']}ms, Hybrid_Rerank={cmp_res['mode_latencies']['hybrid_rerank']}ms\n")

            print(f"{'CHUNK ID':<12} | {'BM25 RANK':<10} | {'SEM RANK':<10} | {'HYBRID RANK':<12} | {'RERANK RANK':<12} | {'SOURCE':<35} | PREVIEW")
            print("-" * 120)
            for row in cmp_res["comparison_table"]:
                b_r = f"#{row['ranks']['bm25']}" if row['ranks']['bm25'] is not None else "-"
                s_r = f"#{row['ranks']['semantic']}" if row['ranks']['semantic'] is not None else "-"
                h_r = f"#{row['ranks']['hybrid']}" if row['ranks']['hybrid'] is not None else "-"
                rr_r = f"#{row['ranks']['hybrid_rerank']}" if row['ranks']['hybrid_rerank'] is not None else "-"
                preview = row['text'][:40].replace("\n", " ") + ("..." if len(row['text']) > 40 else "")
                print(f"{row['chunk_id']:<12} | {b_r:<10} | {s_r:<10} | {h_r:<12} | {rr_r:<12} | {row['source']:<35} | \"{preview}\"")

        except Exception as e:
            print(f"\n[LỖI COMPARE RETRIEVAL MODES] {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
