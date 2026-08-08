"""
UI Helper functions cho Streamlit Dashboard Buổi 09.
Cung cấp các hàm biến đổi dữ liệu thuần Python cho Ma trận Query-Child, Cây Parent-Child,
Bảng so sánh 4 Mode và Định dạng Trích dẫn/Cảnh báo.
"""

import math
from typing import Any, Dict, List, Tuple
import pandas as pd


def map_status_warning_badge(status: str) -> Tuple[str, str, str]:
    """
    Ánh xạ status của pipeline sang Emoji badge, thông điệp tiếng Việt và hướng xử lý.
    """
    status_map = {
        "ready": ("🟢 THÀNH CÔNG", "Pipeline hoàn tất xử lý thành công.", "success"),
        "multi_query_partial": ("🟡 BÁN PHẦN (PARTIAL)", "Một số câu hỏi biến thể bị lỗi nhưng pipeline vẫn tiếp tục với các query còn lại.", "warning"),
        "hierarchy_not_ready": ("🔴 HIERARCHY CHƯA SẴN SÀNG", "Hierarchy Store chưa được xây dựng.", "error"),
        "collection_not_ready": ("🔴 VECTOR DB CHƯA SẴN SÀNG", "ChromaDB Collection chưa được tạo index.", "error"),
        "query_generation_unavailable": ("🟠 KHÔNG THỂ SINH MULTI-QUERY", "Lỗi hoặc thiếu GEMINI_API_KEY để gọi Multi-Query Expansion.", "warning"),
        "reranker_unavailable": ("🔴 RERANKER CHƯA KHỞI TẠO", "Lỗi không thể nạp Cross-Encoder Reranker Model.", "error"),
        "insufficient_evidence": ("⚪ KHÔNG ĐỦ CHỨNG CỨ", "Không có candidate nào vượt qua ngưỡng RERANK_MIN_SCORE.", "info"),
        "generation_error": ("🔴 LỖI GEMINI GENERATION", "Không thể gọi Gemini API để sinh câu trả lời.", "error")
    }
    return status_map.get(status, (f"⚪ {status.upper()}", f"Trạng thái: {status}", "info"))


def build_query_child_matrix(query_set: Dict[str, Any], child_hits: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Tạo Ma trận Query–Child:
    - Hàng (Index): child_id
    - Cột: Q0, Q1... + Support Query Count + MQ-RRF Score
    - Giá trị ô: thứ hạng inner_rrf_rank trong query đó hoặc '—'
    """
    queries = query_set.get("queries", [])
    q_ids = [q["query_id"] for q in queries]

    rows = []
    for hit in child_hits:
        cid = hit["child_id"]
        row_dict = {"Child ID": cid}

        pq_ranks = hit.get("per_query_ranks", {})
        for qid in q_ids:
            if qid in pq_ranks:
                row_dict[qid] = f"Rank {pq_ranks[qid]}"
            else:
                row_dict[qid] = "—"

        row_dict["Support Queries"] = hit.get("support_query_count", len(hit.get("support_query_ids", [])))
        row_dict["MQ-RRF Score"] = f"{hit.get('multi_query_rrf_score', 0.0):.6f}"
        rows.append(row_dict)

    if not rows:
        cols = ["Child ID"] + q_ids + ["Support Queries", "MQ-RRF Score"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    return df


def build_parent_tree_data(parent_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cấu trúc dữ liệu cây Parent Document -> Supporting Child Chunks cho Tab Parent-Child Explorer.
    """
    tree_nodes = []
    for p in parent_candidates:
        r_rrf = p.get("parent_rank", 0)
        r_rr = p.get("parent_rerank_rank", r_rrf)
        delta = p.get("parent_rank_change", 0)

        badge_movement = f"Rank {r_rrf} ➔ Rank {r_rr}"
        if delta > 0:
            badge_movement += f" (▲+{delta})"
        elif delta < 0:
            badge_movement += f" (▼{delta})"
        else:
            badge_movement += " (➖ 0)"

        node = {
            "parent_id": p["parent_id"],
            "heading": p.get("heading", "N/A"),
            "source": p["source"],
            "pages": f"Trang {p['page_start']}-{p['page_end']}",
            "structural_path": p.get("structural_path", {}),
            "parent_rrf_score": p.get("parent_rrf_score", 0.0),
            "parent_rerank_score": p.get("parent_rerank_score", 0.0),
            "raw_logit": p.get("parent_rerank_raw_score", 0.0),
            "rank_movement_badge": badge_movement,
            "anchor_child_id": p.get("anchor_child_id", ""),
            "scoring_child_ids": p.get("scoring_child_ids", []),
            "supporting_child_ids": p.get("supporting_child_ids", []),
            "support_query_ids": p.get("support_query_ids", []),
            "char_count": p.get("char_count", len(p.get("text", ""))),
            "text": p.get("text", ""),
            "ambiguous": p.get("ambiguous", False),
            "warnings": p.get("warnings", [])
        }
        tree_nodes.append(node)

    return tree_nodes


def build_mode_comparison_row(mode: str, res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tạo dòng dữ liệu phẳng để đưa vào bảng So sánh 4 Mode (Mode Comparison).
    """
    status = res.get("status", "unknown")
    child_hits = res.get("child_hits", [])
    parent_cands = res.get("parent_candidates", [])
    accepted = res.get("accepted_evidence", [])

    is_parent = "parent" in mode
    unit_type = "Parent Document" if is_parent else "Child Chunk"

    sources_set = set()
    for ev in accepted:
        sources_set.add(ev.get("source", "N/A"))

    tr = res.get("trace", {})
    stg_lat = tr.get("stage_latencies_ms", {})
    tot_lat = stg_lat.get("total", round(sum(stg_lat.values()), 2))

    api_counts = tr.get("api_call_counts", {"generation_calls": 0, "embedding_calls": 0})
    gen_calls = api_counts.get("generation_calls", 0)
    emb_calls = api_counts.get("embedding_calls", 0)

    # Calculate expansion factor & total context chars
    ctx_chars = sum(p.get("char_count", len(p.get("text", ""))) for p in accepted)
    child_chars = sum(len(c.get("text", "")) for c in child_hits)
    exp_factor = round(ctx_chars / child_chars, 2) if child_chars > 0 else 1.0

    ev_ids = [cit["evidence_id"] for cit in res.get("citations", [])]

    return {
        "Mode": mode,
        "Status": status.upper(),
        "Unit Type": unit_type,
        "Retrieved Child Hits": len(child_hits),
        "Candidates": len(parent_cands) if is_parent else len(child_hits),
        "Accepted Evidence": len(accepted),
        "Evidence IDs": ", ".join(ev_ids) if ev_ids else "—",
        "Unique Sources": len(sources_set),
        "Context Chars": f"{ctx_chars:,}",
        "Expansion Factor": f"x{exp_factor:.2f}" if is_parent else "x1.00",
        "Total Latency (ms)": tot_lat,
        "Gen API Calls": gen_calls,
        "Emb API Calls": emb_calls
    }


def format_citation_display(citations: List[Dict[str, Any]]) -> str:
    """
    Định dạng danh sách Trích dẫn (Citations) sang Markdown hiển thị đẹp mắt.
    """
    if not citations:
        return "*Không có trích dẫn nào.*"

    lines = []
    for cit in citations:
        eid = cit["evidence_id"]
        source = cit.get("source", "N/A")
        pages = f"Trang {cit.get('page_start', 1)}-{cit.get('page_end', 1)}"

        if "parent_id" in cit:
            pid = cit["parent_id"]
            anchor = cit.get("anchor_child_id", "N/A")
            score = cit.get("parent_rerank_score", 0.0)
            amb_badge = " ⚠️ Ambiguous" if cit.get("ambiguous") else ""
            lines.append(f"- **[{eid}]** Parent Document: `{pid}` (Anchor Child: `{anchor}`){amb_badge}  \n  *Nguồn:* {source} ({pages}) | *Rerank Score:* `{score:.4f}`")
        else:
            cid = cit.get("chunk_id", "N/A")
            score = cit.get("rerank_score", 0.0)
            lines.append(f"- **[{eid}]** Child Chunk: `{cid}`  \n  *Nguồn:* {source} ({pages}) | *Rerank Score:* `{score:.4f}`")

    return "\n\n".join(lines)
