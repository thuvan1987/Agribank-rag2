"""
Buổi 09: Streamlit Comparison Dashboard cho Multi-Query & Parent-Child RAG.

Giao diện tương tác trực quan hỗ trợ:
1. Hỏi đáp RAG Hoàn chỉnh (Ask Advanced RAG).
2. Trực quan hóa Mở rộng câu hỏi đòn bẩy (Query Fan-out) & Ma trận Query–Child.
3. Khám phá cây Phân cấp Cấu trúc Luật Ngân hàng (Parent–Child Explorer).
4. So sánh độc lập 4 chế độ Pipeline (Mode Comparison — Retrieval Only).
5. Báo cáo định lượng chất lượng truy xuất (Evaluation Benchmark).
"""

import json
from pathlib import Path
import sys
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import advanced_rag
import hierarchical_rag
import rag
from ui_helpers import (
    build_mode_comparison_row,
    build_parent_tree_data,
    build_query_child_matrix,
    format_citation_display,
    map_status_warning_badge,
)

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval",
    page_icon="🌳",
    layout="wide"
)

# Title & Subtitle Pipeline
st.title("🌳 RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval")
st.caption("🚀 **Pipeline**: Query fan-out ➔ Hybrid per query ➔ Cross-query RRF ➔ Parent expansion ➔ Parent rerank")

# 2. Sidebar — Runtime Configurations & System Status
with st.sidebar:
    st.header("⚙️ Cấu Hình Pipeline (Runtime)")

    selected_mode = st.selectbox(
        "Chế độ RAG (Mode):",
        options=["multi_parent", "single_parent", "multi_flat", "single_flat"],
        index=0,
        help="multi_parent: Q0 + biến thể -> Hybrid -> MQ-RRF -> Parent Expansion -> Parent Rerank."
    )

    st.subheader("🎛️ Tham Số Truy Xuất")
    mq_count = st.slider("Số câu hỏi biến thể (MULTI_QUERY_COUNT):", min_value=1, max_value=5, value=3)
    per_query_cands = st.number_input("Per-Query Child Hits (PER_QUERY_CANDIDATES):", min_value=5, max_value=50, value=20)
    parent_cands = st.number_input("Parent Candidates Rerank (PARENT_CANDIDATES):", min_value=3, max_value=30, value=10)
    final_top_k = st.number_input("Top Parent/Child Cuối Cùng (FINAL_PARENT_TOP_K):", min_value=1, max_value=10, value=3)
    rerank_min_score = st.slider("Ngưỡng Lọc Evidence (RERANK_MIN_SCORE):", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

    use_fast_rerank = st.checkbox(
        "⚡ Chế độ Rerank Tốc độ cao (Fast RRF Reranker)",
        value=True,
        help="Bật để Rerank nhanh dựa trên điểm Parent RRF Score (trả kết quả tức thì trong 2-3s mà không phải chờ tải PyTorch Model 2.2GB)."
    )

    st.markdown("**Chiến lược Chunking:** `hierarchical` (Cố định)")

    st.divider()
    st.header("📊 Trạng Thái Hệ Thống")

    # Read config runtime
    cfg = hierarchical_rag.get_hierarchical_config({
        "multi_query_count": mq_count,
        "per_query_candidates": per_query_cands,
        "parent_candidates": parent_cands,
        "final_parent_top_k": final_top_k,
        "rerank_min_score": rerank_min_score,
        "use_fast_reranker": use_fast_rerank
    })

    # Gemini Key Status
    if cfg.get("api_key"):
        st.success("🟢 Gemini API Key: **Đã cấu hình**")
    else:
        st.warning("⚠️ Gemini API Key: **Chưa cấu hình (.env)**")

    # Hierarchy Store Status
    st_status = hierarchical_rag.hierarchy_status()
    if st_status["store_exists"]:
        st.success("🟢 Hierarchy Store: **Sẵn sàng**")
        m_info = st_status.get("manifest", {}).get("counts", {})
        st.caption(f"👶 Children: **{m_info.get('total_children', 0)}** | 👨‍👦 Parents: **{m_info.get('total_parents', 0)}** | ⚠️ Ambiguous: **{m_info.get('ambiguous_children_count', 0)}**")
    else:
        st.error("🔴 Hierarchy Store: **Chưa sẵn sàng**")

    # Chroma Collection Status
    adv_st = advanced_rag.check_advanced_status(strategy="hierarchical", custom_config=cfg)
    if adv_st.get("collection_exists"):
        st.success(f"🟢 Vector DB: **Chroma Collection Ready** ({adv_st.get('record_count', 0)} records)")
    else:
        st.error("🔴 Vector DB: **Collection Chưa Index**")

    st.divider()
    st.header("🤖 Model Metadata")
    st.caption(f"• **Embedding**: `{cfg['embedding_model']}` (768d)")
    st.caption(f"• **Generation**: `{cfg['generation_model']}`")
    st.caption(f"• **Reranker**: `{cfg['reranker_model']}`")

    st.divider()
    st.header("🛠️ Lệnh Quản Lý (Explicit Action)")
    if st.button("🔨 Xây dựng Hierarchy Store (`build-hierarchy`)"):
        with st.spinner("Đang xây dựng Parent-Child Hierarchy Registry..."):
            children, parents, stats = hierarchical_rag.build_hierarchy_registry()
            manifest = hierarchical_rag.save_hierarchy_store(children, parents, stats)
            st.success(f"Đã lưu Hierarchy Store! Total Children: {manifest['counts']['total_children']}, Parents: {manifest['counts']['total_parents']}")
            st.rerun()

    if st.button("📦 Chuẩn bị Vector DB (`prepare-semantic`)"):
        with st.spinner("Đang khởi tạo Chroma Vector Index..."):
            try:
                res_prep = advanced_rag.prepare_semantic(strategy="hierarchical", custom_config=cfg)
                st.success(f"Đã index {res_prep['indexed_count']} chunks vào Chroma Collection '{res_prep['collection_name']}'!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi chuẩn bị Vector DB: {e}")

# Session State Initialization
if "query_result_b09" not in st.session_state:
    st.session_state["query_result_b09"] = None

if "compare_result_b09" not in st.session_state:
    st.session_state["compare_result_b09"] = None

# 3. Main Dashboard Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask Advanced RAG",
    "🔀 Query Fan-out",
    "🌳 Parent–Child Explorer",
    "📊 Mode Comparison",
    "📈 Evaluation Benchmark"
])

# ---------------------------------------------------------
# TAB 1: ASK ADVANCED RAG
# ---------------------------------------------------------
with tab1:
    st.subheader("💬 Hỏi Đáp Pháp Lý Ngân Hàng với Multi-Query & Parent-Child RAG")
    user_question = st.text_area(
        "Nhập câu hỏi pháp lý ngân hàng:",
        value="Điều kiện vay vốn và các nhu cầu vốn không được cho vay được quy định thế nào?",
        height=100
    )

    col_btn, col_mode_disp = st.columns([2, 5])
    with col_btn:
        run_query_clicked = st.button("🚀 Chạy Hỏi Đáp (Run Query)", type="primary")
    with col_mode_disp:
        st.info(f"Đang chọn Mode: **`{selected_mode}`** | Multi-Query Count: **{mq_count}** | Target: **{final_top_k}** evidence")

    if run_query_clicked:
        if not user_question.strip():
            st.warning("Vui lòng nhập câu hỏi trước khi chạy.")
        else:
            with st.spinner("Đang thực thi pipeline Hỏi đáp Multi-Query & Parent-Child RAG..."):
                res = hierarchical_rag.query_hierarchical_rag(
                    question=user_question,
                    mode=selected_mode,
                    strategy="hierarchical",
                    custom_config=cfg
                )
                st.session_state["query_result_b09"] = res

    # Display Query Result from Session State
    q_res = st.session_state["query_result_b09"]
    if q_res:
        st.divider()
        b_text, b_msg, b_type = map_status_warning_badge(q_res["status"])
        if b_type == "success":
            st.success(f"{b_text}: {b_msg}")
        elif b_type == "warning":
            st.warning(f"{b_text}: {b_msg}")
        elif b_type == "error":
            st.error(f"{b_text}: {b_msg}")
        else:
            st.info(f"{b_text}: {b_msg}")

        # Metrics Row
        tr = q_res.get("trace", {})
        stg_lat = tr.get("stage_latencies_ms", {})
        api_counts = tr.get("api_call_counts", {"generation_calls": 0, "embedding_calls": 0})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("⏱️ Tổng Latency", f"{stg_lat.get('total', 0)} ms")
        with m2:
            st.metric("🧠 Gemini Gen Calls", f"{api_counts.get('generation_calls', 0)} calls")
        with m3:
            st.metric("🔍 Embedding Calls", f"{api_counts.get('embedding_calls', 0)} calls")
        with m4:
            st.metric("📋 Accepted Evidence", f"{len(q_res.get('accepted_evidence', []))} items")

        if q_res.get("warnings"):
            with st.expander("⚠️ Cảnh báo chi tiết trong Pipeline", expanded=False):
                for w in q_res["warnings"]:
                    st.write(f"- ⚠️ {w}")

        # Answer Box
        st.markdown("### 📝 Câu Trả Lời Pháp Lý")
        st.markdown(q_res.get("answer", "*Không có câu trả lời.*"))

        # Citations Section
        st.markdown("### 📚 Trích Dẫn Chứng Cứ (Citations)")
        st.markdown(format_citation_display(q_res.get("citations", [])))


# ---------------------------------------------------------
# TAB 2: QUERY FAN-OUT & QUERY-CHILD MATRIX
# ---------------------------------------------------------
with tab2:
    st.subheader("🔀 Mở Rộng Câu Hỏi Đa Hướng (Query Fan-out) & Ma Trận Query–Child")

    q_res = st.session_state["query_result_b09"]
    if not q_res:
        st.info("Vui lòng thực hiện một lượt hỏi đáp tại **Tab 1 — Ask Advanced RAG** để xem trực quan hóa Fan-out.")
    else:
        q_set = q_res.get("query_set", {})
        queries = q_set.get("queries", [])

        st.markdown(f"#### 🎯 Tập Câu Hỏi Mở Rộng ({len(queries)} Queries)")

        # Layout cards for Q0..Qn
        cols_q = st.columns(min(len(queries), 4))
        for idx, q_item in enumerate(queries):
            c_idx = idx % len(cols_q)
            with cols_q[c_idx]:
                with st.container(border=True):
                    is_orig = q_item.get("origin") == "original"
                    badge_str = "⭐ [GỐC Q0]" if is_orig else f"🔄 [{q_item.get('focus', 'VARIANT').upper()}]"
                    st.markdown(f"**{q_item['query_id']}** {badge_str}")
                    st.write(f"\"{q_item['text']}\"")
                    st.caption(f"Origin: `{q_item.get('origin')}` | Focus: `{q_item.get('focus')}`")

        st.divider()
        st.markdown("#### 🧩 Ma Trận Truy Xuất Query–Child Hits")
        st.caption("Hiển thị thứ hạng (Inner Rank) của từng Child Chunk theo từng nhánh Query và điểm Cross-Query RRF Score.")

        child_hits = q_res.get("child_hits", [])
        if not child_hits:
            st.warning("Không có Child Hits nào được truy xuất.")
        else:
            df_matrix = build_query_child_matrix(q_set, child_hits)
            st.dataframe(df_matrix)


# ---------------------------------------------------------
# TAB 3: PARENT-CHILD EXPLORER
# ---------------------------------------------------------
with tab3:
    st.subheader("🌳 Khám Phá Cây Cấu Trúc Phân Cấp Parent–Child")

    q_res = st.session_state["query_result_b09"]
    if not q_res:
        st.info("Vui lòng thực hiện một lượt hỏi đáp tại **Tab 1 — Ask Advanced RAG** để xem cây Parent–Child.")
    else:
        p_candidates = q_res.get("parent_candidates", [])
        if not p_candidates:
            st.warning("Chế độ hiện tại (Flat Mode) không thực hiện Parent Expansion. Hãy chuyển sang Mode `multi_parent` hoặc `single_parent` tại Sidebar.")
        else:
            tree_data = build_parent_tree_data(p_candidates)
            st.markdown(f"#### 👨‍👦 Danh Sách Parent Candidates ({len(tree_data)})")

            for p_node in tree_data:
                with st.expander(f"📄 [{p_node['parent_id']}] {p_node['heading']} — {p_node['rank_movement_badge']}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**Nguồn:** `{p_node['source']}` ({p_node['pages']})")
                        st.write(f"**Độ dài:** `{p_node['char_count']} ký tự`")
                    with c2:
                        st.write(f"**Parent RRF Score:** `{p_node['parent_rrf_score']:.6f}`")
                        st.write(f"**Parent Rerank Score:** `{p_node['parent_rerank_score']:.6f}` (Raw Logit: `{p_node['raw_logit']:.4f}`)")
                    with c3:
                        st.write(f"**Anchor Child ID:** `{p_node['anchor_child_id']}`")
                        st.write(f"**Support Queries:** `{', '.join(p_node['support_query_ids'])}`")

                    if p_node["warnings"]:
                        for w in p_node["warnings"]:
                            st.warning(f"⚠️ {w}")

                    st.markdown("**Các Child Chunks Hỗ Trợ (Supporting Children):**")
                    for cid in p_node["supporting_child_ids"]:
                        is_anchor = " 🌟 [ANCHOR]" if cid == p_node["anchor_child_id"] else ""
                        is_scoring = " 🎯 [SCORING]" if cid in p_node["scoring_child_ids"] else ""
                        st.write(f"  └── **Child ID:** `{cid}`{is_anchor}{is_scoring}")

                    st.divider()
                    st.markdown("**Toàn Văn Parent Document Context:**")
                    st.text_area(f"Nội dung {p_node['parent_id']}", value=p_node["text"], height=200, key=f"text_{p_node['parent_id']}")


# ---------------------------------------------------------
# TAB 4: MODE COMPARISON
# ---------------------------------------------------------
with tab4:
    st.subheader("📊 So Sánh Độc Lập 4 Chế Độ Pipeline (Retrieval & Reranking Only)")
    st.caption("Chạy cùng một câu hỏi qua `single_flat`, `multi_flat`, `single_parent`, `multi_parent` mà không gọi Gemini Answer Generation.")

    col_cmp_btn, _ = st.columns([3, 4])
    with col_cmp_btn:
        run_compare_clicked = st.button("⚖️ Chạy So Sánh 4 Mode (Run Comparison)", type="primary")

    if run_compare_clicked:
        if not user_question.strip():
            st.warning("Vui lòng nhập câu hỏi tại Tab 1.")
        else:
            with st.spinner("Đang chạy so sánh 4 Mode RAG..."):
                comp_res = hierarchical_rag.compare_hierarchical_rag(
                    question=user_question,
                    strategy="hierarchical",
                    custom_config=cfg
                )
                st.session_state["compare_result_b09"] = comp_res

    c_res = st.session_state["compare_result_b09"]
    if c_res:
        st.markdown(f"#### 📋 Kết Quả So Sánh Cho Câu Hỏi: *\"{c_res['question']}\"*")

        rows = []
        for m_name in c_res["modes_compared"]:
            m_data = c_res["results"][m_name]
            r_row = build_mode_comparison_row(m_name, m_data)
            rows.append(r_row)

        st.dataframe(rows)

        st.info("💡 **Ghi chú**: Không tự động tuyên bố Mode thắng nếu không có tập dữ liệu nhãn chuẩn (Gold Labels). Vui lòng chuyển sang **Tab 5 — Evaluation Benchmark** để đánh giá định lượng.")


# ---------------------------------------------------------
# TAB 5: EVALUATION BENCHMARK
# ---------------------------------------------------------
with tab5:
    st.subheader("📈 Báo Cáo Đánh Giá Định Lượng (Evaluation Benchmark)")
    st.caption("Đọc dữ liệu báo cáo đánh giá chất lượng từ file JSON mới nhất trong thư mục `reports/`.")

    reports_dir = BASE_DIR / "reports"
    json_reports = list(reports_dir.glob("*.json")) if reports_dir.exists() else []

    if not json_reports:
        st.info("Chưa tìm thấy báo cáo đánh giá trong `reports/`. Hãy chạy `evaluate.py` tại CLI để sinh báo cáo.")
    else:
        json_reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_report_file = json_reports[0]

        st.success(f"📄 Đang đọc báo cáo mới nhất: `{latest_report_file.name}`")
        try:
            with open(latest_report_file, "r", encoding="utf-8") as f:
                rep_data = json.load(f)

            summary = rep_data.get("summary", rep_data)
            st.markdown("#### 🎯 Các Chỉ Số Đánh Giá Chính (Summary Metrics)")

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Child Recall@K", f"{summary.get('child_recall_at_k', summary.get('recall_at_k', 0.0)):.4f}")
            with e2:
                st.metric("Parent Recall@K", f"{summary.get('parent_recall_at_k', summary.get('recall_at_k', 0.0)):.4f}")
            with e3:
                st.metric("MRR@K", f"{summary.get('mrr_at_k', 0.0):.4f}")
            with e4:
                st.metric("nDCG@K", f"{summary.get('ndcg_at_k', 0.0):.4f}")

            if summary.get("needs_human_review"):
                st.warning("⚠️ **Cảnh báo**: Tập dữ liệu kiểm thử chứa các mẫu cần con người đánh giá lại (`needs_human_review = True`).")

            st.json(summary)

        except Exception as e:
            st.error(f"Lỗi khi đọc file báo cáo đánh giá: {e}")
