"""
Buổi 08: Streamlit Comparison Dashboard cho Advanced RAG.

Giao diện tương tác trực quan cho phép so sánh song song 4 chế độ retrieval (bm25, semantic, hybrid, hybrid_rerank),
xem chi tiết pipeline trace đa tầng, thứ hạng candidate qua từng stage và báo cáo đánh giá định lượng.
"""

import json
from pathlib import Path
import sys
import time
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import advanced_rag
import rag

# -----------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced RAG Legal Dashboard — Buổi 08",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-accepted {
        border-left: 5px solid #10B981;
        background-color: #F0FDF4;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .card-rejected {
        border-left: 5px solid #F59E0B;
        background-color: #FFFBEB;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .badge-accepted {
        background-color: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-rejected {
        background-color: #F59E0B;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cache Functions (Không cache API Keys hoặc Mutable Session States)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_chunks(input_dir: str, strategy: str):
    """Cache dữ liệu corpus chunks theo strategy."""
    return rag.load_chunks(input_path=input_dir, strategy=strategy)


# -----------------------------------------------------------------------------
# Sidebar: System Configuration & Status Summary
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu Hình Pipeline")
    strategy = st.selectbox(
        "Chiến lược Chunking",
        options=["hierarchical", "flat"],
        index=0,
        help="Hierarchical bảo toàn cấu trúc Chương/Điều; Flat chia nhỏ đều."
    )

    default_mode = st.selectbox(
        "Chế độ Retrieval Mặc định",
        options=["hybrid_rerank", "hybrid", "semantic", "bm25"],
        index=0,
        help="hybrid_rerank là chế độ chính cho Advanced RAG."
    )

    st.markdown("---")
    st.subheader("📊 Tham Số Candidates & Gating")
    
    cfg = advanced_rag.get_advanced_config()
    final_top_k = st.number_input("Final Top K", min_value=1, max_value=20, value=cfg["final_top_k"])
    bm25_k = st.number_input("BM25 Candidate K", min_value=5, max_value=50, value=cfg["bm25_candidates"])
    sem_k = st.number_input("Semantic Candidate K", min_value=5, max_value=50, value=cfg["semantic_candidates"])
    rerank_k = st.number_input("Rerank Candidate K", min_value=5, max_value=50, value=cfg["rerank_candidates"])
    rerank_min_score = st.slider("Rerank Min Score (Gate)", min_value=0.0, max_value=1.0, value=cfg["rerank_min_score"], step=0.05)

    custom_cfg = dict(cfg)
    custom_cfg["final_top_k"] = final_top_k
    custom_cfg["bm25_candidates"] = bm25_k
    custom_cfg["semantic_candidates"] = sem_k
    custom_cfg["rerank_candidates"] = rerank_k
    custom_cfg["rerank_min_score"] = rerank_min_score

    st.markdown("---")
    st.subheader("🔍 Trạng Thái Hệ Thống")
    st_info = advanced_rag.check_advanced_status(strategy=strategy, custom_config=custom_cfg)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.caption("Corpus Chunks")
        st.write(f"**{st_info['corpus_size']}**")
        st.caption("BM25 Index")
        st.write("🟢 Sẵn sàng" if st_info["bm25_ready"] else "🔴 Chưa")
    with col_s2:
        st.caption("Chroma Records")
        st.write(f"**{st_info['record_count']}**")
        st.caption("Reranker Cache")
        st.write("🟢 Có sẵn" if st_info["reranker_cache_exists"] else "🟡 Chưa tải")

    if not st_info["has_api_key"]:
        st.warning("⚠️ Chưa cấu hình GEMINI_API_KEY trong .env")


# -----------------------------------------------------------------------------
# Header App
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">⚖️ Advanced RAG Legal Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Buổi 08 — So sánh Lexical (BM25), Semantic Candidate Retrieval, Reciprocal Rank Fusion & Cross-Encoder Reranking</div>', unsafe_allow_html=True)

# Khởi tạo Session State
if "latest_query" not in st.session_state:
    st.session_state["latest_query"] = "Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?"
if "query_result" not in st.session_state:
    st.session_state["query_result"] = None
if "compare_result" not in st.session_state:
    st.session_state["compare_result"] = None


# -----------------------------------------------------------------------------
# Main Tabs Structure
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Hỏi đáp Advanced RAG",
    "📊 So sánh Retrieval",
    "🔍 Pipeline Trace",
    "📈 Đánh giá Định lượng"
])


# -----------------------------------------------------------------------------
# TAB 1: HỎI ĐÁP ADVANCED RAG
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("💬 Thực thi Advanced RAG Answer Pipeline")
    
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        user_question = st.text_area(
            "Nhập câu hỏi pháp lý / nghiệp vụ ngân hàng:",
            value=st.session_state["latest_query"],
            height=90
        )
    with col_q2:
        selected_mode = st.selectbox(
            "Chế độ thực thi:",
            options=["hybrid_rerank", "hybrid", "semantic", "bm25"],
            index=0
        )
        btn_run = st.button("🚀 Thực thi Pipeline", use_container_width=True, type="primary")

    if btn_run and user_question.strip():
        st.session_state["latest_query"] = user_question.strip()
        with st.spinner("Đang truy xuất và xử lý câu trả lời..."):
            try:
                res = advanced_rag.query_advanced_rag(
                    question=user_question.strip(),
                    mode=selected_mode,
                    strategy=strategy,
                    custom_config=custom_cfg
                )
                st.session_state["query_result"] = res
            except Exception as e:
                st.error(f"❌ Lỗi thực thi Pipeline: {e}")

    result = st.session_state.get("query_result")
    if result:
        st.markdown("---")
        status = result["status"]

        # Banner trạng thái
        if status == "answered":
            st.success(f"✅ Trạng thái: **ANSWERED** | Chế độ: `{result['mode']}` | Accepted Evidence: {result['trace']['accepted']}")
        elif status == "insufficient_evidence":
            st.warning(f"⚠️ Trạng thái: **INSUFFICIENT EVIDENCE** — Không có evidence nào đạt ngưỡng tin cậy ({custom_cfg['rerank_min_score']}). LLM không được gọi.")
        elif status == "retrieval_only":
            st.info(f"ℹ️ Trạng thái: **RETRIEVAL ONLY** — Không gọi LLM Generation (Thiếu API Key hoặc LLM trả về rỗng).")
        elif status == "reranker_unavailable":
            st.error("🚨 Trạng thái: **RERANKER UNAVAILABLE** — Model Cross-Encoder Reranker chưa được tải hoặc lỗi nạp model.")
            st.markdown("""
            > **Hướng dẫn khắc phục:**
            > Chạy lệnh CLI sau trong terminal để tải model Reranker về local cache:
            > ```bash
            > python rag_foundation/buoi_08/advanced_rag.py rerank --question "Test"
            > ```
            """)

        # Hiển thị Answer & Citations
        if result["answer"]:
            st.markdown("### 📝 Câu Trả Lời")
            st.write(result["answer"])

            if result["citations"]:
                st.markdown("#### 📚 Danh Sách Trích Dẫn (Citations)")
                for cit in result["citations"]:
                    st.markdown(f"- **{cit['evidence_id']}**: `{cit['display']}`")

        # Warnings
        if result["warnings"]:
            for w in result["warnings"]:
                st.caption(f"⚠️ Warning: {w}")

        # Evidence Cards
        st.markdown("### 📄 Danh Sách Evidence Candidates")
        for ev in result["evidence"]:
            card_class = "card-accepted" if ev["accepted"] else "card-rejected"
            badge_html = '<span class="badge-accepted">ĐẠT NGƯỠNG</span>' if ev["accepted"] else '<span class="badge-rejected">BỊ LOẠI</span>'
            
            with st.container():
                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                col_c1, col_c2 = st.columns([3, 1])
                with col_c1:
                    st.markdown(f"**Chunk ID:** `{ev['chunk_id']}` | **Nguồn:** `{ev['source']}` (Trang {ev['page_start']}-{ev['page_end']}) {badge_html}", unsafe_allow_html=True)
                with col_c2:
                    if ev['rerank_score'] is not None:
                        st.markdown(f"**Rerank Score:** `{ev['rerank_score']:.4f}` (Rank #{ev['rerank_rank']})")
                    elif ev['semantic_distance'] is not None:
                        st.markdown(f"**Sem Dist:** `{ev['semantic_distance']:.4f}` (Rank #{ev['semantic_rank']})")
                    else:
                        st.markdown(f"**BM25 Score:** `{ev['bm25_score']:.2f}` (Rank #{ev['bm25_rank']})")

                st.text(ev["text"][:300] + ("..." if len(ev["text"]) > 300 else ""))
                
                with st.expander("Chi tiết chỉ số đa tầng (Metrics Breakdown)"):
                    st.json({
                        "BM25 Rank/Score": f"#{ev['bm25_rank']} ({ev['bm25_score']})" if ev['bm25_rank'] else "N/A",
                        "Semantic Rank/Distance": f"#{ev['semantic_rank']} ({ev['semantic_distance']})" if ev['semantic_rank'] else "N/A",
                        "RRF Fused Rank/Score": f"#{ev['fused_rank']} ({ev['rrf_score']:.5f})" if ev['fused_rank'] else "N/A",
                        "Rerank Rank/Score/Change": f"#{ev['rerank_rank']} ({ev['rerank_score']:.4f}) | Change: {ev['rank_change']}" if ev['rerank_rank'] else "N/A",
                        "Accepted Gate": ev["accepted"]
                    })
                st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: SO SÁNH RETRIEVAL (KHÔNG GỌI GENERATION)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 So sánh Song Song 4 Chế Độ Retrieval")
    st.caption("Thực thi so sánh thứ hạng candidate giữa BM25, Semantic, Hybrid RRF và Cross-Encoder Reranker. Không gọi LLM Generation.")

    col_cmp1, col_cmp2 = st.columns([3, 1])
    with col_cmp1:
        cmp_question = st.text_input("Câu hỏi so sánh:", value=st.session_state["latest_query"])
    with col_cmp2:
        btn_compare = st.button("🔍 So Sánh 4 Mode", use_container_width=True)

    if btn_compare and cmp_question.strip():
        with st.spinner("Đang thực thi so sánh 4 chế độ retrieval..."):
            try:
                cmp_res = advanced_rag.compare_retrieval_modes(
                    question=cmp_question.strip(),
                    strategy=strategy,
                    custom_config=custom_cfg
                )
                st.session_state["compare_result"] = cmp_res
            except Exception as e:
                st.error(f"❌ Lỗi khi so sánh: {e}")

    cmp_data = st.session_state.get("compare_result")
    if cmp_data:
        st.markdown("---")
        st.markdown("#### ⏱️ Độ trễ (Latency Breakdown)")
        lats = cmp_data["mode_latencies"]
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        col_l1.metric("BM25 Latency", f"{lats['bm25']} ms")
        col_l2.metric("Semantic Latency", f"{lats['semantic']} ms")
        col_l3.metric("Hybrid RRF Latency", f"{lats['hybrid']} ms")
        col_l4.metric("Hybrid Rerank Latency", f"{lats['hybrid_rerank']} ms")

        st.markdown("#### 📋 Bảng Báo Cáo Thứ Hạng Tổng Hợp")
        table_rows = []
        for row in cmp_data["comparison_table"]:
            b_r = f"#{row['ranks']['bm25']}" if row['ranks']['bm25'] is not None else "-"
            s_r = f"#{row['ranks']['semantic']}" if row['ranks']['semantic'] is not None else "-"
            h_r = f"#{row['ranks']['hybrid']}" if row['ranks']['hybrid'] is not None else "-"
            rr_r = f"#{row['ranks']['hybrid_rerank']}" if row['ranks']['hybrid_rerank'] is not None else "-"
            
            # Tính rank movement giữa Hybrid RRF và Rerank
            if row['ranks']['hybrid'] is not None and row['ranks']['hybrid_rerank'] is not None:
                chg = row['ranks']['hybrid'] - row['ranks']['hybrid_rerank']
                chg_str = f"+{chg}" if chg > 0 else str(chg)
            else:
                chg_str = "-"

            table_rows.append({
                "Chunk ID": row["chunk_id"],
                "Source": row["source"],
                "Trang": f"{row['page_start']}-{row['page_end']}",
                "BM25 Rank": b_r,
                "Semantic Rank": s_r,
                "Fused Rank": h_r,
                "Rerank Rank": rr_r,
                "Rank Change": chg_str
            })

        st.dataframe(table_rows, use_container_width=True)

        st.markdown("#### 📌 Top Candidate Chunks Theo Từng Mode")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown("**1️⃣ BM25 Lexical**")
            for item in cmp_data["mode_results"]["bm25"][:5]:
                st.caption(f"#{item['bm25_rank']} | Score: {item['bm25_score']:.2f} | `{item['chunk_id']}`")
                st.text(item["text"][:100] + "...")

        with col_m2:
            st.markdown("**2️⃣ Semantic**")
            for item in cmp_data["mode_results"]["semantic"][:5]:
                st.caption(f"#{item['semantic_rank']} | Dist: {item['semantic_distance']:.4f} | `{item['chunk_id']}`")
                st.text(item["text"][:100] + "...")

        with col_m3:
            st.markdown("**3️⃣ Hybrid RRF**")
            for item in cmp_data["mode_results"]["hybrid"][:5]:
                st.caption(f"#{item['fused_rank']} | RRF: {item['rrf_score']:.5f} | `{item['chunk_id']}`")
                st.text(item["text"][:100] + "...")

        with col_m4:
            st.markdown("**4️⃣ Cross-Encoder Rerank**")
            for item in cmp_data["mode_results"]["hybrid_rerank"][:5]:
                st.caption(f"#{item['rerank_rank']} | Score: {item['rerank_score']:.4f} | `{item['chunk_id']}`")
                st.text(item["text"][:100] + "...")


# -----------------------------------------------------------------------------
# TAB 3: PIPELINE TRACE
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🔍 Chi Tiết Luồng Xử Lý (Pipeline Trace)")

    if result and "trace" in result:
        tr = result["trace"]
        st.markdown("#### 🔄 Tiến Trình Candidate Chuyển Qua Các Tầng")
        
        c_tr1, c_tr2, c_tr3, c_tr4, c_tr5 = st.columns(5)
        c_tr1.metric("BM25 Candidates", tr["bm25_candidates"])
        c_tr2.metric("Semantic Candidates", tr["semantic_candidates"])
        c_tr3.metric("Union / Overlap", f"{tr['union']} / {tr['overlap']}")
        c_tr4.metric("Reranked", tr["reranked"])
        c_tr5.metric("Accepted Gate", tr["accepted"])

        st.markdown("---")
        st.markdown("#### ⏱️ Phân Bố Thời Gian Thực Thi (Latency Breakdown)")
        st.json(tr["latency_ms"])

    else:
        st.info("Hãy thực thi câu hỏi tại Tab 'Hỏi đáp Advanced RAG' để xem dữ liệu Trace trực quan.")

    st.markdown("---")
    st.markdown("""
    #### 💡 Ghi Chú Ý Nghĩa Chỉ Số Đa Tầng (Metrics Legend)
    - **BM25 Score**: Điểm trùng khớp từ vựng (Exact term match). Điểm càng cao càng tương quan từ vựng tốt.
    - **Cosine Distance**: Khoảng cách ngữ nghĩa giữa query vector và document vector. Giá trị càng **nhỏ** càng tương quan ngữ nghĩa cao.
    - **RRF Score**: Điểm số kết hợp theo thứ hạng Reciprocal Rank Fusion. Điểm càng **cao** thứ hạng tổng hợp càng tốt.
    - **Rerank Score**: Điểm chuẩn hóa từ Logit của Cross-Encoder Reranker qua hàm Sigmoid ($[0, 1]$). Điểm càng **cao** mức độ phù hợp ngữ cảnh càng lớn. *Lưu ý: Rerank score là điểm phân loại của model, không gọi là xác suất đúng tuyệt đối.*
    """)


# -----------------------------------------------------------------------------
# TAB 4: ĐÁNH GIÁ ĐỊNH LƯỢNG (EVALUATION REPORT READ-ONLY)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📈 Báo Cáo Đánh Giá Định Lượng (Offline Evaluation)")
    st.caption("Chỉ đọc file báo cáo JSON do script `evaluate.py` tạo ra. Không tự chạy thử nghiệm API hàng loạt khi mở trang.")

    reports_dir = BASE_DIR / "reports"
    report_files = sorted(list(reports_dir.glob("*.json")), reverse=True)

    if report_files:
        selected_report_file = st.selectbox(
            "Chọn file báo cáo đánh giá:",
            options=report_files,
            format_func=lambda p: p.name
        )

        try:
            with open(selected_report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            st.markdown(f"**Thời gian báo cáo:** `{report_data.get('timestamp', 'N/A')}` | **Strategy:** `{report_data.get('strategy', 'N/A')}`")

            if report_data.get("needs_human_review_warning"):
                st.warning("⚠️ **CẢNH BÁO:** Tập dữ liệu Gold Test chứa câu hỏi cần duyệt thủ công (`needs_human_review = True`). Không tuyên bố mode chiến thắng chính thức.")

            st.markdown("#### 📊 Bảng So Sánh Metrics Theo Mode")
            metrics_table = []
            for mode_name, m_val in report_data.get("metrics_by_mode", {}).items():
                metrics_table.append({
                    "Mode": mode_name,
                    "Recall@K": f"{m_val.get('recall', 0.0):.4f}",
                    "MRR@K": f"{m_val.get('mrr', 0.0):.4f}",
                    "nDCG@K": f"{m_val.get('ndcg', 0.0):.4f}",
                    "Mean Latency (ms)": f"{m_val.get('latency_mean', 0.0):.2f}",
                    "P50 Latency (ms)": f"{m_val.get('latency_p50', 0.0):.2f}"
                })

            st.dataframe(metrics_table, use_container_width=True)

            with st.expander("Xem file JSON Báo cáo gốc"):
                st.json(report_data)

        except Exception as e:
            st.error(f"❌ Không thể đọc file báo cáo `{selected_report_file.name}`: {e}")
    else:
        st.info("Chưa có file báo cáo JSON nào trong thư mục `reports/`.")
        st.markdown("""
        Để tạo báo cáo đánh giá định lượng offline, hãy chạy lệnh CLI sau:
        ```bash
        python rag_foundation/buoi_08/evaluate.py --strategy hierarchical --k 5
        ```
        """)
