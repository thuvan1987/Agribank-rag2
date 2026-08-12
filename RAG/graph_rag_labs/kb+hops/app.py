import os
import sys
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
sys.path.insert(0, str(BASE_DIR))

from graph_rag_retriever import GraphRAGRetriever
from step6_llm_qa_eval import call_gemini_llm

st.set_page_config(
    page_title="Agribank Graph RAG - Buổi 11",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        color: #8B0000;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4A4A4A;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #8B0000;
        margin-bottom: 12px;
    }
    .rel-badge {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Load environment
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip()

# Initialize session state for retriever
@st.cache_resource
def get_retriever():
    return GraphRAGRetriever(env_path=ENV_FILE)

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
    st.title("⚙️ Cấu hình Graph RAG")
    st.markdown("---")

    n_hops = st.radio(
        "🔗 Số bước nhảy Đồ thị (Graph Hops):",
        options=[0, 1, 2],
        index=1,
        format_func=lambda x: f"{x}-Hop (Chỉ Vector Search)" if x == 0 else f"{x}-Hop (Vector + Graph Expansion)"
    )

    top_k = st.slider("🎯 Số lượng Vector Chunks (Top-K):", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.subheader("📊 Trạng thái Hệ thống")
    st.success("✅ Neo4j DB (`kb-hops`): Connected")
    st.info("🤖 Model Vector: `MSMARCO v5` (384d)")
    st.info("🧠 Model LLM: `gemini-3.5-flash-lite`")

    st.markdown("---")
    st.caption("Agribank AI Training - Buổi 11 (Multi-hop Graph RAG)")


st.markdown('<div class="main-title">🏦 AGRIBANK RAG 2 - BUỔI 11: MULTI-HOP GRAPH RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống Tra cứu Văn bản Pháp luật Ngân hàng với Mở rộng Đồ thị Đa bước & Gemini LLM</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Hỏi Đáp Trực Tiếp", "📊 Báo cáo Đánh giá (0-Hop vs 1-Hop)", "🕸️ Cấu trúc Đồ thị Neo4j"])

SAMPLE_QUESTIONS = [
    "Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?",
    "Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì?",
    "Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì?",
    "Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào?",
    "Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không?"
]

with tab1:
    st.subheader("❓ Đặt câu hỏi tra cứu")

    selected_sample = st.selectbox(
        "Chọn câu hỏi mẫu kiểm thử (hoặc tự nhập câu hỏi bên dưới):",
        options=["-- Chọn câu hỏi mẫu --"] + SAMPLE_QUESTIONS
    )

    user_query = st.text_area(
        "Nhập câu hỏi của bạn:",
        value="" if selected_sample == "-- Chọn câu hỏi mẫu --" else selected_sample,
        height=90,
        placeholder="Ví dụ: Văn bản 52/VBHN-NHNN được hợp nhất từ thông tư nào?"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        btn_submit = st.button("🚀 Truy vấn Graph RAG", type="primary", width="stretch")

    if btn_submit and user_query.strip():
        if not api_key:
            st.error("❌ Thiếu GEMINI_API_KEY trong file .env. Vui lòng kiểm tra lại cấu hình!")
        else:
            with st.spinner("🔍 Đang truy vấn Vector & Mở rộng Đồ thị Multi-hop trong Neo4j..."):
                retriever = get_retriever()
                ctx_data = retriever.get_multi_hop_context(user_query, top_k=top_k, n_hops=n_hops)

            st.markdown("### 🤖 Kết quả Trả lời từ Gemini LLM")
            with st.spinner("🧠 Gemini LLM đang tổng hợp câu trả lời dựa trên Ngữ cảnh Đồ thị..."):
                gemini_client = genai.Client(api_key=api_key)
                answer = call_gemini_llm(gemini_client, user_query, ctx_data["formatted_context"])

            st.markdown(f'<div class="metric-card">{answer}</div>', unsafe_allow_html=True)

            with st.expander(f"🔍 Chi tiết Ngữ cảnh Graph RAG Thu thập được ({n_hops}-Hop Expansion)", expanded=True):
                st.markdown(f"**Top-{top_k} Vector Chunks khớp trực tiếp:**")
                for sc in ctx_data["seed_chunks"]:
                    st.write(f"- 📄 **{sc['so_ky_hieu']}** ({sc['doc_title']}) | Chunk: `{sc['chunk_id']}` | Score: `{sc['score']*100:.1f}%`")

                if n_hops > 0:
                    st.markdown(f"**Mối quan hệ Đồ thị truy vết được ({n_hops}-Hop):**")
                    if not ctx_data["multi_hop_relationships"]:
                        st.info("Không có liên kết văn bản liên quan nào trong bán kính bước nhảy.")
                    else:
                        for hr in ctx_data["multi_hop_relationships"]:
                            rels = " -> ".join(hr["rel_types"])
                            st.write(f"- 🔗 `[{hr['seed_so_ky_hieu']}]` --**[{rels}]**--> `[{hr['related_so_ky_hieu']}]` ({hr['related_doc_title']})")

                with st.expander("📄 Xem toàn bộ văn bản Prompt Context truyền tới LLM"):
                    st.code(ctx_data["formatted_context"], language="markdown")

with tab2:
    st.subheader("📊 Báo cáo So sánh Thực nghiệm: 0-Hop vs 1-Hop Graph RAG")
    comparison_file = BASE_DIR / "qa_comparison.md"
    if comparison_file.exists():
        report_text = comparison_file.read_text(encoding="utf-8")
        st.markdown(report_text)
    else:
        st.warning("Chưa tìm thấy tệp báo cáo `qa_comparison.md`. Bạn có thể chạy `step6_llm_qa_eval.py` để khởi tạo báo cáo.")

with tab3:
    st.subheader("🕸️ Thống kê Cơ sở Dữ liệu Đồ thị Neo4j (`kb-hops`)")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Số lượng Văn bản (:Document)", "15 Văn bản")
    col_b.metric("Số lượng Phân đoạn (:Chunk)", "6,465 Chunks")
    col_c.metric("Số lượng Cạnh Mối quan hệ", "12,938 Relationships")

    st.markdown("""
    #### 🧬 Lược đồ Mối quan hệ Pháp lý trong Đồ thị:
    - **`THAY_THE`**: Văn bản mới thay thế hiệu lực của văn bản cũ (VD: *Nghị định 46/2023/NĐ-CP replace Nghị định 73/2016/NĐ-CP*).
    - **`HOP_NHAT`**: Văn bản hợp nhất kết hợp nội dung từ các thông tư gốc (VD: *Văn bản hợp nhất 52/VBHN-NHNN từ Thông tư 56/2024/TT-NHNN*).
    - **`VAN_BAN_BO_SUNG`**: Văn bản sửa đổi, bổ sung nội dung (VD: *Thông tư 63/2025/TT-NHNN bổ sung cho Thông tư 01/2025/TT-NHNN*).
    - **`CAN_CU`**: Căn cứ pháp lý làm cơ sở ban hành (VD: *Thông tư 41/2016/TT-NHNN căn cứ Luật Ngân hàng Nhà nước 46/2010/QH12*).
    """)
