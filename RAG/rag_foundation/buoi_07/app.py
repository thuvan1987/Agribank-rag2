"""
Buổi 07: Ứng dụng giao diện Streamlit RAG Chatbot.

Sử dụng trực tiếp các hàm cốt lõi từ rag.py:
- get_config()
- check_status()
- index_chunks()
- ask_question()
"""

import sys
from pathlib import Path
import streamlit as st

# Đảm bảo import được module rag.py từ cùng thư mục
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Foundation - Buổi 07",
    page_icon="🤖",
    layout="wide"
)

# Khởi tạo Session State
if "index_result" not in st.session_state:
    st.session_state["index_result"] = None
if "query_result" not in st.session_state:
    st.session_state["query_result"] = None

# Đọc cấu hình từ rag.py
try:
    config = rag.get_config()
    config_error = None
except Exception as e:
    config = {
        "has_api_key": False,
        "embedding_model": "gemini-embedding-2",
        "embedding_dim": 768,
        "generation_model": "gemini-3.5-flash-lite",
        "top_k": 5,
        "max_distance": 0.45
    }
    config_error = str(e)

# 2. SIDEBAR - Cấu hình & Trạng thái hệ thống
st.sidebar.title("⚙️ Cấu hình RAG System")

# Selectbox chọn Strategy
selected_strategy = st.sidebar.selectbox(
    "Chiến lược Chunking (Strategy):",
    options=["hierarchical", "semantic", "fixed-size"],
    index=0
)

# Slider chọn Top-K
selected_top_k = st.sidebar.slider(
    "Số lượng evidence truy xuất (Top-K):",
    min_value=1,
    max_value=10,
    value=config.get("top_k", 5)
)

st.sidebar.divider()
st.sidebar.subheader("📊 Trạng thái Hệ thống & Index")

# Gọi hàm status read-only từ rag.py
try:
    status_info = rag.check_status(strategy=selected_strategy)
except Exception as e:
    status_info = {
        "api_key_status": "Có" if config.get("has_api_key") else "Thiếu",
        "embedding_model": config.get("embedding_model"),
        "embedding_dim": config.get("embedding_dim"),
        "strategy": selected_strategy,
        "collection_name": "N/A",
        "collection_exists": False,
        "record_count": 0,
        "storage_dir": "N/A"
    }

# Hiển thị thông số cấu hình
st.sidebar.markdown(f"**API Key:** `{'Có' if config.get('has_api_key') else 'Thiếu'}`")
st.sidebar.markdown(f"**Embedding Model:** `{config.get('embedding_model')}`")
st.sidebar.markdown(f"**Embedding Dim:** `{config.get('embedding_dim')}`")
st.sidebar.markdown(f"**Generation Model:** `{config.get('generation_model')}`")
st.sidebar.markdown(f"**Distance Threshold:** `{config.get('max_distance')}`")
st.sidebar.divider()
st.sidebar.markdown(f"**Collection Name:** `{status_info['collection_name']}`")
st.sidebar.markdown(f"**Collection Tồn tại:** `{'Có' if status_info['collection_exists'] else 'Chưa'}`")
st.sidebar.markdown(f"**Số record hiện có:** `{status_info['record_count']}`")

if not config.get("has_api_key"):
    st.sidebar.warning("⚠️ Thiếu GEMINI_API_KEY trong file .env. Vui lòng bổ sung key trước khi Index/Query.")

# 3. NỘI DUNG CHÍNH (MAIN AREA)
st.title("🤖 Hệ Thống Hỏi Đáp RAG Chatbot - Buổi 07")
st.caption("Kiểm soát chất lượng dữ liệu | Embedding & Chroma persistent | Confidence Gate | Trích dẫn chính xác")

# TÁP 1: INDEX & CÂU HỎI
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🛠️ Index Dữ Liệu vào ChromaDB")
    reset_option = st.checkbox("Reset collection trước khi index (Xóa cũ tạo mới)", value=False)
    
    if st.button("⚡ Thực hiện Index Dữ liệu", use_container_width=True):
        if not config.get("has_api_key"):
            st.error("❌ Không thể index: Thiếu GEMINI_API_KEY trong file .env. Vui lòng mở file rag_foundation/buoi_07/.env và điền API Key.")
        else:
            with st.spinner(f"Đang xử lý loader, tạo embeddings và index dữ liệu cho strategy '{selected_strategy}'..."):
                try:
                    res = rag.index_chunks(strategy=selected_strategy, reset=reset_option)
                    st.session_state["index_result"] = res
                    st.success(f"✅ Index thành công! Đã thêm {res['indexed_chunks']} chunks vào collection '{res['collection_name']}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình index: {e}")

    # Hiển thị kết quả Index gần nhất
    idx_res = st.session_state.get("index_result")
    if idx_res:
        with st.expander("📄 Chi tiết kết quả Index gần nhất", expanded=True):
            st.write(f"- **Collection Name:** `{idx_res['collection_name']}`")
            st.write(f"- **Số chunk vừa index:** `{idx_res['indexed_chunks']}`")
            st.write(f"- **Tổng record hiện có:** `{idx_res['total_collection_records']}`")
            st.write(f"- **Đã Reset collection:** `{'Có' if idx_res['reset_performed'] else 'Không'}`")

with col_right:
    st.subheader("💬 Hỏi Đáp RAG")
    question_text = st.text_area(
        "Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: Điều kiện CIF Incoterms 2020 quy định trách nhiệm giao hàng như thế nào?",
        height=110
    )

    if st.button("🔍 Gửi Câu Hỏi", type="primary", use_container_width=True):
        if not question_text.strip():
            st.warning("⚠️ Vui lòng nhập nội dung câu hỏi trước khi gửi.")
        elif not config.get("has_api_key"):
            st.error("❌ Không thể query: Thiếu GEMINI_API_KEY trong file .env.")
        elif not status_info["collection_exists"]:
            st.error(f"❌ Collection '{status_info['collection_name']}' chưa tồn tại. Vui lòng bấm 'Index Dữ liệu' trước.")
        elif status_info["record_count"] == 0:
            st.error(f"❌ Collection rỗng (0 record). Vui lòng index dữ liệu trước khi hỏi.")
        else:
            with st.spinner("Đang tạo query embedding, truy xuất tài liệu và gọi Gemini LLM..."):
                try:
                    q_res = rag.ask_question(
                        question=question_text,
                        top_k=selected_top_k,
                        strategy=selected_strategy
                    )
                    st.session_state["query_result"] = q_res
                except Exception as e:
                    st.error(f"❌ Lỗi khi thực hiện hỏi đáp: {e}")

st.divider()

# 4. HIỂN THỊ CÂU TRẢ LỜI VÀ EVIDENCES
query_res = st.session_state.get("query_result")

if query_res:
    st.subheader("💡 Kết Quả Trả Lời")

    # Status Badge
    st_val = query_res.get("status")
    if st_val == "answered":
        st.success("✅ **Trạng thái:** Đã tạo câu trả lời tổng hợp từ tài liệu (Answered)")
    elif st_val == "insufficient_evidence":
        st.warning("⚠️ **Trạng thái:** Không tìm thấy đủ thông tin liên quan trong tài liệu (Insufficient Evidence)")
    elif st_val == "retrieval_only":
        st.info("ℹ️ **Trạng thái:** Đã truy xuất được nguồn nhưng chưa tổng hợp câu trả lời (Retrieval Only)")

    # Nội dung câu trả lời
    st.markdown(query_res.get("answer", ""))

    # Hiển thị Warnings nếu có
    if query_res.get("warnings"):
        with st.expander("⚠️ Cảnh báo trong quá trình xử lý", expanded=False):
            for w in query_res["warnings"]:
                st.warning(w)

    # Hiển thị Citations
    if query_res.get("citations"):
        st.markdown("#### 📌 Danh sách Trích dẫn chính xác (Mapped Citations):")
        for cit in query_res["citations"]:
            st.markdown(f"- **[{cit['evidence_id']}]**: `{cit['display']}`")

st.divider()

# 5. HIỂN THỊ NGUỒN THAM KHẢO (EVIDENCES)
st.subheader("📚 Nguồn Tham Khảo (Retrieved Evidences)")
st.caption(
    "Khoảng cách (Cosine Distance) càng nhỏ thể hiện mức độ tương đồng ngữ nghĩa càng cao. "
    f"Các nguồn có `Distance <= {config.get('max_distance')}` mới được chấp nhận đưa vào prompt."
)

if not query_res or not query_res.get("evidence"):
    st.info("Chưa có nguồn tham khảo nào được truy xuất.")
else:
    evidences = query_res["evidence"]
    for ev in evidences:
        acc = ev.get("accepted", False)
        acc_tag = "🟢 [ĐẠT NGƯỠNG]" if acc else "🔴 [BỊ LOẠI]"
        p_start = ev.get("page_start", 1)
        p_end = ev.get("page_end", 1)
        page_str = f"tr. {p_start}" if p_start == p_end else f"tr. {p_start}-{p_end}"

        label_title = f"{ev['evidence_id']} {acc_tag} – {ev['source']} – {page_str} – chunk: {ev['chunk_id']} (Dist: {ev['distance']:.4f})"

        with st.expander(label_title, expanded=acc):
            st.write(f"- **Evidence ID:** `{ev['evidence_id']}`")
            st.write(f"- **Tên nguồn (Source):** `{ev['source']}`")
            st.write(f"- **Trang:** `{page_str}`")
            st.write(f"- **Chunk ID:** `{ev['chunk_id']}`")
            st.write(f"- **Cosine Distance:** `{ev['distance']:.6f}` (Thấp hơn = liên quan hơn)")
            st.write(f"- **Trạng thái Confidence Gate:** `{'Đạt ngưỡng (Accepted)' if acc else 'Bị loại (Distance > Threshold)'}`")
            if not acc:
                st.caption("🔴 *Nguồn này không đạt ngưỡng tin cậy nên KHÔNG được đưa vào Prompt sinh câu trả lời.*")
            st.markdown("**Nội dung Chunk:**")
            st.code(ev.get("text", ""), language="text")
