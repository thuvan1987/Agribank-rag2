import os
import streamlit as st
from dotenv import load_dotenv
import rag

# Cấu hình trang Streamlit
st.set_page_config(page_title="RAG Workshop - Buổi 06", page_icon="🤖", layout="wide")

load_dotenv()

# --- SIDEBAR: TRẠNG THÁI HỆ THỐNG ---
st.sidebar.title("⚙️ Trạng thái Hệ thống")

# Lấy thông tin trạng thái từ rag.py
sys_status = rag.status()
backend = sys_status.get("db_backend", "unknown")
api_key = os.getenv("GEMINI_API_KEY", "").strip()

# 1. Trạng thái PostgreSQL
if backend == "postgres":
    st.sidebar.success("🟢 PostgreSQL: Đã kết nối (`rag_db`)")
else:
    st.sidebar.warning("🟡 PostgreSQL: Chưa chạy (Dùng SQLite local .db)")

# 2. Trạng thái ChromaDB
st.sidebar.info(f"🟢 ChromaDB: Embedded Local ({sys_status.get('chroma_chunks', 0)} chunks)")

# 3. Trạng thái Gemini API Key
if api_key:
    st.sidebar.success("🟢 Gemini API Key: Có")
else:
    st.sidebar.error("🔴 Gemini API Key: Thiếu (Chỉ hỗ trợ Retrieval)")

st.sidebar.markdown("---")
st.sidebar.write(f"📄 **Tài liệu**: {sys_status.get('documents', 0)}")
st.sidebar.write(f"🧩 **Tổng Chunks**: {sys_status.get('chunks', 0)}")


# --- MAIN AREA ---
st.title("🤖 RAG Demo - Buổi 06 Workshop")

# Nút Index Dữ liệu
col_idx, _ = st.columns([1, 4])
with col_idx:
    if st.button("🔄 Index Dữ liệu", type="secondary"):
        with st.spinner("Đang xử lý đọc JSON & nạp vector vào ChromaDB..."):
            res = rag.index()
            if res.get("status") == "success":
                st.success(f"Đã index thành công {res.get('chunks', 0)} chunks từ {res.get('documents', 0)} tài liệu!")
                st.rerun()
            else:
                st.error(res.get("message", "Lỗi khi index dữ liệu."))

st.markdown("---")

# Form nhập câu hỏi & Tham số Top-k
col_q, col_k = st.columns([3, 1])

with col_q:
    question = st.text_input("❓ Nhập câu hỏi:", placeholder="Ví dụ: Incoterms 2020 là gì?")

with col_k:
    top_k = st.number_input("Top-k Retrieval", min_value=1, max_value=10, value=3)

if st.button("🔍 Gửi câu hỏi", type="primary"):
    if not question.strip():
        st.warning("Vui lòng nhập nội dung câu hỏi!")
    else:
        with st.spinner("Đang tìm kiếm và xử lý..."):
            result = rag.ask(question=question, top_k=top_k)

        # Hiển thị câu trả lời (Answer)
        st.markdown("### 💡 Answer (Câu trả lời)")
        if api_key:
            st.success(result.get("answer", ""))
        else:
            st.warning(result.get("answer", ""))

        st.markdown("---")

        # Hiển thị kết quả Top-k Retrieval
        st.markdown(f"### 📚 Kết quả Top-{top_k} Retrieval")
        sources = result.get("sources", [])
        context_text = result.get("context", "")

        if sources:
            for idx, src in enumerate(sources, 1):
                with st.expander(f"Chunk #{idx} - Tài liệu: {src.get('doc_name')} (ID: {src.get('chunk_id')})"):
                    # Tìm nội dung đoạn text tương ứng trong context
                    st.text(src.get("text", context_text))
        else:
            st.info("Không tìm thấy kết quả trích xuất phù hợp.")
