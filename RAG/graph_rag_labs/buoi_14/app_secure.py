import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Thêm src vào đường dẫn để import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.secure_retriever import SecureRetriever
from src.config import Roles

# Cấu hình trang
st.set_page_config(page_title="Secure RAG Search — Buổi 15", layout="wide")

st.title("Secure RAG Search — Buổi 15 (RBAC)")

st.markdown("---")

# Cache bộ truy xuất để không phải load lại model khi thao tác trên giao diện
@st.cache_resource
def get_secure_retriever():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "processed", "chunks_secure.csv")
    if not os.path.exists(corpus_path):
        st.error(f"Không tìm thấy dữ liệu: {corpus_path}")
        st.stop()
    df = pd.read_csv(corpus_path)
    # Khởi tạo
    retriever = SecureRetriever(df)
    return retriever

try:
    retriever = get_secure_retriever()
except Exception as e:
    st.error(f"Lỗi khởi tạo Retriever: {e}")
    st.stop()

# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình Tìm kiếm")
    
    st.subheader("Bảo mật (RBAC)")
    valid_roles = Roles.get_valid_roles()
    selected_roles = st.multiselect(
        "Vai trò của bạn (Your Roles):", 
        options=valid_roles, 
        default=["Guest"],
        help="Chọn vai trò giả lập để kiểm thử phân quyền truy cập."
    )
    
    st.markdown("---")
    st.subheader("Thông số Model")
    method = st.selectbox("Chọn phương pháp truy xuất:", ["hybrid_secure", "bm25_secure", "dense_secure", "graph_secure"])
    candidate_k = st.number_input("Số lượng Candidate (K) ban đầu:", min_value=10, max_value=100, value=20)
    top_k = st.number_input("Top-k kết quả cuối cùng:", min_value=1, max_value=20, value=5)

# --- MAIN AREA ---
query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Quy định về bảo quản tiền mặt?")
search_clicked = st.button("Tìm kiếm", type="primary")

if search_clicked and query:
    if not selected_roles:
        st.error("Bạn phải chọn ít nhất một vai trò để thực hiện tìm kiếm an toàn!")
    else:
        st.markdown("---")
        
        with st.spinner(f"Đang tìm kiếm bằng phương pháp {method.upper()} với quyền {selected_roles}..."):
            # Gọi search tương ứng
            if method == "hybrid_secure":
                results = retriever.search(query, selected_roles, candidate_k=candidate_k, top_k=top_k)
            elif method == "bm25_secure":
                results = retriever.secure_bm25_search(query, selected_roles, top_k=top_k)
            elif method == "dense_secure":
                results = retriever.secure_dense_search(query, selected_roles, top_k=top_k)
            elif method == "graph_secure":
                results = retriever.secure_graph_search(query, selected_roles, top_k=top_k)
            else:
                results = []
        
        if not results:
            st.warning("Không tìm thấy kết quả nào hoặc bạn không có quyền truy cập các tài liệu phù hợp.")
        else:
            st.success(f"Tìm thấy {len(results)} kết quả hợp lệ.")
            
            # Thông báo đã áp dụng bộ lọc quyền
            st.info(f"🔒 Đã lọc kết quả theo quyền: {selected_roles}. Các tài liệu không đủ quyền truy cập đã bị ẩn khỏi hệ thống.", icon="🛡️")
            
            # --- HIỂN THỊ KẾT QUẢ ---
            st.subheader("Kết quả Truy xuất An toàn")
            for r in results:
                rank = r.get('final_rank', r.get('rank', '?'))
                score_str = ""
                if 'rerank_score' in r:
                    score_str = f" | Rerank Score: {r['rerank_score']:.4f}"
                elif 'rrf_score' in r:
                    score_str = f" | RRF Score: {r['rrf_score']:.4f}"
                elif 'retrieval_score' in r:
                    score_str = f" | Score: {r['retrieval_score']}"
                
                # Hiển thị vai trò cho phép của chunk
                doc_roles = r.get('allowed_roles', [])
                roles_str = ", ".join(doc_roles)
                
                with st.expander(f"[{rank}] Chunk: {r.get('chunk_id', 'N/A')}{score_str}"):
                    st.markdown(f"**Quyền xem (Allowed Roles):** `{roles_str}` 🔒")
                    st.markdown(f"**Văn bản (Document ID):** {r.get('document_id', 'N/A')}")
                    st.markdown(f"**Phương pháp:** {r.get('retrieval_method', 'N/A').upper()}")
                    if 'citation' in r:
                        st.markdown(f"**Trích dẫn (Citation):** {r['citation']}")
                    st.text_area("Nội dung:", r.get('text', ''), height=150, disabled=True)
                    
            st.markdown("---")
            
            # --- GRAPH HINTS ---
            st.subheader("Graph Hints (Neo4j) - Secure Mode")
            
            if not retriever.driver:
                st.warning("Trạng thái Neo4j: NOT READY.")
            else:
                doc_ids = list(set([str(r["document_id"]) for r in results if r.get("document_id")]))
                chunk_ids = list(set([str(r["chunk_id"]) for r in results if r.get("chunk_id")]))
                
                with retriever.driver.session() as session:
                    st.markdown("**Mối quan hệ pháp lý trực tiếp (từ Neo4j):**")
                    if doc_ids:
                        # Kiểm tra quyền trực tiếp trên query Neo4j
                        query_rel = """
                        MATCH (v1:VanBan {lab_session: 'buoi_15'})-[r]->(v2:VanBan {lab_session: 'buoi_15'})
                        WHERE (v1.document_id IN $doc_ids OR v2.document_id IN $doc_ids)
                          AND any(role IN v1.allowed_roles WHERE role IN $user_roles)
                          AND any(role IN v2.allowed_roles WHERE role IN $user_roles)
                        RETURN v1.document_id AS v1_id, type(r) AS rel_type, v2.document_id AS v2_id
                        LIMIT 10
                        """
                        try:
                            res = session.run(query_rel, doc_ids=doc_ids, user_roles=selected_roles)
                            rel_count = 0
                            for record in res:
                                rel_count += 1
                                st.write(f"- Văn bản `{record['v1_id']}` --[{record['rel_type']}]--> `{record['v2_id']}`")
                            if rel_count == 0:
                                st.write("- *(Không có quan hệ văn bản nào hoặc bạn không có quyền xem)*")
                        except Exception as e:
                            st.error(f"Lỗi query Neo4j: {e}")
                    else:
                        st.write("- *(Không có văn bản liên quan)*")

elif search_clicked and not query:
    st.warning("Vui lòng nhập câu hỏi để tìm kiếm.")
