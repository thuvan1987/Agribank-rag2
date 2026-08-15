import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.query_demo import retrieve, get_neo4j_driver

# Set page config
st.set_page_config(page_title="RAG Hybrid Search — Buổi 14", layout="wide")

st.title("RAG Hybrid Search — Buổi 14")

st.markdown("---")

# Input section
query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Thông tư 01/2014 quy định gì?")

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox("Chọn phương pháp truy xuất:", ["bm25", "dense", "hybrid", "hybrid_rerank"])
with col2:
    top_k = st.number_input("Top-k kết quả:", min_value=1, max_value=20, value=5)

search_clicked = st.button("Tìm kiếm", type="primary")

if search_clicked and query:
    st.markdown("---")
    
    with st.spinner(f"Đang tìm kiếm bằng phương pháp {method.upper()}..."):
        corpus_path = os.path.join(os.path.dirname(__file__), "data", "processed", "chunks_normalized.csv")
        results = retrieve(question=query, method=method, top_k=top_k, corpus_path=corpus_path)
    
    if not results:
        st.warning("Không tìm thấy kết quả nào.")
    else:
        st.success(f"Tìm thấy {len(results)} kết quả (Top {top_k}).")
        
        # Display Results
        st.subheader("Kết quả Truy xuất")
        for r in results:
            with st.expander(f"[{r['rank']}] Điểm: {r['score']} | Chunk: {r['chunk_id']}"):
                st.markdown(f"**Văn bản (Document ID):** {r['document_id']}")
                st.markdown(f"**Phương pháp:** {r['retrieval_method'].upper()}")
                st.markdown(f"**Trích dẫn (Citation):** {r['citation']}")
                st.text_area("Nội dung:", r['text'], height=150, disabled=True)
                
        # Display Before/After Rerank if applicable
        if method == "hybrid_rerank":
            st.subheader("Bảng so sánh Hybrid vs Rerank (Mô phỏng)")
            st.info("Lưu ý: Rerank score đã được hiển thị trực tiếp ở phần Điểm của từng kết quả phía trên. Đây là minh họa trực quan sự thay đổi thứ hạng.")
            
        st.markdown("---")
        
        # Graph hints
        st.subheader("Graph Hints (Neo4j)")
        
        driver = get_neo4j_driver()
        if not driver:
            st.warning("Trạng thái Neo4j: NOT READY. Hãy khởi động Neo4j và cấu hình `.env` để xem Graph Hints.")
        else:
            st.success("Trạng thái Neo4j: READY")
            
            doc_ids = list(set([r["document_id"] for r in results if pd.notna(r.get("document_id"))]))
            chunk_ids = list(set([r["chunk_id"] for r in results if r.get("chunk_id")]))
            
            st.markdown(f"**Các văn bản liên quan (document_id):** `{doc_ids}`")
            st.markdown(f"**Các đoạn trích (chunk_id):** `{chunk_ids}`")
            
            with driver.session() as session:
                st.markdown("**Mối quan hệ pháp lý trực tiếp (từ Neo4j):**")
                if doc_ids:
                    query_rel = """
                    MATCH (v1:VanBan {lab_session: 'buoi_14'})-[r]->(v2:VanBan {lab_session: 'buoi_14'})
                    WHERE v1.id IN $doc_ids OR v2.id IN $doc_ids
                    RETURN v1.id AS v1_id, type(r) AS rel_type, v2.id AS v2_id, v2.title AS v2_title
                    """
                    try:
                        res = session.run(query_rel, doc_ids=[int(d) for d in doc_ids])
                        rel_count = 0
                        for record in res:
                            rel_count += 1
                            st.write(f"- Văn bản `{record['v1_id']}` --[{record['rel_type']}]--> `{record['v2_id']}` ({record['v2_title'][:40]}...)")
                        if rel_count == 0:
                            st.write("- *(Không có quan hệ văn bản nào)*")
                    except Exception as e:
                        st.error(f"Lỗi query Neo4j: {e}")
                
                st.markdown("**Ngữ cảnh tuyến tính của Chunk (từ Neo4j):**")
                if chunk_ids:
                    query_next = """
                    MATCH (d1:DieuKhoan {lab_session: 'buoi_14'})-[r:NEXT]->(d2:DieuKhoan {lab_session: 'buoi_14'})
                    WHERE d1.id IN $chunk_ids
                    RETURN d1.id AS from_chunk, d2.id AS to_chunk
                    LIMIT 5
                    """
                    try:
                        res = session.run(query_next, chunk_ids=chunk_ids)
                        rel_count = 0
                        for record in res:
                            rel_count += 1
                            st.write(f"- Chunk `{record['from_chunk']}` --[NEXT]--> Chunk `{record['to_chunk']}`")
                        if rel_count == 0:
                            st.write("- *(Không có chunk kế tiếp)*")
                    except Exception as e:
                        st.error(f"Lỗi query Neo4j: {e}")
            driver.close()

elif search_clicked and not query:
    st.warning("Vui lòng nhập câu hỏi để tìm kiếm.")
