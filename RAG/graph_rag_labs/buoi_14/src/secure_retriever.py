import os
import json
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.reranker import Reranker

class SecureRetriever:
    def __init__(self, chunks_df: pd.DataFrame, rrf_k: int = 60):
        # Chuyển đổi DataFrame thành list các dict và parse trường allowed_roles
        self.chunks = []
        for _, row in chunks_df.iterrows():
            chunk_dict = row.to_dict()
            if isinstance(chunk_dict.get('allowed_roles'), str):
                try:
                    chunk_dict['allowed_roles'] = json.loads(chunk_dict['allowed_roles'])
                except json.JSONDecodeError:
                    chunk_dict['allowed_roles'] = []
            self.chunks.append(chunk_dict)
            
        self.chunks_by_id = {str(c["chunk_id"]).strip(): c for c in self.chunks}
        self.rrf_k = rrf_k

        # Khởi tạo các Retriever cơ bản
        print("[SecureRetriever] Khởi tạo BM25 Retriever...")
        self.bm25_retriever = BM25Retriever(self.chunks)
        
        print("[SecureRetriever] Khởi tạo Dense Retriever...")
        self.dense_retriever = DenseRetriever(self.chunks)
        
        print("[SecureRetriever] Khởi tạo Reranker...")
        self.reranker = Reranker()
        
        # Kết nối Neo4j với cấu hình từ .env
        load_dotenv()
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

    def close(self):
        """Đóng kết nối Neo4j."""
        self.driver.close()

    def has_access(self, chunk_roles: List[str], user_roles: List[str]) -> bool:
        """Kiểm tra xem người dùng có ít nhất một quyền nằm trong allowed_roles không."""
        if not chunk_roles:
            return False
        return any(role in chunk_roles for role in user_roles)

    def secure_bm25_search(self, question: str, user_roles: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Lấy top_k ứng viên từ BM25 nhưng đã được lọc qua bảo mật."""
        # Lấy số lượng lớn hơn để đảm bảo sau khi lọc vẫn đủ top_k
        candidate_k = max(top_k * 5, 50)
        raw_results = self.bm25_retriever.search(question, top_k=candidate_k)
        
        filtered = []
        for hit in raw_results:
            chunk = self.chunks_by_id.get(str(hit["chunk_id"]))
            if chunk and self.has_access(chunk.get("allowed_roles", []), user_roles):
                hit["allowed_roles"] = chunk.get("allowed_roles", [])
                filtered.append(hit)
                if len(filtered) == top_k:
                    break
        return filtered

    def secure_dense_search(self, question: str, user_roles: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Lấy top_k ứng viên từ Dense nhưng đã được hậu lọc qua bảo mật."""
        candidate_k = max(top_k * 5, 50)
        raw_results = self.dense_retriever.search(question, top_k=candidate_k)
        
        filtered = []
        for hit in raw_results:
            chunk = self.chunks_by_id.get(str(hit["chunk_id"]))
            if chunk and self.has_access(chunk.get("allowed_roles", []), user_roles):
                hit["allowed_roles"] = chunk.get("allowed_roles", [])
                filtered.append(hit)
                if len(filtered) == top_k:
                    break
        return filtered

    def secure_graph_search(self, question: str, user_roles: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Truy vấn đồ thị với Neo4j có tích hợp mệnh đề kiểm tra quyền RBAC.
        Dùng Cypher để lấy các node thỏa mãn từ khóa và user_roles.
        """
        # Trích xuất keyword cơ bản (có thể cải tiến bằng NER)
        keywords = [w for w in question.split() if len(w) > 3]
        if not keywords:
            keywords = [question]
            
        # Truy vấn Cypher kèm theo kiểm tra quyền allowed_roles
        cypher_query = """
        MATCH (d:DieuKhoan)
        WHERE any(role IN d.allowed_roles WHERE role IN $user_roles)
          AND any(kw IN $keywords WHERE toLower(d.text) CONTAINS toLower(kw))
          AND d.lab_session = 'buoi_15'
        RETURN d.chunk_id AS chunk_id, d.text AS text, d.allowed_roles AS allowed_roles
        LIMIT $top_k
        """
        
        results = []
        try:
            with self.driver.session() as session:
                records = session.run(cypher_query, user_roles=user_roles, keywords=keywords, top_k=top_k)
                for i, record in enumerate(records):
                    chunk_id = record["chunk_id"]
                    chunk = self.chunks_by_id.get(str(chunk_id), {})
                    results.append({
                        "rank": i + 1,
                        "chunk_id": chunk_id,
                        "document_id": chunk.get("document_id", ""),
                        "title": chunk.get("title", ""),
                        "article": chunk.get("article", ""),
                        "text": record["text"],
                        "allowed_roles": record["allowed_roles"],
                        "retrieval_method": "graph"
                    })
        except Exception as e:
            print(f"[Graph Search] Lỗi truy vấn: {e}")
            
        return results

    def search(self, question: str, user_roles: List[str], candidate_k: int = 20, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hàm Search tổng hợp kết hợp BM25, Dense và Graph với Hybrid Fusion (RRF).
        Đảm bảo RRF và Reranker chỉ làm việc trên các chunk đã qua bảo mật.
        """
        if not question.strip():
            return []

        # 1. Tìm kiếm và lọc bảo mật ở từng phương thức (BM25, Dense, Graph)
        b_hits = self.secure_bm25_search(question, user_roles, top_k=candidate_k)
        d_hits = self.secure_dense_search(question, user_roles, top_k=candidate_k)
        g_hits = self.secure_graph_search(question, user_roles, top_k=candidate_k)

        # 2. Xây dựng bản đồ Rank để chuẩn bị cho RRF Fusion
        b_map = {h["chunk_id"]: h["rank"] for h in b_hits}
        d_map = {h["chunk_id"]: h["rank"] for h in d_hits}
        g_map = {h["chunk_id"]: h["rank"] for h in g_hits}

        all_ids = set(b_map.keys()) | set(d_map.keys()) | set(g_map.keys())

        scored_candidates = []
        for cid in all_ids:
            b_rank = b_map.get(cid)
            d_rank = d_map.get(cid)
            g_rank = g_map.get(cid)

            # Tính điểm RRF
            rrf_score = 0.0
            if b_rank is not None: rrf_score += 1.0 / (self.rrf_k + b_rank)
            if d_rank is not None: rrf_score += 1.0 / (self.rrf_k + d_rank)
            if g_rank is not None: rrf_score += 1.0 / (self.rrf_k + g_rank)

            c_info = self.chunks_by_id.get(cid, {})

            scored_candidates.append({
                "chunk_id": cid,
                "document_id": c_info.get("document_id", ""),
                "text": c_info.get("text", ""),
                "title": c_info.get("title", ""),
                "article": c_info.get("article", ""),
                "citation": f"[{c_info.get('title', '')} | {c_info.get('article', '')} | {cid}]",
                "allowed_roles": c_info.get("allowed_roles", []),
                "bm25_rank": b_rank if b_rank is not None else "N/A",
                "dense_rank": d_rank if d_rank is not None else "N/A",
                "graph_rank": g_rank if g_rank is not None else "N/A",
                "rrf_score": round(rrf_score, 6),
                "retrieval_method": "secure_hybrid_rrf"
            })

        # Sắp xếp theo điểm RRF và chỉ lấy số lượng ứng viên cần thiết cho Reranker
        scored_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_candidates = scored_candidates[:top_k * 2] # Gửi nhiều hơn một chút cho reranker

        # 3. Chạy Reranker trên các ứng viên đã được chứng thực quyền truy cập
        if not top_candidates:
            return []
            
        final_results = self.reranker.rerank(question, top_candidates, top_k=top_k)
        
        return final_results
