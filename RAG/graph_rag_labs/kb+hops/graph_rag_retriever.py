import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
MODEL_NAME = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"


class GraphRAGRetriever:
    """
    Module hỗ trợ Truy vấn Vector và Mối quan hệ Đa bước (Multi-hop Graph RAG).
    Nhiệm vụ:
    1. Chuyển đổi câu hỏi thành Vector Embedding (dùng MSMARCO v5).
    2. Truy vấn Vector Similarity Top-K chunks trong Neo4j (nhãn :Chunk).
    3. Mở rộng đồ thị N-hops từ các tài liệu nguồn (:Document) qua các mối quan hệ:
       `CAN_CU`, `THAY_THE`, `HOP_NHAT`, `SUA_DOI_BO_SUNG`, `VAN_BAN_BO_SUNG`.
    4. Định dạng Ngữ cảnh (Context) cấu trúc phục vụ cho LLM.
    """

    def __init__(self, env_path: Optional[Path] = None, model_name: str = MODEL_NAME):
        self.env_path = env_path or ENV_FILE
        if self.env_path.exists():
            load_dotenv(dotenv_path=self.env_path, override=True)

        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
        self.username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
        self.password = os.getenv("NEO4J_PASSWORD", "password").strip()
        self.database = os.getenv("NEO4J_DATABASE", "kb-hops").strip()

        # Khởi tạo kết nối Neo4j
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self.driver.verify_connectivity()

        # Nạp mô hình nhúng văn bản
        print(f"✓ Đang nạp mô hình SentenceTransformer local: '{model_name}'...")
        self.model = SentenceTransformer(model_name, device="cpu")
        print(f"✓ Khởi tạo GraphRAGRetriever thành công (DB: '{self.database}').")

    def close(self):
        if self.driver:
            self.driver.close()

    def encode_query(self, query: str) -> List[float]:
        """Chuyển câu hỏi người dùng thành vector 384 chiều đã chuẩn hóa."""
        vec = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vec.tolist()

    def search_vector_chunks(self, query_vec: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm Vector similarity trong Neo4j để tìm top-k chunks khớp nhất.
        """
        cypher = """
        CALL db.index.vector.queryNodes("chunk_vector_index", $top_k, $query_vec)
        YIELD node, score
        MATCH (node)-[:PART_OF]->(doc:Document)
        RETURN
            node.chunk_id AS chunk_id,
            node.clean_text AS text,
            node.title AS chunk_title,
            node.level AS level,
            doc.id AS doc_id,
            doc.title AS doc_title,
            doc.so_ky_hieu AS so_ky_hieu,
            score
        ORDER BY score DESC
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, query_vec=query_vec, top_k=top_k)
            return [dict(r) for r in result]

    def expand_multi_hop(
        self,
        seed_doc_ids: List[str],
        n_hops: int = 1,
        max_chunks_per_related_doc: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Mở rộng Đa bước (Multi-hop) từ các tài liệu gốc (seed documents)
        qua các mối quan hệ liên kết pháp lý (CAN_CU, THAY_THE, HOP_NHAT, SUA_DOI_BO_SUNG, VAN_BAN_BO_SUNG).
        """
        if n_hops <= 0 or not seed_doc_ids:
            return []

        cypher = """
        UNWIND $seed_doc_ids AS seed_id
        MATCH (seedDoc:Document {id: seed_id})
        MATCH path = (seedDoc)-[r:CAN_CU|THAY_THE|HOP_NHAT|SUA_DOI_BO_SUNG|VAN_BAN_BO_SUNG*1..%d]-(relatedDoc:Document)
        WHERE relatedDoc.id <> seedDoc.id
        WITH DISTINCT seedDoc, relatedDoc, [rel IN relationships(path) | type(rel)] AS rel_types, length(path) AS hop_count
        OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(relatedDoc)
        WITH seedDoc, relatedDoc, rel_types, hop_count, collect(c)[..$max_chunks] AS related_chunks
        RETURN
            seedDoc.id AS seed_doc_id,
            seedDoc.title AS seed_doc_title,
            seedDoc.so_ky_hieu AS seed_so_ky_hieu,
            relatedDoc.id AS related_doc_id,
            relatedDoc.title AS related_doc_title,
            relatedDoc.so_ky_hieu AS related_so_ky_hieu,
            relatedDoc.loai_van_ban AS related_doc_type,
            relatedDoc.tinh_trang_hieu_luc AS related_doc_status,
            rel_types,
            hop_count,
            [c IN related_chunks | {
                chunk_id: c.chunk_id,
                title: c.title,
                level: c.level,
                text: c.clean_text
            }] AS chunks
        ORDER BY hop_count ASC, relatedDoc.id ASC
        """ % n_hops

        with self.driver.session(database=self.database) as session:
            result = session.run(
                cypher,
                seed_doc_ids=seed_doc_ids,
                max_chunks=max_chunks_per_related_doc
            )
            return [dict(r) for r in result]

    def search_documents_by_keyword(self, query: str) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bổ sung các nút :Document nếu câu hỏi chứa số ký hiệu (ví dụ: 41/2016, 01/2014, 52/VBHN, 46/2023, 01/2025).
        """
        import re
        pattern = r"\b\d+/(?:\d{4}/TT-NHNN|\d{4}/NĐ-CP|\d{4}/TT-BTC|VBHN-NHNN|\d{4}|VBHN)\b"
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        if not matches:
            pattern2 = r"\b\d+/\d{4}\b"
            matches = re.findall(pattern2, query)

        if not matches:
            return []

        cypher = """
        UNWIND $keywords AS kw
        MATCH (d:Document)
        WHERE toLower(d.so_ky_hieu) CONTAINS toLower(kw) OR toLower(d.title) CONTAINS toLower(kw)
        OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
        WITH d, collect(c)[..2] AS chunks
        RETURN
            d.id AS doc_id,
            d.title AS doc_title,
            d.so_ky_hieu AS so_ky_hieu,
            [c IN chunks | {chunk_id: c.chunk_id, title: c.title, text: c.clean_text}] AS sample_chunks
        """
        with self.driver.session(database=self.database) as session:
            res = session.run(cypher, keywords=matches)
            return [dict(r) for r in res]

    def get_multi_hop_context(
        self,
        query: str,
        top_k: int = 3,
        n_hops: int = 1,
        max_chunks_per_related_doc: int = 2
    ) -> Dict[str, Any]:
        """
        Tổng hợp Ngữ cảnh RAG Đa bước (Multi-hop Context Builder).
        Kết hợp:
        1. Top-K vector chunks khớp trực tiếp từ câu hỏi.
        2. Tìm kiếm theo Số ký hiệu văn bản nếu câu hỏi có nhắc tới.
        3. Mở rộng N bước nhảy từ các tài liệu gốc tới các tài liệu liên quan.
        4. Chuẩn hóa & đóng gói văn bản ngữ cảnh dạng Markdown hoàn chỉnh cho LLM.
        """
        # 1. Embed câu hỏi
        query_vec = self.encode_query(query)

        # 2. Tìm kiếm Vector Chunks gốc
        seed_chunks = self.search_vector_chunks(query_vec, top_k=top_k)

        # 3. Thu thập danh sách ID các tài liệu gốc
        seed_doc_ids: List[str] = []
        seed_docs_info: Dict[str, Dict[str, str]] = {}
        for sc in seed_chunks:
            did = str(sc.get("doc_id", ""))
            if did and did not in seed_docs_info:
                seed_doc_ids.append(did)
                seed_docs_info[did] = {
                    "doc_id": did,
                    "title": sc.get("doc_title", ""),
                    "so_ky_hieu": sc.get("so_ky_hieu", "")
                }

        # Bổ sung các tài liệu khớp theo Số ký hiệu trong câu hỏi
        matched_docs = self.search_documents_by_keyword(query)
        for md in matched_docs:
            did = str(md.get("doc_id", ""))
            if did and did not in seed_docs_info:
                seed_doc_ids.append(did)
                seed_docs_info[did] = {
                    "doc_id": did,
                    "title": md.get("doc_title", ""),
                    "so_ky_hieu": md.get("so_ky_hieu", "")
                }

        # 4. Truy vấn mở rộng Đa bước (Multi-hop) nếu n_hops > 0
        hop_results = []
        if n_hops > 0 and seed_doc_ids:
            hop_results = self.expand_multi_hop(
                seed_doc_ids=seed_doc_ids,
                n_hops=n_hops,
                max_chunks_per_related_doc=max_chunks_per_related_doc
            )

        # 5. Định dạng Ngữ cảnh Markdown (Formatted Context)
        context_blocks = []
        context_blocks.append(f"### [I] NGỮ CẢNH TRỰC TIẾP TỪ VECTOR SEARCH (Top-{top_k} Chunks Khớp Nhất)\n")

        for idx, sc in enumerate(seed_chunks, start=1):
            score_pct = sc["score"] * 100
            block = (
                f"--- Vector Chunk #{idx} (Score: {score_pct:.1f}%) ---\n"
                f"• Tài liệu: {sc['doc_title']} (Số hiệu: {sc['so_ky_hieu']}, ID: {sc['doc_id']})\n"
                f"• Chunk ID: {sc['chunk_id']} | Tiêu đề đoạn: {sc['chunk_title']}\n"
                f"• Nội dung:\n{sc['text']}\n"
            )
            context_blocks.append(block)

        if n_hops > 0:
            context_blocks.append(f"\n### [II] MỞ RỘNG MỐI QUAN HỆ ĐỒ THỊ ĐA BƯỚC ({n_hops}-Hop Expansion)\n")
            if not hop_results:
                context_blocks.append("*(Không tìm thấy liên kết tài liệu liên quan nào trong đồ thị)*\n")
            else:
                seen_rel_pairs: Set[Tuple[str, str]] = set()
                for idx, hr in enumerate(hop_results, start=1):
                    pair_key = (hr["seed_doc_id"], hr["related_doc_id"])
                    if pair_key in seen_rel_pairs:
                        continue
                    seen_rel_pairs.add(pair_key)

                    rels_str = " -> ".join(hr["rel_types"])
                    block = (
                        f"--- Liên kết Đồ thị #{idx} ({hr['hop_count']}-Hop) ---\n"
                        f"• Từ Tài liệu Gốc: [{hr['seed_so_ky_hieu']}] (ID: {hr['seed_doc_id']})\n"
                        f"• Mối quan hệ: --[{rels_str}]-->\n"
                        f"• Tới Tài liệu Liên quan: [{hr['related_so_ky_hieu']}] - {hr['related_doc_title']}\n"
                        f"  (Loại: {hr['related_doc_type']}, Tình trạng: {hr['related_doc_status']})\n"
                    )

                    r_chunks = hr.get("chunks", [])
                    if r_chunks:
                        block += "  • Các đoạn trích liên quan từ tài liệu này:\n"
                        for rc in r_chunks:
                            txt_sample = rc['text'][:300].replace("\n", " ")
                            block += f"    - [{rc['chunk_id']} - {rc['title']}]: {txt_sample}...\n"
                    else:
                        block += "  • (Tài liệu này chưa được gán chunk nội dung)\n"

                    context_blocks.append(block)

        formatted_context = "\n".join(context_blocks)

        return {
            "query": query,
            "top_k": top_k,
            "n_hops": n_hops,
            "seed_chunks": seed_chunks,
            "seed_documents": list(seed_docs_info.values()),
            "multi_hop_relationships": hop_results,
            "formatted_context": formatted_context
        }


def main():
    print("==================================================")
    print(" BẮT ĐẦU KIỂM THỬ BƯỚC 2 - MULTI-HOP GRAPH RETRIEVAL ")
    print("==================================================")

    retriever = GraphRAGRetriever()

    # Thử nghiệm câu hỏi mẫu 1 trong Buổi 11
    sample_query = "Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?"

    print(f"\n❓ CÂU HỎI THỬ NGHIỆM: '{sample_query}'\n")

    print("--- 1. CHẠY TRUY VẤN VỚI 0-HOP (CHỈ VECTOR SEARCH) ---")
    ctx_0hop = retriever.get_multi_hop_context(sample_query, top_k=2, n_hops=0)
    print(ctx_0hop["formatted_context"])

    print("\n==================================================\n")

    print("--- 2. CHẠY TRUY VẤN VỚI 1-HOP (VECTOR + ĐỒ THỊ BƯỚC NHẢY 1) ---")
    ctx_1hop = retriever.get_multi_hop_context(sample_query, top_k=2, n_hops=1)
    print(ctx_1hop["formatted_context"])

    retriever.close()


if __name__ == "__main__":
    main()
