import pandas as pd
from typing import List, Dict, Any, Optional
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever

class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        rrf_k: int = 60
    ):
        self.chunks = chunks
        self.chunks_by_id = {str(c["chunk_id"]).strip(): c for c in chunks}
        self.rrf_k = rrf_k

        self.bm25_retriever = bm25_retriever if bm25_retriever is not None else BM25Retriever(chunks)
        self.dense_retriever = dense_retriever if dense_retriever is not None else DenseRetriever(chunks)

    def search(
        self,
        question: str,
        candidate_k: int = 20,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not question.strip():
            return []

        # 1. Lấy kết quả Top Candidate-K từ BM25 và Dense
        bm25_hits = self.bm25_retriever.search(question, top_k=candidate_k)
        dense_hits = self.dense_retriever.search(question, top_k=candidate_k)

        # 2. Xây dựng bản đồ Rank cho từng Chunk ID
        bm25_rank_map = {h["chunk_id"]: h["rank"] for h in bm25_hits}
        dense_rank_map = {h["chunk_id"]: h["rank"] for h in dense_hits}

        # Tập hợp tất cả các candidate IDs duy nhất xuất hiện ở ít nhất 1 retriever
        all_candidate_ids = set(bm25_rank_map.keys()) | set(dense_rank_map.keys())

        # 3. Tính điểm Reciprocal Rank Fusion (RRF)
        scored_candidates = []
        for cid in all_candidate_ids:
            b_rank = bm25_rank_map.get(cid)
            d_rank = dense_rank_map.get(cid)

            rrf_score = 0.0
            if b_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + b_rank)
            if d_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + d_rank)

            c_info = self.chunks_by_id.get(cid, {})

            scored_candidates.append({
                "chunk_id": cid,
                "document_id": c_info.get("document_id", ""),
                "bm25_rank": b_rank if b_rank is not None else "N/A",
                "dense_rank": d_rank if d_rank is not None else "N/A",
                "rrf_score": rrf_score,
                "text": c_info.get("text", ""),
                "citation": c_info.get("citation", f"[{c_info.get('title', '')} | {c_info.get('article', '')} | {cid}]"),
                "title": c_info.get("title", ""),
                "article": c_info.get("article", ""),
                "retrieval_method": "hybrid_rrf"
            })

        # 4. Sắp xếp giảm dần theo điểm rrf_score
        scored_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 5. Gán final_rank và cắt top_k
        results = []
        for rank, cand in enumerate(scored_candidates[:top_k], start=1):
            cand_copy = dict(cand)
            cand_copy["final_rank"] = rank
            cand_copy["rrf_score"] = round(cand_copy["rrf_score"], 6)
            results.append(cand_copy)

        return results
