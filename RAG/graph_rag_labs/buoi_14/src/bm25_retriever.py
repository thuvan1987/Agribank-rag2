import pandas as pd
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.tokenizer import tokenize_legal_text

def build_citation(chunk: Dict[str, Any]) -> str:
    """Tạo trích dẫn citation chuẩn từ metadata thực."""
    title = str(chunk.get("title", "")).strip()
    so_kh = str(chunk.get("so_ky_hieu", "")).strip()
    art = str(chunk.get("article", "")).strip()
    cid = str(chunk.get("chunk_id", "")).strip()

    name_part = so_kh if so_kh else title
    if len(name_part) > 60:
        name_part = name_part[:60] + "..."

    parts = [p for p in [name_part, art, cid] if p]
    return f"[{' | '.join(parts)}]"

class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = []
        for c in self.chunks:
            # Gộp cả title, article và text để BM25 truy xuất chính xác tên văn bản & điều khoản
            full_content = f"{c.get('title', '')} {c.get('so_ky_hieu', '')} {c.get('article', '')} {c.get('text', '')}"
            tokens = tokenize_legal_text(full_content)
            self.corpus_tokens.append(tokens)

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_tokens = tokenize_legal_text(question)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Sắp xếp giảm dần theo điểm số
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            c = self.chunks[idx]
            score = float(scores[idx])
            results.append({
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "text": c["text"],
                "retrieval_score": round(score, 4),
                "retrieval_method": "bm25",
                "citation": build_citation(c),
                "title": c.get("title", ""),
                "article": c.get("article", "")
            })
        return results
