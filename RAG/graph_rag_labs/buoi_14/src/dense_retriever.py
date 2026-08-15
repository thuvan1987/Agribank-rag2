import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.bm25_retriever import build_citation

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "dense_embeddings.pkl"

DEFAULT_MODEL_NAME = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"

class DenseRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        model_name: str = DEFAULT_MODEL_NAME,
        cache_file: Path = CACHE_FILE
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.cache_file = cache_file

        print(f"[DenseRetriever] Khởi tạo mô hình Embedding: '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        self.doc_embeddings = self._get_or_create_embeddings()

    def _get_or_create_embeddings(self) -> np.ndarray:
        """Đọc vector embeddings từ cache nếu có; nếu chưa có thì encode và lưu cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("chunks_count") == len(self.chunks) and cached_data.get("model_name") == self.model_name:
                    print(f"[DenseRetriever] ✅ Đã nạp thành công {len(self.chunks)} vector embeddings từ cache: {self.cache_file}")
                    return cached_data["embeddings"]
            except Exception as e:
                print(f"[DenseRetriever] ⚠️ Lỗi đọc cache ({e}), tiến hành encode lại...")

        print(f"[DenseRetriever] ⏳ Đang tính toán vector embeddings cho {len(self.chunks)} chunks văn bản...")
        texts_to_encode = []
        for c in self.chunks:
            # Gộp title, article và content để đại diện đầy đủ thông tin ngữ nghĩa
            combined_text = f"{c.get('title', '')} {c.get('article', '')}\n{c.get('text', '')}"
            texts_to_encode.append(combined_text)

        embeddings = self.model.encode(
            texts_to_encode,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        with open(self.cache_file, "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "chunks_count": len(self.chunks),
                "embeddings": embeddings
            }, f)
        print(f"[DenseRetriever] ✅ Đã lưu cache embeddings tại: {self.cache_file}")

        return embeddings

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not question.strip():
            return []

        # Encode query vector
        query_vec = self.model.encode(question, convert_to_numpy=True, normalize_embeddings=True)

        # Tính Cosine Similarity (vì vector đã normalize nên dot product = cosine similarity)
        scores = np.dot(self.doc_embeddings, query_vec)

        # Sắp xếp top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

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
                "retrieval_method": "dense",
                "citation": build_citation(c),
                "title": c.get("title", ""),
                "article": c.get("article", "")
            })

        return results
