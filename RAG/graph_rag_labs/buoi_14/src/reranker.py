import logging
import time
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'):
        """
        Khởi tạo Cross-Encoder Reranker.
        Ưu tiên dùng model multilingual phù hợp tiếng Việt.
        """
        self.model_name = model_name
        logger.info(f"[Reranker] Đang tải mô hình Cross-Encoder: '{model_name}'...")
        start_time = time.time()
        
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info(f"[Reranker] ✅ Tải mô hình thành công sau {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"[Reranker] ❌ Lỗi khi tải mô hình reranker: {e}")
            self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank danh sách candidates dựa trên truy vấn.
        """
        if not candidates:
            return []
            
        if self.model is None:
            logger.warning("[Reranker] FALLBACK: Mô hình không khả dụng, trả về danh sách gốc.")
            return candidates[:top_k]

        # Chuẩn bị input cho CrossEncoder: List of (query, document)
        pairs = [[query, doc.get("text", "")] for doc in candidates]
        
        # Dự đoán điểm
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"[Reranker] ❌ Lỗi khi rerank: {e}")
            logger.warning("[Reranker] FALLBACK: Trả về danh sách gốc.")
            return candidates[:top_k]
            
        # Gắn điểm vào candidates và sắp xếp
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])
            
        # Sắp xếp giảm dần theo điểm rerank
        reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        # Cập nhật rank mới
        for rank, doc in enumerate(reranked_candidates):
            doc["final_rank"] = rank + 1
            
        return reranked_candidates[:top_k]
