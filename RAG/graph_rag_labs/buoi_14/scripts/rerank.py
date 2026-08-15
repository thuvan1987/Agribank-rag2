import sys
import os
import argparse
import pandas as pd
import logging

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test Reranking Pipeline")
    parser.add_argument('--query', type=str, required=True, help="Câu truy vấn")
    parser.add_argument('--candidate-k', type=int, default=20, help="Số lượng candidates từ Hybrid Search")
    parser.add_argument('--top-k', type=int, default=5, help="Số lượng kết quả cuối cùng sau khi rerank")
    parser.add_argument('--corpus', type=str, default="data/processed/chunks_normalized.csv")
    args = parser.parse_args()

    logger.info("=== BẮT ĐẦU PIPELINE RERANKING ===")
    
    # 1. Đọc corpus
    try:
        df = pd.read_csv(args.corpus)
        chunks = df.to_dict('records')
        logger.info(f"ℹ️  Đã nạp {len(chunks)} chunks từ {args.corpus}")
    except Exception as e:
        logger.error(f"❌ Lỗi đọc corpus: {e}")
        return

    # 2. Khởi tạo Hybrid Retriever
    logger.info("⏳ Đang khởi tạo Hybrid Retriever (BM25 + Dense)...")
    hybrid = HybridRetriever(chunks=chunks)

    # 3. Lấy Candidates từ Hybrid
    logger.info(f"\n🔍 Thực thi Truy vấn: \"{args.query}\"")
    logger.info(f"Lấy Top {args.candidate_k} candidates từ Hybrid Search...")
    hybrid_candidates = hybrid.search(question=args.query, candidate_k=args.candidate_k, top_k=args.candidate_k)

    if not hybrid_candidates:
        logger.warning("Không tìm thấy candidates nào.")
        return

    print("\n" + "="*50)
    print(f"BEFORE RERANK (Top {args.top_k} from {args.candidate_k} Candidates)")
    print("="*50)
    
    # In kết quả trước khi rerank
    print(f"{'Rank':<5} | {'Chunk ID':<30} | {'Hybrid Score':<15} | {'Citation'}")
    print("-" * 80)
    for i, res in enumerate(hybrid_candidates[:args.top_k]):
        citation = res.get('citation', '')
        # cắt bớt citation nếu quá dài
        if len(citation) > 40:
            citation = citation[:37] + "..."
        print(f"{i+1:<5} | {res['chunk_id']:<30} | {res['rrf_score']:<15.4f} | {citation}")

    # 4. Khởi tạo Reranker
    reranker = Reranker()

    # 5. Rerank Candidates
    logger.info(f"\n⏳ Đang Rerank Top {args.candidate_k} candidates...")
    reranked_results = reranker.rerank(query=args.query, candidates=hybrid_candidates, top_k=args.top_k)

    print("\n" + "="*50)
    print(f"AFTER RERANK (Top {args.top_k})")
    print("="*50)
    
    print(f"{'Rank':<5} | {'Chunk ID':<30} | {'Rerank Score':<15} | {'Citation'}")
    print("-" * 80)
    for res in reranked_results:
        citation = res.get('citation', '')
        if len(citation) > 40:
            citation = citation[:37] + "..."
        # Lấy hybrid_score ban đầu từ res['rrf_score']
        print(f"{res['final_rank']:<5} | {res['chunk_id']:<30} | {res.get('rerank_score', 0.0):<15.4f} | {citation}")

if __name__ == "__main__":
    main()
