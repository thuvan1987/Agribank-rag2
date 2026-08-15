#!/usr/bin/env python3
"""
scripts/baseline_retrieval.py — Buổi 14
Mục đích: Thực thi truy vấn thử nghiệm độc lập cho 2 baseline:
  1. BM25-only Retrieval
  2. Dense-only Retrieval

Sử dụng:
  python scripts/baseline_retrieval.py --query "Thủ tục giao nhận tiền mặt theo Thông tư 01/2014" --top-k 5
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Thêm thư mục root buoi_14 vào sys.path để import src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever

CORPUS_CSV = BASE_DIR / "data" / "processed" / "chunks_normalized.csv"

def print_results(header: str, results: list):
    print(f"\n========================================")
    print(header)
    print(f"========================================")

    if not results:
        print("Không tìm thấy kết quả phù hợp.")
        return

    for res in results:
        rank = res["rank"]
        score = res["retrieval_score"]
        method = res["retrieval_method"].upper()
        citation = res["citation"]
        cid = res["chunk_id"]
        text_snippet = res["text"].replace("\n", " ")[:150]

        print(f"\n[{method} Rank {rank}] Score: {score:.4f}")
        print(f"  Citation: {citation}")
        print(f"  Chunk ID: {cid}")
        print(f"  Snippet:  {text_snippet}...")

def main():
    parser = argparse.ArgumentParser(description="Buổi 14 — Baseline Retrieval CLI (BM25 vs Dense)")
    parser.add_argument("--query", "-q", type=str, required=True, help="Câu hỏi/truy vấn tra cứu")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Số lượng kết quả trả về (mặc định: 5)")
    args = parser.parse_args()

    if not CORPUS_CSV.exists():
        print(f"❌ LỖI: Không tìm thấy file corpus chuẩn hóa tại '{CORPUS_CSV}'. Hãy chạy 'python scripts/prepare_corpus.py' trước.")
        sys.exit(1)

    # Đọc corpus chunks
    df = pd.read_csv(CORPUS_CSV, encoding="utf-8").fillna("")
    chunks = df.to_dict(orient="records")
    print(f"ℹ️  Đã nạp {len(chunks)} chunks từ corpus chuẩn hóa.")

    # 1. Khởi tạo & Chạy BM25 Retrieval
    print("\n⏳ Đang thực thi BM25 Search...")
    bm25_retriever = BM25Retriever(chunks)
    bm25_hits = bm25_retriever.search(args.query, top_k=args.top_k)

    # 2. Khởi tạo & Chạy Dense Retrieval
    print("⏳ Đang thực thi Dense Search...")
    dense_retriever = DenseRetriever(chunks)
    dense_hits = dense_retriever.search(args.query, top_k=args.top_k)

    # 3. In kết quả
    print(f"\nTRUY VẤN: \"{args.query}\"")
    print_results("BM25 RESULTS", bm25_hits)
    print_results("DENSE RESULTS", dense_hits)

if __name__ == "__main__":
    main()
