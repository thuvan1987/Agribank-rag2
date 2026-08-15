#!/usr/bin/env python3
"""
scripts/hybrid_search.py — Buổi 14
Mục đích: Thực thi truy vấn Hybrid Search kết hợp BM25 + Dense thông qua thuật toán Reciprocal Rank Fusion (RRF).

Sử dụng:
  python scripts/hybrid_search.py --query "Quy định về bảo quản tiền mặt theo Thông tư 01/2014" --candidate-k 20 --top-k 5
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.hybrid_retriever import HybridRetriever

CORPUS_CSV = BASE_DIR / "data" / "processed" / "chunks_normalized.csv"

def print_hybrid_table(results: list):
    print("\n====================================================================================================")
    print("HYBRID RESULTS (Reciprocal Rank Fusion - RRF)")
    print("====================================================================================================")
    print(f"{'Rank':<5} | {'Chunk ID':<18} | {'BM25 Rank':<10} | {'Dense Rank':<10} | {'RRF Score':<10} | {'Citation'}")
    print("-" * 100)

    if not results:
        print("Không tìm thấy kết quả phù hợp.")
        return

    for res in results:
        rank = res["final_rank"]
        cid = res["chunk_id"]
        b_rank = str(res["bm25_rank"])
        d_rank = str(res["dense_rank"])
        score = f"{res['rrf_score']:.6f}"
        citation = res["citation"]
        if len(citation) > 40:
            citation = citation[:37] + "..."

        print(f"{rank:<5} | {cid:<18} | {b_rank:<10} | {d_rank:<10} | {score:<10} | {citation}")

    print("-" * 100)

    print("\n--- CHI TIẾT NỘI DUNG TOP CANDIDATES ---")
    for res in results:
        rank = res["final_rank"]
        cid = res["chunk_id"]
        citation = res["citation"]
        snippet = res["text"].replace("\n", " ")[:150]
        print(f"\n[Hybrid Rank {rank}] (BM25: {res['bm25_rank']} | Dense: {res['dense_rank']} | RRF: {res['rrf_score']:.6f})")
        print(f"  Citation: {citation}")
        print(f"  Chunk ID: {cid}")
        print(f"  Snippet:  {snippet}...")

def main():
    parser = argparse.ArgumentParser(description="Buổi 14 — Hybrid Search (BM25 + Dense RRF Fusion)")
    parser.add_argument("--query", "-q", type=str, required=True, help="Câu hỏi/truy vấn tra cứu")
    parser.add_argument("--candidate-k", "-ck", type=int, default=20, help="Số lượng ứng viên lấy từ mỗi retriever (mặc định: 20)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Số lượng kết quả cuối cùng (mặc định: 5)")
    args = parser.parse_args()

    if not CORPUS_CSV.exists():
        print(f"❌ LỖI: Không tìm thấy file corpus chuẩn hóa tại '{CORPUS_CSV}'.")
        sys.exit(1)

    df = pd.read_csv(CORPUS_CSV, encoding="utf-8").fillna("")
    chunks = df.to_dict(orient="records")

    print(f"ℹ️  Đã nạp {len(chunks)} chunks từ corpus chuẩn hóa.")
    print("⏳ Đang thực thi Hybrid Search (BM25 + Dense RRF Fusion)...")

    hybrid_retriever = HybridRetriever(chunks)
    hybrid_hits = hybrid_retriever.search(
        question=args.query,
        candidate_k=args.candidate_k,
        top_k=args.top_k
    )

    print(f"\nTRUY VẤN: \"{args.query}\"")
    print_hybrid_table(hybrid_hits)

if __name__ == "__main__":
    main()
