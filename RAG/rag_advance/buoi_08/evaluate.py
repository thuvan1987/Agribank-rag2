"""
Buổi 08: Module Đánh giá Retrieval Quality (Recall@K, MRR@K, nDCG@K & Latency Benchmarks).

Module này đo đạc và so sánh hiệu năng của 4 chế độ retrieval trên tập câu hỏi benchmark,
tính toán các chỉ số chất lượng tìm kiếm và xuất báo cáo JSON định dạng chuẩn.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import advanced_rag
import rag


def calculate_recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """Tính chỉ số Recall@K."""
    if not isinstance(retrieved_ids, list) or not isinstance(gold_ids, list):
        raise TypeError("retrieved_ids và gold_ids phải là danh sách (list).")
    if k <= 0:
        raise ValueError(f"Tham số k ({k}) phải là số nguyên dương > 0.")
    if not gold_ids:
        return 0.0

    ret_k = set(retrieved_ids[:k])
    gold = set(gold_ids)
    return len(ret_k & gold) / len(gold)


def calculate_mrr_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """Tính chỉ số Mean Reciprocal Rank (MRR@K)."""
    if not isinstance(retrieved_ids, list) or not isinstance(gold_ids, list):
        raise TypeError("retrieved_ids và gold_ids phải là danh sách (list).")
    if k <= 0:
        raise ValueError(f"Tham số k ({k}) phải là số nguyên dương > 0.")
    if not gold_ids:
        return 0.0

    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """Tính chỉ số Normalized Discounted Cumulative Gain (nDCG@K) với binary relevance."""
    if not isinstance(retrieved_ids, list) or not isinstance(gold_ids, list):
        raise TypeError("retrieved_ids và gold_ids phải là danh sách (list).")
    if k <= 0:
        raise ValueError(f"Tham số k ({k}) phải là số nguyên dương > 0.")
    if not gold_ids:
        return 0.0

    gold = set(gold_ids)
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if cid in gold else 0.0
        dcg += rel / math.log2(rank + 1)

    idcg = 0.0
    ideal_hits = min(k, len(gold))
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    return dcg / idcg if idcg > 0.0 else 0.0


def run_evaluation_benchmark(
    questions_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    strategy: str = "hierarchical",
    k: int = 5,
    modes: Optional[List[str]] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    reranker_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Chạy đánh giá benchmark trên cả 4 retrieval mode và xuất báo cáo JSON.
    ĐẢM BẢO KHÔNG GỌI LLM GENERATION TRONG QUÁ TRÌNH BENCHMARK.
    """
    if questions_path is None:
        q_file = BASE_DIR / "eval" / "questions.json"
    else:
        q_file = Path(questions_path)

    if not q_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi benchmark: '{q_file}'")

    with open(q_file, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)

    if not isinstance(eval_questions, list) or not eval_questions:
        raise ValueError("File câu hỏi benchmark rỗng hoặc không đúng cấu trúc danh sách JSON.")

    if modes is None:
        modes = ["bm25", "semantic", "hybrid", "hybrid_rerank"]

    cfg = advanced_rag.get_advanced_config(custom_config)
    norm_strat = rag.normalize_strategy(strategy)

    # Kiểm tra xem có câu hỏi nào cần duyêt thủ công không
    needs_human_review = any(item.get("needs_human_review", False) for item in eval_questions)

    mode_metrics: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        recalls = []
        mrrs = []
        ndcgs = []
        latencies = []
        query_results = []

        for q_item in eval_questions:
            qid = q_item.get("query_id", "Q_UNK")
            q_text = q_item.get("question", "")
            gold_ids = q_item.get("relevant_chunk_ids", [])

            t0 = time.perf_counter()
            retrieved_cands = []
            error_msg = None

            try:
                if mode == "bm25":
                    chunks, _ = rag.load_chunks(strategy=norm_strat)
                    retrieved_cands = advanced_rag.bm25_retrieval(query=q_text, chunks=chunks, top_k=max(20, k))
                elif mode == "semantic":
                    retrieved_cands = advanced_rag.semantic_retrieval(query=q_text, strategy=norm_strat, top_k=max(20, k), custom_config=cfg)
                elif mode == "hybrid":
                    hyb = advanced_rag.hybrid_retrieval(question=q_text, strategy=norm_strat, custom_config=cfg)
                    retrieved_cands = hyb["candidates"]
                elif mode == "hybrid_rerank":
                    hyb = advanced_rag.hybrid_retrieval(question=q_text, strategy=norm_strat, custom_config=cfg)
                    retrieved_cands = advanced_rag.rerank_candidates(
                        query=q_text,
                        candidates=hyb["candidates"],
                        top_k=k,
                        custom_config=cfg,
                        reranker_fn=reranker_fn
                    )
            except Exception as e:
                error_msg = str(e)

            t_lat = (time.perf_counter() - t0) * 1000.0
            retrieved_ids = [c["chunk_id"] for c in retrieved_cands]

            rec_val = calculate_recall_at_k(retrieved_ids, gold_ids, k)
            mrr_val = calculate_mrr_at_k(retrieved_ids, gold_ids, k)
            ndcg_val = calculate_ndcg_at_k(retrieved_ids, gold_ids, k)

            recalls.append(rec_val)
            mrrs.append(mrr_val)
            ndcgs.append(ndcg_val)
            latencies.append(t_lat)

            query_results.append({
                "query_id": qid,
                "question": q_text,
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved_ids[:k],
                "recall": round(rec_val, 4),
                "mrr": round(mrr_val, 4),
                "ndcg": round(ndcg_val, 4),
                "latency_ms": round(t_lat, 2),
                "error": error_msg
            })

        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2] if n > 0 else 0.0
        mean_lat = sum(latencies) / n if n > 0 else 0.0

        mode_metrics[mode] = {
            "recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "ndcg": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            "latency_mean": round(mean_lat, 2),
            "latency_p50": round(p50, 2),
            "query_details": query_results
        }

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "strategy": norm_strat,
        "eval_k": k,
        "total_questions": len(eval_questions),
        "needs_human_review_warning": needs_human_review,
        "config": {
            "embedding_model": cfg["embedding_model"],
            "reranker_model": cfg["reranker_model"],
            "rerank_min_score": cfg["rerank_min_score"]
        },
        "metrics_by_mode": mode_metrics
    }

    if output_dir is None:
        out_dir = BASE_DIR / "reports"
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_report_{norm_strat}_k{k}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Offline Evaluation Benchmark — Buổi 08")
    parser.add_argument("--questions", type=str, default=str(BASE_DIR / "eval" / "questions.json"), help="Đường dẫn file câu hỏi eval JSON")
    parser.add_argument("--strategy", type=str, default="hierarchical", help="Chiến lược chunking (hierarchical, flat)")
    parser.add_argument("--k", type=int, default=5, help="Số lượng Top K để đánh giá")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "reports"), help="Thư mục lưu báo cáo JSON")

    args = parser.parse_args()

    print(f"\n=== ĐANG THỰC THI EVALUATION BENCHMARK (BUỔI 08) ===")
    print(f"File câu hỏi: {args.questions}")
    print(f"Chiến lược:   {args.strategy}")
    print(f"Eval Top K:   {args.k}\n")

    try:
        report = run_evaluation_benchmark(
            questions_path=args.questions,
            output_dir=args.output_dir,
            strategy=args.strategy,
            k=args.k
        )

        print("=== KẾT QUẢ BÁO CÁO ĐÁNH GIÁ (EVALUATION REPORT) ===")
        print(f"Thời gian tạo báo cáo: {report['timestamp']}")
        print(f"Tổng số câu hỏi eval: {report['total_questions']}")
        if report['needs_human_review_warning']:
            print("\n⚠️  CẢNH BÁO: Tập dữ liệu Gold Test chứa câu hỏi cần duyệt thủ công (needs_human_review = True). Không tuyên bố mode chiến thắng chính thức.")

        print(f"\n{'MODE':<15} | {'RECALL@K':<10} | {'MRR@K':<10} | {'nDCG@K':<10} | {'LATENCY MEAN':<14} | {'LATENCY P50':<12}")
        print("-" * 80)
        for mode_name, m_val in report["metrics_by_mode"].items():
            print(f"{mode_name:<15} | {m_val['recall']:<10.4f} | {m_val['mrr']:<10.4f} | {m_val['ndcg']:<10.4f} | {m_val['latency_mean']:<14.2f}ms | {m_val['latency_p50']:<12.2f}ms")

        out_path = Path(args.output_dir) / f"eval_report_{report['strategy']}_k{args.k}.json"
        print(f"\nĐã xuất báo cáo chi tiết ra file: '{out_path}'")

    except Exception as e:
        print(f"\n[LỖI EVALUATION] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
