"""
Buổi 09: Module Đánh Giá Định Lượng Benchmark Multi-Query & Parent-Child RAG.

Đo đạc định lượng Child Recall@K, Parent Recall@K, MRR@K, nDCG@K và Latency
cho cả 4 chế độ Pipeline: single_flat, multi_flat, single_parent, multi_parent.
Xuất báo cáo nguyên tử (Atomic Write) ra thư mục reports/.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hierarchical_rag


def compute_dcg_at_k(retrieved_ids: List[str], rel_set: Set[str], k: int) -> float:
    """Tính DCG@K với binary relevance (0 hoặc 1)."""
    dcg = 0.0
    for idx, item_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if item_id in rel_set else 0.0
        dcg += rel / math.log2(idx + 1)
    return dcg


def compute_ndcg_at_k(retrieved_ids: List[str], rel_set: Set[str], k: int) -> float:
    """Tính nDCG@K với binary relevance."""
    if not rel_set:
        return 0.0
    dcg = compute_dcg_at_k(retrieved_ids, rel_set, k)
    # Ideal DCG: tất cả phần tử liên quan đứng ở vị trí đầu tiên
    ideal_hits = ["rel"] * min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx, _ in enumerate(ideal_hits, start=1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def compute_mrr_at_k(retrieved_ids: List[str], rel_set: Set[str], k: int) -> float:
    """Tính MRR@K (Mean Reciprocal Rank)."""
    if not rel_set:
        return 0.0
    for idx, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in rel_set:
            return round(1.0 / idx, 4)
    return 0.0


def run_evaluation_benchmark(
    questions_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    strategy: str = "hierarchical",
    k: int = 3,
    custom_config: Optional[Dict[str, Any]] = None,
    query_generator_fn: Optional[Any] = None,
    hybrid_retriever_fn: Optional[Any] = None,
    reranker_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Thực thi Benchmark Đánh Giá Định Lượng 4 Chế Độ Pipeline.
    Không gọi Gemini Answer Generation trong quá trình benchmark.
    """
    if questions_path is None:
        questions_path = str(BASE_DIR / "eval" / "questions.json")

    if output_dir is None:
        output_dir = str(BASE_DIR / "reports")

    q_file = Path(questions_path)
    if not q_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi đánh giá tại '{questions_path}'.")

    with open(q_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # 1. Verify hierarchy store readiness & integrity
    st_status = hierarchical_rag.hierarchy_status()
    if not st_status["store_exists"]:
        raise RuntimeError("Hierarchy store chưa tồn tại. Hãy chạy 'python hierarchical_rag.py build-hierarchy' trước.")

    children_reg, parents_reg, _ = hierarchical_rag.load_hierarchy_store()
    valid_child_ids = {c["child_id"] for c in children_reg}
    valid_parent_ids = {p["parent_id"] for p in parents_reg}

    # Validate ground truth IDs in dataset
    human_review_flag = False
    for q_item in questions_data:
        if q_item.get("needs_human_review"):
            human_review_flag = True
        for cid in q_item.get("relevant_child_ids", []):
            if cid not in valid_child_ids:
                raise ValueError(f"Ground truth child_id '{cid}' trong câu hỏi {q_item.get('question_id')} không tồn tại trong hierarchy store.")
        for pid in q_item.get("relevant_parent_ids", []):
            if pid not in valid_parent_ids:
                raise ValueError(f"Ground truth parent_id '{pid}' trong câu hỏi {q_item.get('question_id')} không tồn tại trong hierarchy store.")

    cfg = hierarchical_rag.get_hierarchical_config(custom_config)
    cfg["final_parent_top_k"] = k
    cfg["final_top_k"] = k

    modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
    mode_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    print(f"\n🚀 Đang thực thi Evaluation Benchmark ({len(questions_data)} câu hỏi, Top-K={k})...\n")

    for q_idx, q_item in enumerate(questions_data, start=1):
        q_id = q_item.get("question_id", f"Q{q_idx}")
        q_text = q_item["question"]
        rel_children = set(q_item.get("relevant_child_ids", []))
        rel_parents = set(q_item.get("relevant_parent_ids", []))

        print(f"[{q_idx}/{len(questions_data)}] Đánh giá {q_id}: '{q_text[:50]}...'")

        for mode in modes:
            t0 = time.perf_counter()

            # Retrieval-only query (without answer generation)
            if mode in ["single_parent", "multi_parent"]:
                parent_res = hierarchical_rag.retrieve_parent_candidates(
                    question=q_text,
                    mode=mode,
                    strategy=strategy,
                    custom_config=cfg,
                    query_generator_fn=query_generator_fn,
                    hybrid_retriever_fn=hybrid_retriever_fn
                )
                child_hits = parent_res.get("child_hits", [])
                raw_parents = parent_res.get("parent_candidates", [])
                if raw_parents and parent_res.get("status") == "ready":
                    try:
                        reranked_parents = hierarchical_rag.rerank_parent_candidates(
                            question=q_text,
                            parent_candidates=raw_parents,
                            custom_config=cfg,
                            reranker_fn=reranker_fn
                        )
                    except Exception:
                        reranked_parents = raw_parents
                else:
                    reranked_parents = []

                accepted_ev = [p for p in reranked_parents if p.get("parent_rerank_score", 0.0) >= cfg["rerank_min_score"]][:k]
                retrieved_parents = [p["parent_id"] for p in accepted_ev]

                # Retrieved child IDs inside accepted parents
                retrieved_children = []
                for p in accepted_ev:
                    retrieved_children.extend(p.get("supporting_child_ids", []))

                ctx_chars = sum(p.get("char_count", len(p.get("text", ""))) for p in accepted_ev)
                child_chars = sum(len(c.get("text", "")) for c in child_hits)
                exp_factor = round(ctx_chars / child_chars, 2) if child_chars > 0 else 1.0
                gen_calls = 1 if mode == "multi_parent" else 0
                emb_calls = (cfg["multi_query_count"] + 1) if mode == "multi_parent" else 1

            else:
                if mode == "multi_flat":
                    multi_child_res = hierarchical_rag.retrieve_multi_query_child_hits(
                        question=q_text,
                        strategy=strategy,
                        custom_config=cfg,
                        query_generator_fn=query_generator_fn,
                        hybrid_retriever_fn=hybrid_retriever_fn
                    )
                    child_hits = multi_child_res.get("child_hits", [])
                    gen_calls = 1
                    emb_calls = cfg["multi_query_count"] + 1
                else:
                    if hybrid_retriever_fn is not None:
                        h_res = hybrid_retriever_fn(q_text, strategy, cfg)
                    else:
                        import advanced_rag
                        h_res = advanced_rag.hybrid_retrieval(question=q_text, strategy=strategy, custom_config=cfg)
                    raw_cands = h_res.get("candidates", []) if isinstance(h_res, dict) else h_res
                    top_cands = raw_cands[:cfg["per_query_candidates"]]
                    child_hits = []
                    for idx, c in enumerate(top_cands, start=1):
                        child_hits.append({
                            "child_id": str(c["chunk_id"]).strip(),
                            "text": c["text"],
                            "source": c["source"],
                            "fused_rank": idx
                        })
                    gen_calls = 0
                    emb_calls = 1

                if child_hits:
                    try:
                        import advanced_rag
                        reranked_children = advanced_rag.rerank_candidates(
                            query=q_text,
                            candidates=child_hits,
                            top_k=k,
                            custom_config=cfg,
                            reranker_fn=reranker_fn
                        )
                    except Exception:
                        reranked_children = child_hits[:k]
                else:
                    reranked_children = []

                accepted_ev = [c for c in reranked_children if c.get("rerank_score", 0.0) >= cfg["rerank_min_score"]][:k]
                retrieved_children = [c.get("child_id", c.get("chunk_id", "")) for c in accepted_ev]

                # Map retrieved children to parent IDs
                child_to_parent_map = {c["child_id"]: c["parent_id"] for c in children_reg}
                retrieved_parents = []
                for cid in retrieved_children:
                    p_id = child_to_parent_map.get(cid)
                    if p_id:
                        retrieved_parents.append(p_id)

                ctx_chars = sum(len(c.get("text", "")) for c in accepted_ev)
                exp_factor = 1.0

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Calculate metrics
            child_recall = round(len(set(retrieved_children) & rel_children) / len(rel_children), 4) if rel_children else 1.0
            parent_recall = round(len(set(retrieved_parents) & rel_parents) / len(rel_parents), 4) if rel_parents else 1.0
            mrr_val = compute_mrr_at_k(retrieved_parents if "parent" in mode else retrieved_children, rel_parents if "parent" in mode else rel_children, k)
            ndcg_val = compute_ndcg_at_k(retrieved_parents if "parent" in mode else retrieved_children, rel_parents if "parent" in mode else rel_children, k)

            q_record = {
                "question_id": q_id,
                "question_type": q_item.get("question_type", "exact"),
                "child_recall_at_k": child_recall,
                "parent_recall_at_k": parent_recall,
                "mrr_at_k": mrr_val,
                "ndcg_at_k": ndcg_val,
                "latency_ms": latency_ms,
                "retrieved_child_count": len(child_hits),
                "accepted_evidence_count": len(accepted_ev),
                "context_chars": ctx_chars,
                "expansion_factor": exp_factor,
                "gen_api_calls": gen_calls,
                "emb_api_calls": emb_calls
            }
            mode_results[mode].append(q_record)

    # 2. Compute aggregate metrics per mode
    aggregate_metrics: Dict[str, Dict[str, Any]] = {}
    for mode, q_list in mode_results.items():
        if not q_list:
            continue
        n_q = len(q_list)
        avg_c_rec = round(sum(item["child_recall_at_k"] for item in q_list) / n_q, 4)
        avg_p_rec = round(sum(item["parent_recall_at_k"] for item in q_list) / n_q, 4)
        avg_mrr = round(sum(item["mrr_at_k"] for item in q_list) / n_q, 4)
        avg_ndcg = round(sum(item["ndcg_at_k"] for item in q_list) / n_q, 4)
        avg_lat = round(sum(item["latency_ms"] for item in q_list) / n_q, 2)
        avg_ctx = round(sum(item["context_chars"] for item in q_list) / n_q, 1)
        avg_exp = round(sum(item["expansion_factor"] for item in q_list) / n_q, 2)
        tot_gen = sum(item["gen_api_calls"] for item in q_list)
        tot_emb = sum(item["emb_api_calls"] for item in q_list)

        aggregate_metrics[mode] = {
            "child_recall_at_k": avg_c_rec,
            "parent_recall_at_k": avg_p_rec,
            "mrr_at_k": avg_mrr,
            "ndcg_at_k": avg_ndcg,
            "mean_latency_ms": avg_lat,
            "mean_context_chars": avg_ctx,
            "mean_expansion_factor": avg_exp,
            "total_gen_api_calls": tot_gen,
            "total_emb_api_calls": tot_emb
        }

    # 3. Build Atomic Report JSON
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rep_dir = Path(output_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    report_filename = f"report_{now_str}.json"
    target_path = rep_dir / report_filename
    latest_path = rep_dir / "latest_report.json"

    report_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "k": k,
        "total_questions": len(questions_data),
        "needs_human_review": human_review_flag,
        "warning": "Tập dữ liệu chứa các câu hỏi cần con người đánh giá (needs_human_review = True). Không tự động kết luận mode thắng tuyệt đối." if human_review_flag else None,
        "system_config": {
            "embedding_model": cfg["embedding_model"],
            "generation_model": cfg["generation_model"],
            "reranker_model": cfg["reranker_model"],
            "multi_query_count": cfg["multi_query_count"],
            "per_query_candidates": cfg["per_query_candidates"],
            "parent_candidates": cfg["parent_candidates"],
            "final_parent_top_k": cfg["final_parent_top_k"],
            "rerank_min_score": cfg["rerank_min_score"]
        },
        "aggregate_metrics": aggregate_metrics,
        "per_question_results": mode_results
    }

    # Atomic write to temp file then rename
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, target_path)

    # Update latest_report.json atomically
    tmp_latest = latest_path.with_suffix(".tmp")
    with open(tmp_latest, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_latest, latest_path)

    print(f"\n✅ Đã xuất báo cáo benchmark nguyên tử thành công:")
    print(f"   • File báo cáo:  {target_path}")
    print(f"   • Latest Report: {latest_path}\n")

    return report_payload


def main():
    parser = argparse.ArgumentParser(description="CLI Benchmark Đánh Giá Multi-Query & Parent-Child RAG (Buổi 09)")
    parser.add_argument("--questions", type=str, default=str(BASE_DIR / "eval" / "questions.json"), help="Đường dẫn file JSON tập câu hỏi đánh giá")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "reports"), help="Thư mục xuất báo cáo JSON")
    parser.add_argument("--top-k", type=int, default=3, help="Giá trị K cho Recall@K, MRR@K, nDCG@K")
    args = parser.parse_args()

    try:
        res = run_evaluation_benchmark(
            questions_path=args.questions,
            output_dir=args.output_dir,
            k=args.top_k
        )
        print("=== BẢNG TỔNG HỢP KẾT QUẢ EVALUATION BENCHMARK (BUỔI 09) ===")
        header = f"{'Mode':<15} | {'Child Recall':<12} | {'Parent Recall':<13} | {'MRR@K':<8} | {'nDCG@K':<8} | {'Latency (ms)':<12} | {'Exp Factor':<10}"
        print(header)
        print("-" * len(header))
        for mode, m in res["aggregate_metrics"].items():
            print(f"{mode:<15} | {m['child_recall_at_k']:<12.4f} | {m['parent_recall_at_k']:<13.4f} | {m['mrr_at_k']:<8.4f} | {m['ndcg_at_k']:<8.4f} | {m['mean_latency_ms']:<12.2f} | x{m['mean_expansion_factor']:<10.2f}")

    except Exception as e:
        print(f"\n[LỖI BENCHMARK EVALUATE] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
