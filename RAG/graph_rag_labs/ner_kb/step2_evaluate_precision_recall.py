#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Đối sánh và Xác minh Kết quả Dự đoán Mối quan hệ bằng LLM
So sánh kết quả do LLM dự đoán trong ner_kb/relationships.csv với bộ nhãn chuẩn kb+hops/relationships.csv.
Tính toán các chỉ số Precision, Recall và F1-Score.
"""

import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
GT_PATH = BASE_DIR.parent / "kb+hops" / "relationships.csv"
PRED_PATH = BASE_DIR / "relationships.csv"

def calculate_metrics(tp: int, fp: int, fn: int):
    """Tính toán Precision, Recall, F1-Score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def main():
    print("==================================================")
    print("📊 BƯỚC 2: ĐỐI SÁNH VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC DỰ ĐOÁN (LLM EVALUATION)")
    print("==================================================")

    if not GT_PATH.exists():
        print(f"❌ Không tìm thấy file nhãn chuẩn Ground Truth tại: {GT_PATH}")
        sys.exit(1)

    if not PRED_PATH.exists():
        print(f"❌ Không tìm thấy file kết quả dự đoán tại: {PRED_PATH}")
        sys.exit(1)

    gt_df = pd.read_csv(GT_PATH)
    pred_df = pd.read_csv(PRED_PATH)

    print(f"\n📂 File Ground Truth ({GT_PATH.name}): {len(gt_df)} mối quan hệ chuẩn")
    print(f"📂 File Dự đoán ({PRED_PATH.name}): {len(pred_df)} mối quan hệ do LLM dự đoán")

    # Clean data
    gt_df['doc_id'] = gt_df['doc_id'].astype(str).str.strip()
    gt_df['other_doc_id'] = gt_df['other_doc_id'].astype(str).str.strip()
    gt_df['relationship_type'] = gt_df['relationship_type'].astype(str).str.strip()

    pred_df['doc_id'] = pred_df['doc_id'].astype(str).str.strip()
    pred_df['other_doc_id'] = pred_df['other_doc_id'].astype(str).str.strip()
    pred_df['relationship_type'] = pred_df['relationship_type'].astype(str).str.strip()

    # Sets for evaluation
    # 1. Strict set: (doc_id, other_doc_id, relationship_type)
    gt_strict = set(zip(gt_df['doc_id'], gt_df['other_doc_id'], gt_df['relationship_type']))
    pred_strict = set(zip(pred_df['doc_id'], pred_df['other_doc_id'], pred_df['relationship_type']))

    # 2. Pair set: (doc_id, other_doc_id)
    gt_pair = set(zip(gt_df['doc_id'], gt_df['other_doc_id']))
    pred_pair = set(zip(pred_df['doc_id'], pred_df['other_doc_id']))

    # 3. Undirected pair set: frozenset({doc_id, other_doc_id})
    gt_undirected = set(frozenset([u, v]) for u, v in gt_pair)
    pred_undirected = set(frozenset([u, v]) for u, v in pred_pair)

    # ----------------------------------------------------
    # A. Strict Evaluation (Chuẩn xác 100%: Cặp + Loại quan hệ)
    # ----------------------------------------------------
    tp_strict = len(pred_strict.intersection(gt_strict))
    fp_strict = len(pred_strict - gt_strict)
    fn_strict = len(gt_strict - pred_strict)
    p_strict, r_strict, f1_strict = calculate_metrics(tp_strict, fp_strict, fn_strict)

    # ----------------------------------------------------
    # B. Pair Evaluation (Đúng Cặp có hướng)
    # ----------------------------------------------------
    tp_pair = len(pred_pair.intersection(gt_pair))
    fp_pair = len(pred_pair - gt_pair)
    fn_pair = len(gt_pair - pred_pair)
    p_pair, r_pair, f1_pair = calculate_metrics(tp_pair, fp_pair, fn_pair)

    # ----------------------------------------------------
    # C. Undirected Pair Evaluation (Đúng Liên kết Đồ thị)
    # ----------------------------------------------------
    tp_und = len(pred_undirected.intersection(gt_undirected))
    fp_und = len(pred_undirected - gt_undirected)
    fn_und = len(gt_undirected - pred_undirected)
    p_und, r_und, f1_und = calculate_metrics(tp_und, fp_und, fn_und)

    print("\n" + "="*50)
    print("📈 KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)")
    print("="*50)

    print("\n1️⃣ ĐÁNH GIÁ NGHIÊM NGẶT (STRICT MATCH: ĐÚNG CẶP + ĐÚNG LOẠI QUAN HỆ)")
    print(f"   - True Positives (TP) : {tp_strict}")
    print(f"   - False Positives (FP): {fp_strict}")
    print(f"   - False Negatives (FN): {fn_strict}")
    print(f"   🎯 Precision : {p_strict:.4f} ({p_strict*100:.2f}%)")
    print(f"   🎯 Recall    : {r_strict:.4f} ({r_strict*100:.2f}%)")
    print(f"   🎯 F1-Score  : {f1_strict:.4f} ({f1_strict*100:.2f}%)")

    print("\n2️⃣ ĐÁNH GIÁ THEO CẶP CÓ HƯỚNG (DIRECTIONAL PAIR MATCH)")
    print(f"   - True Positives (TP) : {tp_pair}")
    print(f"   - False Positives (FP): {fp_pair}")
    print(f"   - False Negatives (FN): {fn_pair}")
    print(f"   🎯 Precision : {p_pair:.4f} ({p_pair*100:.2f}%)")
    print(f"   🎯 Recall    : {r_pair:.4f} ({r_pair*100:.2f}%)")
    print(f"   🎯 F1-Score  : {f1_pair:.4f} ({f1_pair*100:.2f}%)")

    print("\n3️⃣ ĐÁNH GIÁ VỀ LIÊN KẾT ĐỒ THỊ (UNDIRECTED GRAPH LINK MATCH)")
    print(f"   - True Positives (TP) : {tp_und}")
    print(f"   - False Positives (FP): {fp_und}")
    print(f"   - False Negatives (FN): {fn_und}")
    print(f"   🎯 Precision : {p_und:.4f} ({p_und*100:.2f}%)")
    print(f"   🎯 Recall    : {r_und:.4f} ({r_und*100:.2f}%)")
    print(f"   🎯 F1-Score  : {f1_und:.4f} ({f1_und*100:.2f}%)")

    print("\n" + "="*50)
    print("🔍 CHI TIẾT ĐỐI SÁNH:")
    print("="*50)

    print("\n✅ CÁC MỐI QUAN HỆ CHUẨN XÁC NẠP ĐƯỢC (TRUE POSITIVES):")
    for u, v, r in sorted(pred_strict.intersection(gt_strict)):
        print(f"   [+] {u} -> {v} | {r}")

    print("\n⚠️ MỐI QUAN HỆ TRONG BỘ CHUẨN BỊ BỎ SÓT (FALSE NEGATIVES):")
    missing_gt = gt_strict - pred_strict
    if missing_gt:
        for u, v, r in sorted(missing_gt):
            print(f"   [-] {u} -> {v} | {r}")
    else:
        print("   -> Không có! Đã bao phủ 100% bộ nhãn chuẩn.")

    print("\n💡 CÁC MỐI QUAN HỆ MỚI PHÁT HIỆN THÊM BỞI LLM (EXPANDED DISCOVERIES):")
    new_disc = pred_strict - gt_strict
    print(f"   -> Đã phát hiện thêm {len(new_disc)} mối quan hệ pháp lý mới từ tập 30 tài liệu.")
    for u, v, r in sorted(new_disc)[:10]:
        print(f"   [*] {u} -> {v} | {r}")
    if len(new_disc) > 10:
        print(f"   ... và {len(new_disc) - 10} mối quan hệ khác.")

    print("\n==================================================")

if __name__ == "__main__":
    main()
