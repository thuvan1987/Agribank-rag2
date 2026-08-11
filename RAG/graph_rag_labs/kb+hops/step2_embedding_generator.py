import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_CHUNKS_FILE = OUTPUT_DIR / "chunks.json"

OUTPUT_EMBEDDINGS_JSON = OUTPUT_DIR / "chunks_with_embeddings.json"
OUTPUT_EMBEDDINGS_JSONL = OUTPUT_DIR / "chunks_with_embeddings.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "embedding_summary.json"

MODEL_NAME = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"

def main():
    print("==================================================")
    print(" BẮT ĐẦU THỰC THI BƯỚC 2 - BUỔI 10: EMBEDDING GENERATION ")
    print("==================================================")

    # 1. Kiểm tra file input từ Bước 1
    if not INPUT_CHUNKS_FILE.exists():
        print(f"❌ LỖI: Không tìm thấy file đầu vào {INPUT_CHUNKS_FILE}")
        sys.exit(1)

    with open(INPUT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_chunks = len(chunks)
    print(f"✓ Đã nạp thành công {total_chunks} chunks từ {INPUT_CHUNKS_FILE.name}.")

    # 2. Kiểm tra thiết bị (CPU) & Nạp mô hình SentenceTransformers
    device = "cpu"
    print(f"✓ Sử dụng thiết bị tính toán: {device.upper()} (Torch device: {torch.device(device)})")
    print(f"✓ Đang nạp mô hình local từ HuggingFace: '{MODEL_NAME}'...")

    start_model_load = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    load_duration = time.time() - start_model_load
    print(f"✓ Nạp mô hình thành công (thời gian: {load_duration:.2f}s).")

    # 3. Chuẩn bị dữ liệu văn bản để encode
    texts = [c.get("clean_text", "") for c in chunks]

    print(f"✓ Đang tiến hành tạo embedding cho {total_chunks} chunks trên CPU...")
    start_embed = time.time()

    # Thực hiện encode với progress bar và batch_size thích hợp cho CPU
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embed_duration = time.time() - start_embed
    print(f"✓ Tạo embedding hoàn tất (thời gian: {embed_duration:.2f}s, tốc độ: {total_chunks / embed_duration:.1f} chunks/s).")

    # 4. Kiểm tra kích thước vector nhúng
    if len(embeddings) == 0:
        print("❌ LỖI: Không có vector embedding nào được tạo.")
        sys.exit(1)

    embedding_dim = int(embeddings[0].shape[0])
    print(f"✓ Kiểm tra thực tế kích thước vector embedding: {embedding_dim} chiều.")

    # 5. Gắn embedding vào chunks object
    embedded_chunks = []
    success_count = 0
    error_count = 0
    doc_ids = set()

    for idx, c in enumerate(chunks):
        vec = embeddings[idx].tolist()
        doc_ids.add(c.get("doc_id"))

        if len(vec) == embedding_dim and not any(math.isnan(x) for x in vec):
            success_count += 1
        else:
            error_count += 1

        c_with_emb = dict(c)
        c_with_emb["embedding"] = vec
        embedded_chunks.append(c_with_emb)

    # 6. Ghi xuất kết quả ra file JSON / JSONL
    print(f"✓ Đang ghi kết quả ra các file trong {OUTPUT_DIR.name}...")

    with open(OUTPUT_EMBEDDINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_EMBEDDINGS_JSONL, "w", encoding="utf-8") as f:
        for item in embedded_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Ghi file summary
    summary = {
        "total_documents": len(doc_ids),
        "total_chunks": total_chunks,
        "successfully_embedded": success_count,
        "error_chunks": error_count,
        "embedding_dimension": embedding_dim,
        "device": device,
        "model_name": MODEL_NAME,
        "embedding_duration_seconds": round(embed_duration, 2)
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==================================================")
    print(" XUẤT OUTPUT BƯỚC 2 THÀNH CÔNG:")
    print(f"  1. {OUTPUT_EMBEDDINGS_JSON}")
    print(f"  2. {OUTPUT_EMBEDDINGS_JSONL}")
    print(f"  3. {SUMMARY_FILE}")
    print("==================================================\n")

    # 7. In ví dụ Console thực tế (Preview Chunk + Vector 5 phần tử)
    sample_chunk = embedded_chunks[0]
    sample_vec = sample_chunk["embedding"]

    print("==================================================")
    print(" DEMO CONSOLE: VÍ DỤ CHUNK EMBEDDING THỰC TẾ ")
    print("==================================================")
    print(f"  • chunk_id            : {sample_chunk['chunk_id']}")
    print(f"  • doc_id              : {sample_chunk['doc_id']}")
    print(f"  • level               : {sample_chunk['level']}")
    print(f"  • title               : {sample_chunk['title']}")
    print(f"  • clean_text preview  : {sample_chunk['clean_text'][:120]}...")
    print(f"  • embedding dimension : {len(sample_vec)} chiều")
    print(f"  • 5 giá trị đầu tiên  : {[round(x, 6) for x in sample_vec[:5]]}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
