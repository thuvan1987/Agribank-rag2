#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Phân tích Dữ liệu và Dự đoán Mối quan hệ giữa các Văn bản bằng LLM (Gemini API)
"""

import os
import sys
import json
import time
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR.parent / "kb+hops" / ".env"

if not ENV_FILE.exists():
    ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Không tìm thấy GEMINI_API_KEY trong file môi trường .env!")
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """Bạn là một Chuyên gia Pháp lý AI cao cấp chuyên phân tích văn bản pháp luật Ngân hàng & Tài chính Việt Nam.
Nhiệm vụ của bạn là xác định chính xác mối quan hệ pháp lý giữa 2 văn bản pháp luật (Văn bản A -> Văn bản B).

CÁC LOẠI QUAN HỆ PHÁP LÝ:
1. CAN_CU (Tên hiển thị: "Căn cứ"): Văn bản A sử dụng Văn bản B làm căn cứ pháp lý để ban hành (thường nằm ở phần căn cứ "Căn cứ Luật...", "Căn cứ Nghị định...").
2. THAY_THE (Tên hiển thị: "Thay thế"): Văn bản A ban hành để thay thế cho Văn bản B.
3. SUA_DOI_BO_SUNG (Tên hiển thị: "Sửa đổi, bổ sung"): Văn bản A ban hành để sửa đổi, bổ sung một số điều của Văn bản B.
4. HOP_NHAT (Tên hiển thị: "Hợp nhất"): Văn bản A là văn bản hợp nhất nội dung của Văn bản B.
5. VAN_BAN_BO_SUNG (Tên hiển thị: "Văn bản bổ sung"): Văn bản A hướng dẫn hoặc bổ sung chi tiết cho Văn bản B.
6. NONE (Tên hiển thị: "Không"): Không có mối quan hệ pháp lý trực tiếp nêu trên từ A tới B.

Bạn phải trả về đúng cấu trúc JSON sau:
{
    "has_relationship": true/false,
    "relationship_type": "CAN_CU" | "THAY_THE" | "SUA_DOI_BO_SUNG" | "HOP_NHAT" | "VAN_BAN_BO_SUNG" | "NONE",
    "relationship_name": "Căn cứ" | "Thay thế" | "Sửa đổi, bổ sung" | "Hợp nhất" | "Văn bản bổ sung" | "Không",
    "explanation": "Lý do ngắn gọn dẫn chứng từ văn bản"
}
"""

def extract_text_from_html(html_content: str) -> str:
    """Làm sạch HTML và chuyển đổi thành văn bản thuần."""
    if pd.isna(html_content) or not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    # Loại bỏ các dòng trống dư thừa
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def load_dataset():
    """Tải và tiền xử lý tập dữ liệu metadata.csv và content.csv."""
    meta_path = BASE_DIR / "metadata.csv"
    content_path = BASE_DIR / "content.csv"

    if not meta_path.exists() or not content_path.exists():
        print(f"❌ Không tìm thấy file dữ liệu tại {meta_path} hoặc {content_path}")
        sys.exit(1)

    meta_df = pd.read_csv(meta_path)
    content_df = pd.read_csv(content_path)
    merged_df = pd.merge(meta_df, content_df, on="id", how="inner")

    docs = {}
    for _, row in merged_df.iterrows():
        doc_id = str(row["id"])
        clean_txt = extract_text_from_html(row["content_html"])
        preamble = clean_txt[:3000] # Lấy 3000 ký tự đầu làm phần mở đầu / căn cứ

        docs[doc_id] = {
            "id": doc_id,
            "title": str(row["title"]) if pd.notna(row["title"]) else "",
            "so_ky_hieu": str(row["so_ky_hieu"]) if pd.notna(row["so_ky_hieu"]) else "",
            "loai_van_ban": str(row["loai_van_ban"]) if pd.notna(row["loai_van_ban"]) else "",
            "preamble": preamble,
            "full_text": clean_txt
        }
    return docs


def find_candidate_pairs(docs: dict) -> list:
    """Tìm các cặp tài liệu tiềm năng (A, B) dựa trên trích dẫn số ký hiệu, tiêu đề và phần căn cứ."""
    candidates = []
    seen = set()

    for doc_a_id, a in docs.items():
        for doc_b_id, b in docs.items():
            if doc_a_id == doc_b_id:
                continue

            pair_key = (doc_a_id, doc_b_id)
            if pair_key in seen:
                continue

            skh_b = b["so_ky_hieu"]
            found = False
            reason = ""

            # Check 1: Số ký hiệu B có trong tiêu đề hoặc nội dung A
            if skh_b and len(skh_b) >= 4:
                pattern = re.escape(skh_b)
                if re.search(pattern, a["title"], re.IGNORECASE):
                    found = True
                    reason = f"Số ký hiệu {skh_b} xuất hiện trong tiêu đề A"
                elif re.search(pattern, a["preamble"], re.IGNORECASE):
                    found = True
                    reason = f"Số ký hiệu {skh_b} xuất hiện trong phần căn cứ A"
                elif re.search(pattern, a["full_text"], re.IGNORECASE):
                    found = True
                    reason = f"Số ký hiệu {skh_b} xuất hiện trong thân văn bản A"

            # Check 2: Tìm theo cụm tên tiêu đề văn bản B trong căn cứ văn bản A
            if not found:
                title_b = b["title"]
                clean_title_b = re.sub(r'số\s+[\w\/]+', '', title_b, flags=re.IGNORECASE).strip()
                clean_title_b = re.sub(r'^\s*(Luật|Nghị định|Thông tư)\s+', '', clean_title_b, flags=re.IGNORECASE).strip()
                if len(clean_title_b) > 8 and clean_title_b.lower() in a["preamble"].lower():
                    found = True
                    reason = f"Trích dẫn tên văn bản '{clean_title_b[:30]}' trong căn cứ A"

            # Check 3: Trích dẫn đảo ngược (ví dụ A sửa đổi/bổ sung B hoặc B căn cứ A)
            if not found:
                skh_a = a["so_ky_hieu"]
                if skh_a and len(skh_a) >= 4:
                    if re.search(re.escape(skh_a), b["full_text"], re.IGNORECASE):
                        found = True
                        reason = f"Trích dẫn đảo ngược số ký hiệu A ({skh_a}) trong B"

            # Check 4: Văn bản hợp nhất (VBHN) và tiêu đề tương đồng
            if not found:
                if "Văn bản hợp nhất" in a["loai_van_ban"] or "Văn bản hợp nhất" in a["title"]:
                    words_b = [w for w in b["title"].split() if len(w) > 4 and w.lower() not in ["thông", "tư", "nghị", "định", "luật", "quy", "định"]]
                    match_count = sum(1 for w in words_b if w.lower() in a["title"].lower())
                    if match_count >= 3:
                        found = True
                        reason = f"Văn bản hợp nhất khớp tiêu đề ({match_count} từ)"

            if found:
                seen.add(pair_key)
                candidates.append((a, b, reason))

    return candidates


def predict_relationship_llm(doc_a: dict, doc_b: dict, reason: str) -> dict:
    """Gọi Gemini LLM để phân tích mối quan hệ giữa Văn bản A và Văn bản B."""
    prompt = f"""Phân tích mối quan hệ pháp lý từ VĂN BẢN A đến VĂN BẢN B:

=== VĂN BẢN A ===
- ID: {doc_a['id']}
- Tiêu đề: {doc_a['title']}
- Số ký hiệu: {doc_a['so_ky_hieu']}
- Loại văn bản: {doc_a['loai_van_ban']}
- Phần mở đầu / Căn cứ:
{doc_a['preamble'][:1500]}

=== VĂN BẢN B ===
- ID: {doc_b['id']}
- Tiêu đề: {doc_b['title']}
- Số ký hiệu: {doc_b['so_ky_hieu']}
- Loại văn bản: {doc_b['loai_van_ban']}
- Phần mở đầu / Căn cứ:
{doc_b['preamble'][:1500]}

Gợi ý phát hiện: {reason}

Trả về kết quả phân tích theo đúng định dạng JSON yêu cầu.
"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            if res and res.text:
                data = json.loads(res.text.strip())
                return data
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()) and attempt < max_retries - 1:
                sleep_sec = 10 * (attempt + 1)
                print(f"  ⚠️ Quota limit 429. Thử lại sau {sleep_sec}s...")
                time.sleep(sleep_sec)
                continue
            else:
                print(f"  ❌ Lỗi gọi LLM cho cặp ({doc_a['id']} -> {doc_b['id']}): {e}")
                break

    return {
        "has_relationship": False,
        "relationship_type": "NONE",
        "relationship_name": "Không",
        "explanation": "Lỗi kết nối hoặc không phân tích được"
    }


def main():
    print("==================================================")
    print("🚀 BƯỚC 1: DỰ ĐOÁN MỐI QUAN HỆ VĂN BẢN BẰNG GEMINI LLM")
    print("==================================================")

    print("\n📂 1. Tải dữ liệu từ metadata.csv và content.csv...")
    docs = load_dataset()
    print(f"   -> Đã tải {len(docs)} tài liệu.")

    print("\n🔍 2. Lọc các cặp văn bản ứng viên tiềm năng...")
    candidates = find_candidate_pairs(docs)
    print(f"   -> Tìm thấy {len(candidates)} cặp ứng viên để gửi tới LLM.")

    print("\n🤖 3. Đang gửi các cặp tới Gemini LLM để phân tích...")
    results = []

    for idx, (doc_a, doc_b, reason) in enumerate(candidates, 1):
        print(f"\n[{idx}/{len(candidates)}] Đang phân tích cặp:")
        print(f"   A: [{doc_a['id']}] {doc_a['so_ky_hieu']} - {doc_a['title'][:60]}")
        print(f"   B: [{doc_b['id']}] {doc_b['so_ky_hieu']} - {doc_b['title'][:60]}")

        pred = predict_relationship_llm(doc_a, doc_b, reason)

        has_rel = pred.get("has_relationship", False)
        rel_type = pred.get("relationship_type", "NONE")
        rel_name = pred.get("relationship_name", "Không")
        explanation = pred.get("explanation", "")

        print(f"   => Dự đoán: {has_rel} | {rel_type} ({rel_name})", flush=True)
        print(f"   => Diễn giải: {explanation}", flush=True)

        if has_rel and rel_type != "NONE":
            results.append({
                "doc_id": doc_a["id"],
                "other_doc_id": doc_b["id"],
                "relationship": rel_name,
                "relationship_type": rel_type,
                "explanation": explanation
            })
            # Ghi lưu ngay vào relationships.csv để cập nhật tiến trình
            output_path = BASE_DIR / "relationships.csv"
            res_df = pd.DataFrame(results)
            save_df = res_df[["doc_id", "other_doc_id", "relationship", "relationship_type"]]
            save_df.to_csv(output_path, index=False, encoding="utf-8")

        # Nghỉ ngắn giữa các request để tránh rate limit
        time.sleep(0.5)

    print(f"\n✅ Hoàn thành phân tích {len(candidates)} cặp!")
    print(f"📊 Tổng số mối quan hệ phát hiện được: {len(results)}")

    # Ghi vào relationships.csv
    output_path = BASE_DIR / "relationships.csv"
    res_df = pd.DataFrame(results)

    if not res_df.empty:
        # Giữ lại đúng schema chuẩn: doc_id,other_doc_id,relationship,relationship_type
        save_df = res_df[["doc_id", "other_doc_id", "relationship", "relationship_type"]]
        save_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n💾 Đã lưu kết quả vào {output_path}")
    else:
        print("\n⚠️ Không phát hiện được mối quan hệ nào!")

    print("\n==================================================")
    print("TỔNG HỢP CÁC MỐI QUAN HỆ ĐÃ DỰ ĐOÁN:")
    print("==================================================")
    if not res_df.empty:
        for idx, r in res_df.iterrows():
            print(f"{r['doc_id']} -> {r['other_doc_id']} | {r['relationship']} ({r['relationship_type']})")
    print("==================================================")

if __name__ == "__main__":
    main()
