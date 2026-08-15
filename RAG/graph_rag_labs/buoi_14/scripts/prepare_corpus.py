#!/usr/bin/env python3
"""
scripts/prepare_corpus.py — Buổi 14
Mục đích: Chuẩn hóa Corpus dữ liệu văn bản pháp lý ngân hàng từ 3 file nguồn:
  - ../kb+hops/metadata.csv
  - ../kb+hops/content.csv
  - ../kb+hops/relationships.csv

Kết xuất:
  - buoi_14/data/processed/chunks_normalized.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = (BASE_DIR.parent / "kb+hops").resolve()
OUTPUT_DIR = BASE_DIR / "data" / "processed"

METADATA_CSV = KB_DIR / "metadata.csv"
CONTENT_CSV = KB_DIR / "content.csv"
RELATIONS_CSV = KB_DIR / "relationships.csv"
TARGET_CSV = OUTPUT_DIR / "chunks_normalized.csv"

def clean_element_text(element) -> str:
    """Làm sạch thẻ HTML và chuẩn hóa khoảng trắng/dòng trống."""
    if isinstance(element, str):
        text = element
    elif element.name == "table":
        rows = []
        for tr in element.find_all("tr"):
            cells = [cell.get_text(separator=" ", strip=True) for cell in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        text = "\n".join(rows)
    else:
        text = element.get_text(separator=" ", strip=True)

    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

def prepare_corpus():
    print("=== BẮT ĐẦU CHUẨN HÓA CORPUS BUỔI 14 ===")
    print(f"Thư mục nguồn (Read-Only): {KB_DIR}")
    print(f"Tệp đầu ra:                  {TARGET_CSV}")

    if not METADATA_CSV.exists() or not CONTENT_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn trong '{KB_DIR}'.")

    # Đọc dữ liệu nguồn
    meta_df = pd.read_csv(METADATA_CSV, encoding="utf-8")
    content_df = pd.read_csv(CONTENT_CSV, encoding="utf-8")

    # Map metadata theo id (string)
    meta_dict = {}
    for _, row in meta_df.iterrows():
        meta_dict[str(row["id"]).strip()] = row.to_dict()

    chunks = []
    total_docs = len(content_df)

    for idx, row in content_df.iterrows():
        doc_id = str(row["id"]).strip()
        meta = meta_dict.get(doc_id, {})

        html_str = str(row["content_html"])
        soup = BeautifulSoup(html_str, "html.parser")

        elements = soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "table"])
        
        current_chapter = ""
        current_section = ""
        current_article = ""
        current_chunk_text = []

        def save_chunk(c_text, c_chap, c_sec, c_art):
            full_text = "\n".join(c_text).strip()
            if not full_text:
                return
            cid = f"{doc_id}_chunk_{len(chunks)+1}"
            chunks.append({
                "chunk_id": cid,
                "document_id": doc_id,
                "text": full_text,
                "source_file": "content.csv",
                "title": meta.get("title", ""),
                "document_type": meta.get("loai_van_ban", ""),
                "chapter": c_chap,
                "section": c_sec,
                "article": c_art,
                "clause": "",
                "effective_date": meta.get("ngay_co_hieu_luc", ""),
                "status": meta.get("tinh_trang_hieu_luc", ""),
                "so_ky_hieu": meta.get("so_ky_hieu", ""),
                "co_quan_ban_hanh": meta.get("co_quan_ban_hanh", ""),
                "ngay_ban_hanh": meta.get("ngay_ban_hanh", "")
            })

        for el in elements:
            t = clean_element_text(el)
            if not t:
                continue

            m_ch = re.match(r"^(Chương\s+[IVXLCDM0-9]+(?:\:|\.|\s|$).*)", t, re.IGNORECASE)
            if m_ch:
                current_chapter = m_ch.group(1).strip()
                continue

            m_sec = re.match(r"^(Mục\s+[0-9IVXLCDM]+(?:\:|\.|\s|$).*)", t, re.IGNORECASE)
            if m_sec:
                current_section = m_sec.group(1).strip()
                continue

            m_art = re.match(r"^(Điều\s+\d+[\.\:]?.*)", t, re.IGNORECASE)
            if m_art:
                if current_chunk_text:
                    save_chunk(current_chunk_text, current_chapter, current_section, current_article)
                    current_chunk_text = []
                current_article = m_art.group(1).strip()
                current_chunk_text.append(t)
            else:
                if current_chunk_text:
                    current_chunk_text.append(t)

        if current_chunk_text:
            save_chunk(current_chunk_text, current_chapter, current_section, current_article)

    # Đưa vào DataFrame và kiểm tra chất lượng
    df = pd.DataFrame(chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TARGET_CSV, index=False, encoding="utf-8")

    # Thống kê kết quả
    total_chunks = len(df)
    unique_chunk_ids = df["chunk_id"].nunique()
    unique_docs = df["document_id"].nunique()
    empty_chunks = (df["text"].str.strip() == "").sum()
    duplicate_rows = df.duplicated().sum()

    print("\n=== KẾT QUẢ CHUẨN HÓA CORPUS ===")
    print(f"Tổng số Chunk (total_chunks):     {total_chunks}")
    print(f"Số Document đại diện (unique_docs): {unique_docs}/{total_docs}")
    print(f"Số Chunk bị thiếu text (empty):    {empty_chunks}")
    print(f"Số dòng bị trùng (duplicates):     {duplicate_rows}")
    print(f"Số chunk_id duy nhất (unique_ids): {unique_chunk_ids}/{total_chunks}")

    print("\n--- 3 SAMPLE RECORDS ---")
    sample_records = df.head(3).to_dict(orient="records")
    for idx, s in enumerate(sample_records, start=1):
        print(f"\n[Sample #{idx}]")
        print(f"  chunk_id:        {s['chunk_id']}")
        print(f"  document_id:     {s['document_id']}")
        print(f"  title:           {s['title']}")
        print(f"  article:         {s['article']}")
        print(f"  document_type:   {s['document_type']}")
        print(f"  effective_date:  {s['effective_date']}")
        print(f"  status:          {s['status']}")
        print(f"  text preview:    {s['text'][:120]}...")

    print(f"\n✅ Đã lưu corpus chuẩn hóa thành công tại: {TARGET_CSV}")

if __name__ == "__main__":
    prepare_corpus()
