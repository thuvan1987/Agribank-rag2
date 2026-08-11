import csv
import json
import os
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString

# Tăng giới hạn đọc field CSV cho các chuỗi HTML kích thước lớn
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
CONTENT_CSV = BASE_DIR / "content.csv"
METADATA_CSV = BASE_DIR / "metadata.csv"
RELATIONS_CSV = BASE_DIR / "relationships.csv"
OUTPUT_DIR = BASE_DIR / "output"

# Đánh số thứ tự cấp bậc cho cấu trúc phân cấp (Hierarchical Ranks)
LEVEL_RANKS = {
    "Document": 0,
    "Chapter": 1,
    "Section": 2,
    "SubSection": 2.5,
    "Article": 3,
    "Clause": 4,
    "Item": 5,
    "Content": 6
}

def clean_element_text(element):
    """
    Chuyển đổi thẻ HTML thành văn bản sạch (clean text),
    loại bỏ các thẻ HTML rác và giữ lại cấu trúc nội dung + bảng biểu.
    """
    if isinstance(element, str):
        text = element
    elif element.name == "table":
        # Xử lý bảng biểu thành văn bản dạng bảng sạch
        rows = []
        for tr in element.find_all("tr"):
            cells = [cell.get_text(separator=" ", strip=True) for cell in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        text = "\n".join(rows)
    else:
        text = element.get_text(separator=" ", strip=True)
    
    # Loại bỏ ký tự đặc biệt & chuẩn hóa khoảng trắng
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

def detect_level_and_title(tag, current_context_level="Document"):
    """
    Nhận diện cấp độ (level) và tiêu đề (title) dựa trên:
    1. CSS Class (`prov-chapter`, `prov-article`, `prov-clause`,...)
    2. Fallback Regex (Chương..., Mục..., Điều..., Khoản...) nếu không có class prov-*
    """
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    
    class_str = " ".join(classes)
    raw_text = clean_element_text(tag)
    
    if not raw_text:
        return None, None, None

    # --- Mode 1: Class-based Detection ---
    if "prov-chapter" in class_str:
        return "Chapter", raw_text, raw_text
    elif "prov-section" in class_str:
        return "Section", raw_text, raw_text
    elif "prov-subsection" in class_str:
        return "SubSection", raw_text, raw_text
    elif "prov-article" in class_str:
        return "Article", raw_text, raw_text
    elif "prov-clause" in class_str:
        return "Clause", raw_text[:120], raw_text
    elif "prov-item" in class_str:
        return "Item", raw_text[:120], raw_text

    # --- Mode 2: Regex-based Detection (Cho MsoNormal và thẻ không class) ---
    # Chapter Regex
    m_ch = re.match(r"^(Chương\s+[IVXLCDM0-9]+(?:\:|\.|\s|$).*)", raw_text, re.IGNORECASE)
    if m_ch:
        return "Chapter", m_ch.group(1).strip(), raw_text

    # Section Regex
    m_sec = re.match(r"^(Mục\s+[0-9IVXLCDM]+(?:\:|\.|\s|$).*)", raw_text, re.IGNORECASE)
    if m_sec:
        return "Section", m_sec.group(1).strip(), raw_text

    # Article Regex
    m_art = re.match(r"^(Điều\s+\d+[\.\:]?.*)", raw_text, re.IGNORECASE)
    if m_art:
        return "Article", m_art.group(1).strip(), raw_text

    # Clause Regex (ví dụ: "1. Nghị định này...", "2. Điều khoản...")
    if current_context_level in ["Article", "Clause", "Item"]:
        m_cl = re.match(r"^(\d+\.\s+.*)", raw_text)
        if m_cl:
            return "Clause", raw_text[:120], raw_text

    # Item Regex (ví dụ: "a) Phạm vi...", "b) Đối tượng...")
    if current_context_level in ["Clause", "Item"]:
        m_it = re.match(r"^([a-zđ]\)\s+.*)", raw_text, re.IGNORECASE)
        if m_it:
            return "Item", raw_text[:120], raw_text

    # Nội dung thông thường (Content)
    return "Content", raw_text[:100], raw_text

def parse_html_document(doc_id, html_content, doc_metadata=None):
    """
    Phân tích toàn bộ 1 văn bản HTML thành danh sách Chunks có cấu trúc phân cấp
    và các mối quan hệ (PART_OF, PARENT_OF, NEXT).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Loại bỏ thẻ script, style, head, meta
    for s in soup(["script", "style", "head", "meta"]):
        s.decompose()
        
    chunks = []
    relationships = []
    
    # Tạo Chunk gốc cho Document
    doc_title = doc_metadata.get("title", f"Document {doc_id}") if doc_metadata else f"Document {doc_id}"
    doc_chunk_id = f"doc_{doc_id}"
    
    # Stack quản lý các cấp độ cha-con: mỗi item dạng (level_rank, chunk_id)
    stack = [(LEVEL_RANKS["Document"], doc_chunk_id)]
    
    # Lấy các phần tử nội dung chính (bỏ qua bảng tiêu đề Quốc hiệu/Header hành chính)
    elements = []
    for elem in soup.find_all(["p", "table"]):
        # Bỏ qua các bảng tiêu đề hành chính ở đầu trang (Quốc hiệu, Tên cơ quan)
        if elem.name == "table":
            tbl_text = clean_element_text(elem)
            if "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in tbl_text or "NGÂN HÀNG NHÀ NƯỚC" in tbl_text:
                continue
        
        # Bỏ qua dòng tiêu đề hành chính trong <p> ở đầu văn bản
        p_text = clean_element_text(elem)
        if not p_text:
            continue
            
        elements.append(elem)

    chunk_counter = 0
    prev_chunk_id = None
    
    for elem in elements:
        current_context = "Document"
        for rank, cid in reversed(stack):
            for lvlname, lvllank in LEVEL_RANKS.items():
                if lvllank == rank:
                    current_context = lvlname
                    break
            break

        level, title, clean_text = detect_level_and_title(elem, current_context)
        if not level or not clean_text:
            continue
            
        rank = LEVEL_RANKS.get(level, LEVEL_RANKS["Content"])
        
        # Nếu là Content thông thường mà chunk trước đó là Clause/Item/Article,
        # gộp văn bản vào chunk trước đó thay vì tạo chunk rời rạc không tiêu đề
        if level == "Content" and len(chunks) > 0 and chunks[-1]["level"] in ["Article", "Clause", "Item", "Content"]:
            chunks[-1]["clean_text"] += "\n" + clean_text
            continue

        chunk_counter += 1
        chunk_id = f"doc_{doc_id}_chunk_{chunk_counter:04d}"
        
        # Tìm Parent thích hợp trên Stack
        while len(stack) > 1 and stack[-1][0] >= rank:
            stack.pop()
            
        parent_id = stack[-1][1]
        
        chunk_obj = {
            "chunk_id": chunk_id,
            "doc_id": str(doc_id),
            "level": level,
            "title": title,
            "clean_text": clean_text,
            "parent_id": parent_id
        }
        chunks.append(chunk_obj)
        
        # 1. Quan hệ PART_OF (Chunk -> Document)
        relationships.append({
            "source": chunk_id,
            "target": doc_chunk_id,
            "type": "PART_OF"
        })
        
        # 2. Quan hệ PARENT_OF (Parent -> Child)
        if parent_id:
            relationships.append({
                "source": parent_id,
                "target": chunk_id,
                "type": "PARENT_OF"
            })
            
        # 3. Quan hệ NEXT (Chunk trước -> Chunk sau trong cùng Document)
        if prev_chunk_id:
            relationships.append({
                "source": prev_chunk_id,
                "target": chunk_id,
                "type": "NEXT"
            })
            
        prev_chunk_id = chunk_id
        
        # Đưa chunk mới vào Stack (trừ Content)
        if level != "Content":
            stack.append((rank, chunk_id))
            
    return chunks, relationships

def main():
    print("==================================================")
    print(" BẮT ĐẦU THỰC THI BƯỚC 1 - BUỔI 10: HTML CHUNKING ")
    print("==================================================")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Đọc metadata.csv
    metadata_map = {}
    if METADATA_CSV.exists():
        with open(METADATA_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_map[row["id"]] = row
        print(f"✓ Đã đọc metadata của {len(metadata_map)} văn bản.")

    # 2. Đọc content.csv và tiến hành chunking
    all_chunks = []
    all_relationships = []
    doc_summary_list = []
    
    with open(CONTENT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        print(f"✓ Đã nạp {len(rows)} bản ghi từ content.csv.")

        for idx, row in enumerate(rows, 1):
            doc_id = row[0]
            html_content = row[1]
            doc_meta = metadata_map.get(doc_id, {})
            
            chunks, rels = parse_html_document(doc_id, html_content, doc_meta)
            all_chunks.extend(chunks)
            all_relationships.extend(rels)
            
            # Thống kê theo level cho từng doc
            level_counts = {}
            for c in chunks:
                lvl = c["level"]
                level_counts[lvl] = level_counts.get(lvl, 0) + 1
                
            doc_summary_list.append({
                "doc_id": doc_id,
                "title": doc_meta.get("title", f"Doc {doc_id}"),
                "total_chunks": len(chunks),
                "level_breakdown": level_counts
            })
            print(f"  - Document [{doc_id}] ({idx}/{len(rows)}): {len(chunks)} chunks được tạo.")

    # 3. Xuất file output
    chunks_json_file = OUTPUT_DIR / "chunks.json"
    chunks_jsonl_file = OUTPUT_DIR / "chunks.jsonl"
    rels_json_file = OUTPUT_DIR / "relationships.json"
    summary_json_file = OUTPUT_DIR / "summary.json"
    
    # Ghi chunks.json
    with open(chunks_json_file, mode="w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    # Ghi chunks.jsonl
    with open(chunks_jsonl_file, mode="w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    # Ghi relationships.json
    with open(rels_json_file, mode="w", encoding="utf-8") as f:
        json.dump(all_relationships, f, ensure_ascii=False, indent=2)

    # Thống kê tổng hợp
    total_level_counts = {}
    for c in all_chunks:
        lvl = c["level"]
        total_level_counts[lvl] = total_level_counts.get(lvl, 0) + 1
        
    rel_type_counts = {}
    for r in all_relationships:
        rtype = r["type"]
        rel_type_counts[rtype] = rel_type_counts.get(rtype, 0) + 1

    summary_data = {
        "total_documents": len(rows),
        "total_chunks": len(all_chunks),
        "level_breakdown": total_level_counts,
        "relationship_breakdown": rel_type_counts,
        "documents": doc_summary_list
    }
    
    with open(summary_json_file, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n==================================================")
    print(" XUẤT OUTPUT THÀNH CÔNG:")
    print(f"  1. {chunks_json_file}")
    print(f"  2. {chunks_jsonl_file}")
    print(f"  3. {rels_json_file}")
    print(f"  4. {summary_json_file}")
    print("==================================================\n")

    # 4. In ví dụ hierarchical chunk thực tế theo đường đi phân cấp
    print("==================================================")
    print(" DEMO CONSOLE: CHUỖI HIERARCHICAL CHUNK THỰC TẾ ")
    print(" (Document -> Chương -> Điều -> Khoản/Điểm)      ")
    print("==================================================")
    
    # Tìm 1 chuỗi thực tế trong Document (ưu tiên doc 163441 hoặc 44209)
    sample_doc_id = "163441"
    sample_chunks = [c for c in all_chunks if c["doc_id"] == sample_doc_id]
    
    # Chọn ra 1 Chapter, 1 Article thuộc Chapter đó, và 1 Clause thuộc Article đó
    sample_chapter = next((c for c in sample_chunks if c["level"] == "Chapter"), None)
    if sample_chapter:
        sample_article = next((c for c in sample_chunks if c["level"] == "Article" and c["parent_id"] == sample_chapter["chunk_id"]), None)
        if not sample_article:
            sample_article = next((c for c in sample_chunks if c["level"] == "Article"), None)
    else:
        sample_article = next((c for c in sample_chunks if c["level"] == "Article"), None)
        
    sample_clause = next((c for c in sample_chunks if c["level"] in ["Clause", "Item"] and sample_article and c["parent_id"] == sample_article["chunk_id"]), None)

    nodes_to_show = [c for c in [sample_chapter, sample_article, sample_clause] if c]
    
    for idx, node in enumerate(nodes_to_show, 1):
        # Tìm quan hệ NEXT của node này
        next_rel = next((r for r in all_relationships if r["source"] == node["chunk_id"] and r["type"] == "NEXT"), None)
        next_target = next_rel["target"] if next_rel else "N/A (End of Document)"
        
        print(f"\n--- Node [{idx}] - Cấp: {node['level']} ---")
        print(f"  • chunk_id    : {node['chunk_id']}")
        print(f"  • doc_id      : {node['doc_id']}")
        print(f"  • level       : {node['level']}")
        print(f"  • title       : {node['title']}")
        print(f"  • parent_id   : {node['parent_id']}")
        print(f"  • NEXT link   : {next_target}")
        print(f"  • clean_text  : {node['clean_text'][:150]}...")

    # Kiểm tra văn bản 25692 (MsoNormal legacy)
    doc25692_chunks = [c for c in all_chunks if c["doc_id"] == "25692"]
    doc25692_chapters = [c for c in doc25692_chunks if c["level"] == "Chapter"]
    doc25692_articles = [c for c in doc25692_chunks if c["level"] == "Article"]
    
    print("\n--------------------------------------------------")
    print(f" KẾT QUẢ PARSE VĂN BẢN LEGACY 25692 (MsoNormal):")
    print(f"  • Tổng số Chunks : {len(doc25692_chunks)}")
    print(f"  • Số Chương       : {len(doc25692_chapters)} (Mẫu: {[c['title'] for c in doc25692_chapters[:3]]})")
    print(f"  • Số Điều         : {len(doc25692_articles)} (Mẫu: {[c['title'] for c in doc25692_articles[:3]]})")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
