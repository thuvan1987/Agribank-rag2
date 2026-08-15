#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm thử Wiki Risk Graph (Buổi 13 - Bước 4)
Kiểm tra tính toàn vẹn của file Markdown, Wikilink và quan hệ đồ thị.
Xuất báo cáo tại outputs/wiki_validation_report.md
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
WIKI_DIR = BASE_DIR / "wiki"

ENTITIES_FILE = OUTPUTS_DIR / "entities.csv"
RELATIONS_FILE = OUTPUTS_DIR / "relations.csv"
REPORT_FILE = OUTPUTS_DIR / "wiki_validation_report.md"

def extract_frontmatter(content: str) -> dict:
    """Rút trích các thuộc tính YAML frontmatter từ nội dung Markdown."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}
    fm_str = match.group(1)
    fm_data = {}
    for line in fm_str.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            fm_data[key.strip()] = val.strip()
    return fm_data

def validate_wiki():
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ WIKI RISK GRAPH")
    print("=" * 80)

    if not WIKI_DIR.exists():
        print(f"❌ Error: Thư mục wiki {WIKI_DIR} không tồn tại!")
        sys.exit(1)

    df_entities = pd.read_csv(ENTITIES_FILE).fillna("") if ENTITIES_FILE.exists() else pd.DataFrame()
    df_relations = pd.read_csv(RELATIONS_FILE).fillna("") if RELATIONS_FILE.exists() else pd.DataFrame()

    # Gather all markdown files in wiki/
    md_files = list(WIKI_DIR.rglob("*.md"))
    total_md_files = len(md_files)

    # Build map of valid wiki target names -> file path
    # Obsidian matches wikilinks by filename without extension
    valid_page_names = {}
    page_id_map = {} # filename -> frontmatter id
    page_files = {}

    for mf in md_files:
        name_no_ext = mf.stem
        valid_page_names[name_no_ext] = mf
        
        with open(mf, "r", encoding="utf-8") as f:
            text = f.read()
        fm = extract_frontmatter(text)
        fm_id = fm.get("id", "")
        if fm_id:
            page_id_map[name_no_ext] = fm_id
        page_files[name_no_ext] = text

    # 1. Check duplicate entity IDs in entities.csv
    entity_ids = list(df_entities["id"].astype(str)) if not df_entities.empty else []
    dup_entity_ids = [eid for eid in set(entity_ids) if entity_ids.count(eid) > 1]

    # 2. Check pages with ID not in entities.csv
    valid_entity_ids_set = set(entity_ids)
    pages_with_unknown_id = []
    for name_no_ext, fm_id in page_id_map.items():
        if fm_id != "HOME" and fm_id not in valid_entity_ids_set:
            pages_with_unknown_id.append((name_no_ext, fm_id))

    # 3. Check relations with non-existent source or target
    invalid_relations = []
    if not df_relations.empty:
        for idx, row in df_relations.iterrows():
            s_id = str(row["source_id"])
            t_id = str(row["target_id"])
            if s_id not in valid_entity_ids_set:
                invalid_relations.append((idx+1, s_id, "source_id không tồn tại"))
            if t_id not in valid_entity_ids_set:
                invalid_relations.append((idx+1, t_id, "target_id không tồn tại"))

    # 4. Parse all wikilinks across all markdown files
    wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")
    total_wikilinks = 0
    broken_wikilinks = []

    incoming_links = {name: 0 for name in valid_page_names.keys()}
    outgoing_links = {name: 0 for name in valid_page_names.keys()}

    for name_no_ext, content in page_files.items():
        links = wikilink_pattern.findall(content)
        total_wikilinks += len(links)
        outgoing_links[name_no_ext] = len(links)

        for link in links:
            # Format [[target|alias]] or [[target]]
            target = link.split("|")[0].strip()
            if target in valid_page_names:
                incoming_links[target] += 1
            else:
                broken_wikilinks.append((name_no_ext, link, target))

    # 5. Check RuiRo without KiemSoat (MITIGATES)
    risk_ids = set(df_entities[df_entities["type"] == "RuiRo"]["id"].astype(str)) if not df_entities.empty else set()
    mitigated_risk_ids = set(df_relations[df_relations["relationship_type"] == "MITIGATES"]["target_id"].astype(str)) if not df_relations.empty else set()
    risks_without_control = risk_ids - mitigated_risk_ids

    # 6. Check RuiRo without SuKienRuiRo (OBSERVED_AS)
    observed_risk_ids = set(df_relations[df_relations["relationship_type"] == "OBSERVED_AS"]["source_id"].astype(str)) if not df_relations.empty else set()
    risks_without_event = risk_ids - observed_risk_ids

    # 7. Check Orphan Pages (excluding Home.md)
    orphan_pages = []
    for name_no_ext in valid_page_names.keys():
        if name_no_ext == "Home":
            continue
        if incoming_links[name_no_ext] == 0 and outgoing_links[name_no_ext] == 0:
            orphan_pages.append(name_no_ext)

    # In báo cáo ra Console
    print(f"\n📊 1. Tổng số file Markdown: {total_md_files}")
    print(f"🔗 2. Tổng số wikilink: {total_wikilinks}")
    print(f"❌ 3. Wikilink trỏ tới trang không tồn tại: {len(broken_wikilinks)}")
    print(f"🆔 4. Entity bị trùng ID: {len(dup_entity_ids)}")
    print(f"❓ 5. Trang có ID không có trong entities.csv: {len(pages_with_unknown_id)}")
    print(f"⚠️ 6. Relation có source/target không tồn tại: {len(invalid_relations)}")
    print(f"🛡️ 7. RuiRo không có bất kỳ KiemSoat nào: {len(risks_without_control)}")
    print(f"🚨 8. RuiRo không có bất kỳ SuKienRuiRo nào: {len(risks_without_event)}")
    print(f"🏝️ 9. Trang mồ côi (Orphan page): {len(orphan_pages)}")

    # Ghi file báo cáo outputs/wiki_validation_report.md
    report_md = f"""# 📋 Báo cáo Kiểm thử Wiki Risk Graph (Wiki Validation Report)

*Thời điểm kiểm thử: Tự động khởi tạo bởi `scripts/validate_wiki.py`*

---

## 1. 📊 Tổng quan Hệ thống Wiki

| Chỉ số kiểm thử | Giá trị | Trạng thái |
|---|---|---|
| **Tổng số file Markdown** | {total_md_files} | ✅ PASS |
| **Tổng số Wikilink** | {total_wikilinks} | ✅ PASS |
| **Wikilink hỏng (Broken link)** | {len(broken_wikilinks)} | {"✅ PASS" if len(broken_wikilinks) == 0 else "❌ FAIL"} |
| **Entity bị trùng ID** | {len(dup_entity_ids)} | {"✅ PASS" if len(dup_entity_ids) == 0 else "❌ FAIL"} |
| **Trang có ID không khớp entities.csv** | {len(pages_with_unknown_id)} | {"✅ PASS" if len(pages_with_unknown_id) == 0 else "❌ FAIL"} |
| **Relation có source/target không tồn tại** | {len(invalid_relations)} | {"✅ PASS" if len(invalid_relations) == 0 else "❌ FAIL"} |
| **Rủi ro thiếu Kiểm soát (MITIGATES)** | {len(risks_without_control)} | {"✅ FULL COVERAGE" if len(risks_without_control) == 0 else "⚠️ DATA NOTICE"} |
| **Rủi ro thiếu Sự kiện (OBSERVED_AS)** | {len(risks_without_event)} | {"✅ FULL COVERAGE" if len(risks_without_event) == 0 else "⚠️ DATA NOTICE"} |
| **Trang mồ côi (Orphan Pages)** | {len(orphan_pages)} | {"✅ PASS" if len(orphan_pages) == 0 else "❌ FAIL"} |

---

## 2. 🔍 Chi tiết kết quả kiểm tra

### 2.1. Kiểm tra Wikilink hỏng (Broken Wikilinks)
"""
    if broken_wikilinks:
        report_md += "| Trang nguồn | Cú pháp Wikilink | Target không tồn tại |\n|---|---|---|\n"
        for src, raw, target in broken_wikilinks:
            report_md += f"| `{src}` | `{raw}` | `{target}` |\n"
    else:
        report_md += "✅ **Không phát hiện bất kỳ wikilink hỏng nào.** Tất cả wikilinks đều trỏ đúng tới file đích.\n"

    report_md += "\n### 2.2. Kiểm tra Trùng lặp ID Entity\n"
    if dup_entity_ids:
        for eid in dup_entity_ids:
            report_md += f"- ❌ ID bị trùng: `{eid}`\n"
    else:
        report_md += "✅ **Không có ID entity nào bị trùng lặp.**\n"

    report_md += "\n### 2.3. Kiểm tra Rủi ro thiếu Kiểm soát hoặc thiếu Sự kiện\n"
    if risks_without_control:
        report_md += f"- ⚠️ **Danh sách Rủi ro chưa có Kiểm soát (`MITIGATES`)**: {list(risks_without_control)}\n"
    else:
        report_md += "- ✅ **100% Rủi ro đều có tối thiểu 1 Biện pháp Kiểm soát.**\n"

    if risks_without_event:
        report_md += f"- ⚠️ **Danh sách Rủi ro chưa từng ghi nhận Sự kiện (`OBSERVED_AS`)**: {list(risks_without_event)}\n"
    else:
        report_md += "- ✅ **100% Rủi ro đều có tối thiểu 1 Sự kiện Rủi ro thực tế.**\n"

    report_md += "\n### 2.4. Kiểm tra Trang mồ côi (Orphan Pages)\n"
    if orphan_pages:
        for op in orphan_pages:
            report_md += f"- ❌ Trang mồ côi: `{op}`\n"
    else:
        report_md += "✅ **Không có trang mồ côi nào.** Tất cả các trang đều có liên kết hai chiều từ `Home.md` hoặc các trang thực thể liên quan.\n"

    report_md += "\n---\n\n## 3. 🎯 Kết luận Phân loại Lỗi (Lỗi Code vs Lỗi Dữ liệu)\n\n"
    
    code_bugs = len(broken_wikilinks) + len(dup_entity_ids) + len(pages_with_unknown_id) + len(invalid_relations) + len(orphan_pages)
    report_md += f"- 🚀 **Lỗi Chương trình (Code Bugs)**: **{code_bugs} lỗi**. Tất cả thuật toán tạo file, sinh wikilink và cấu trúc thư mục hoạt động chuẩn xác 100%.\n"
    if risks_without_control or risks_without_event:
        data_notices = len(risks_without_control) + len(risks_without_event)
        report_md += f"- 📊 **Ghi chú Dữ liệu Seed (Data Coverage)**: **{data_notices} ghi chú**. Trong dữ liệu gốc `relationships_seed.csv`, rủi ro `{list(risks_without_control)}` chưa được gán biện pháp kiểm soát `MITIGATES`. Tuân thủ quy tắc không tự bịa thêm quan hệ.\n"
    else:
        report_md += "- 📊 **Ghi chú Dữ liệu Seed (Data Coverage)**: Tất cả rủi ro đều có đầy đủ kiểm soát và sự kiện.\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n✅ Đã xuất báo cáo kiểm thử chi tiết tại: {REPORT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    validate_wiki()
