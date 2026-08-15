#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script tự động sinh Wiki Markdown cho Wiki Risk Graph (Buổi 13 - Bước 3)
Đọc từ outputs/entities.csv và outputs/relations.csv
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

def safe_filename(name: str) -> str:
    """Xử lý tên file an toàn cho hệ điều hành nhưng giữ nguyên cho Obsidian wikilink."""
    clean = re.sub(r'[\\/*?:"<>|]', '-', str(name))
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Giới hạn độ dài tên file nếu quá dài
    if len(clean) > 100:
        clean = clean[:97] + "..."
    return clean

def build_wiki():
    print("=" * 80)
    print("BẮT ĐẦU SINH WIKI MARKDOWN CHO RISK GRAPH")
    print("=" * 80)

    if not ENTITIES_FILE.exists() or not RELATIONS_FILE.exists():
        print("❌ Error: Không tìm thấy entities.csv hoặc relations.csv trong thư mục outputs/")
        sys.exit(1)

    df_entities = pd.read_csv(ENTITIES_FILE).fillna("")
    df_relations = pd.read_csv(RELATIONS_FILE).fillna("")

    # Tạo các thư mục wiki
    risks_dir = WIKI_DIR / "risks"
    controls_dir = WIKI_DIR / "controls"
    events_dir = WIKI_DIR / "events"

    for d in [WIKI_DIR, risks_dir, controls_dir, events_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Tạo mapping từ entity_id -> (safe_name, entity_row)
    entity_map = {}
    id_to_filename = {}

    for _, row in df_entities.iterrows():
        e_id = str(row["id"])
        e_name = str(row["name"])
        s_name = safe_filename(e_name)
        entity_map[e_id] = row
        id_to_filename[e_id] = s_name

    # 2. Gom nhóm quan hệ theo source_id và target_id
    # rels_from_source[source_id] = list of relation dicts
    # rels_to_target[target_id] = list of relation dicts
    rels_from_source = {}
    rels_to_target = {}

    for _, row in df_relations.iterrows():
        s_id = str(row["source_id"])
        t_id = str(row["target_id"])

        rel_data = {
            "source_id": s_id,
            "target_id": t_id,
            "relationship_type": str(row["relationship_type"]),
            "source": str(row["source"]),
            "evidence_quote": str(row["evidence_quote"]),
            "confidence": str(row["confidence"]),
            "verification_status": str(row["verification_status"]),
            "data_origin": str(row["data_origin"])
        }

        rels_from_source.setdefault(s_id, []).append(rel_data)
        rels_to_target.setdefault(t_id, []).append(rel_data)

    created_pages = 0
    total_wikilinks = 0

    # 3. Sinh trang cho từng RuiRo
    risks_list = []
    for _, row in df_entities[df_entities["type"] == "RuiRo"].iterrows():
        e_id = str(row["id"])
        name = str(row["name"])
        fname = id_to_filename[e_id]
        file_path = risks_dir / f"{fname}.md"
        risks_list.append((e_id, name, fname))

        # Tìm các kiểm soát MITIGATES rủi ro này (target_id == e_id)
        mitigating_controls = [r for r in rels_to_target.get(e_id, []) if r["relationship_type"] == "MITIGATES"]
        # Tìm các sự kiện OBSERVED_AS từ rủi ro này (source_id == e_id)
        observed_events = [r for r in rels_from_source.get(e_id, []) if r["relationship_type"] == "OBSERVED_AS"]

        content = f"""---
id: {e_id}
type: RuiRo
verification_status: {row['verification_status']}
data_origin: {row['data_origin']}
---

# {name}

## 📌 Thông tin chung
- **Mã Rủi ro**: `{e_id}`
- **Danh mục (Category)**: {row['category']}
- **Đơn vị sở hữu (Owner Unit)**: `{row['owner_unit_id']}`
- **Mức độ rủi ro tiềm tàng (Inherent Level)**: `{row['inherent_level']}`
- **Mức độ rủi ro còn lại (Residual Level)**: `{row['residual_level']}`

## 📝 Mô tả & Phân tích Nguyên nhân - Hậu quả
- **Mô tả**: {row['description']}
- **Nguyên nhân (Cause)**: {row['cause']}
- **Sự kiện Rủi ro (Event)**: {row['event']}
- **Tác động (Impact)**: {row['impact']}

## 🛡️ Biện pháp Kiểm soát liên quan (MITIGATES)
"""
        if mitigating_controls:
            for r in mitigating_controls:
                c_id = r["source_id"]
                c_name = id_to_filename.get(c_id, c_id)
                content += f"- [[{c_name}]]\n"
                content += f"  - **Loại quan hệ**: `{r['relationship_type']}`\n"
                content += f"  - **Trạng thái xác minh**: `{r['verification_status']}`\n"
                content += f"  - **Trích dẫn bằng chứng**: *\"{r['evidence_quote']}\"*\n"
                total_wikilinks += 1
        else:
            content += "*Chưa có kiểm soát liên quan được ghi nhận.*\n"

        content += "\n## ⚠️ Sự kiện Rủi ro đã ghi nhận (OBSERVED_AS)\n"
        if observed_events:
            for r in observed_events:
                ev_id = r["target_id"]
                ev_name = id_to_filename.get(ev_id, ev_id)
                content += f"- [[{ev_name}]]\n"
                content += f"  - **Loại quan hệ**: `{r['relationship_type']}`\n"
                content += f"  - **Trạng thái xác minh**: `{r['verification_status']}`\n"
                content += f"  - **Trích dẫn bằng chứng**: *\"{r['evidence_quote']}\"*\n"
                total_wikilinks += 1
        else:
            content += "*Chưa có sự kiện thực tế nào được ghi nhận.*\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        created_pages += 1

    # 4. Sinh trang cho từng KiemSoat
    controls_list = []
    for _, row in df_entities[df_entities["type"] == "KiemSoat"].iterrows():
        e_id = str(row["id"])
        name = str(row["name"])
        fname = id_to_filename[e_id]
        file_path = controls_dir / f"{fname}.md"
        controls_list.append((e_id, name, fname))

        # Tìm rủi ro mà kiểm soát này MITIGATES (source_id == e_id)
        mitigated_risks = [r for r in rels_from_source.get(e_id, []) if r["relationship_type"] == "MITIGATES"]

        content = f"""---
id: {e_id}
type: KiemSoat
verification_status: {row['verification_status']}
data_origin: {row['data_origin']}
---

# {name}

## 📌 Thông tin Kiểm soát
- **Mã Kiểm soát**: `{e_id}`
- **Loại kiểm soát (Control Type)**: {row['control_type']}
- **Tần suất (Frequency)**: {row['frequency']}
- **Vai trò phụ trách (Owner Role)**: `{row['owner_role_id']}`
- **Hiệu quả (Effectiveness)**: `{row['effectiveness']}`

## 🎯 Rủi ro giảm thiểu (MITIGATES)
"""
        if mitigated_risks:
            for r in mitigated_risks:
                rk_id = r["target_id"]
                rk_name = id_to_filename.get(rk_id, rk_id)
                content += f"- [[{rk_name}]]\n"
                content += f"  - **Loại quan hệ**: `{r['relationship_type']}`\n"
                content += f"  - **Trạng thái xác minh**: `{r['verification_status']}`\n"
                content += f"  - **Trích dẫn bằng chứng**: *\"{r['evidence_quote']}\"*\n"
                total_wikilinks += 1
        else:
            content += "*Chưa liên kết với rủi ro nào.*\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        created_pages += 1

    # 5. Sinh trang cho từng SuKienRuiRo
    events_list = []
    for _, row in df_entities[df_entities["type"] == "SuKienRuiRo"].iterrows():
        e_id = str(row["id"])
        name = str(row["name"])
        fname = id_to_filename[e_id]
        file_path = events_dir / f"{fname}.md"
        events_list.append((e_id, name, fname))

        # Tìm rủi ro mà sự kiện này thuộc về (target_id == e_id và rel_type == OBSERVED_AS)
        parent_risks = [r for r in rels_to_target.get(e_id, []) if r["relationship_type"] == "OBSERVED_AS"]

        content = f"""---
id: {e_id}
type: SuKienRuiRo
verification_status: {row['verification_status']}
data_origin: {row['data_origin']}
---

# {name}

## 📌 Chi tiết Sự kiện Rủi ro
- **Mã Sự kiện**: `{e_id}`
- **Ngày xảy ra (Occurred At)**: {row['occurred_at']}
- **Ngày phát hiện (Discovered At)**: {row['discovered_at']}
- **Mức độ nghiêm trọng (Severity)**: `{row['severity']}`
- **Tổn thất tài chính (VND)**: {row['loss_amount_vnd']} VNĐ
- **Mô tả sự kiện**: {row['description']}

## 🔗 Rủi ro tương ứng (OBSERVED_AS)
"""
        if parent_risks:
            for r in parent_risks:
                rk_id = r["source_id"]
                rk_name = id_to_filename.get(rk_id, rk_id)
                content += f"- [[{rk_name}]]\n"
                content += f"  - **Loại quan hệ**: `{r['relationship_type']}`\n"
                content += f"  - **Trạng thái xác minh**: `{r['verification_status']}`\n"
                content += f"  - **Trích dẫn bằng chứng**: *\"{r['evidence_quote']}\"*\n"
                total_wikilinks += 1
        else:
            content += "*Chưa liên kết với rủi ro gốc nào.*\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        created_pages += 1

    # 6. Sinh trang wiki/Home.md
    home_content = f"""---
id: HOME
type: Dashboard
verification_status: VERIFIED
data_origin: system
---

# 🏠 Wiki Risk Graph - Cổng thông tin Quản trị Rủi ro

Được khởi tạo từ dữ liệu chuẩn hóa [entities.csv](../outputs/entities.csv) và [relations.csv](../outputs/relations.csv).

## 📊 Thống kê Hệ thống Đồ thị
- **Tổng số Nodes (Thực thể)**: {len(df_entities)}
  - 🛡️ **Biện pháp Kiểm soát (`KiemSoat`)**: {len(controls_list)}
  - ⚠️ **Hồ sơ Rủi ro (`RuiRo`)**: {len(risks_list)}
  - 🚨 **Sự kiện Rủi ro (`SuKienRuiRo`)**: {len(events_list)}
- **Tổng số Edges (Mối quan hệ)**: {len(df_relations)}
  - `MITIGATES` (`KiemSoat` -> `RuiRo`): {len(df_relations[df_relations['relationship_type'] == 'MITIGATES'])}
  - `OBSERVED_AS` (`RuiRo` -> `SuKienRuiRo`): {len(df_relations[df_relations['relationship_type'] == 'OBSERVED_AS'])}
- **Tổng số Wikilinks liên kết nội bộ**: {total_wikilinks}

---

## 🗂️ Danh mục Trang Chi tiết

### 1. 🛡️ Danh sách Biện pháp Kiểm soát (`KiemSoat`)
"""
    for c_id, c_name, fname in controls_list:
        home_content += f"- [[{fname}|{c_id} - {c_name}]]\n"
        total_wikilinks += 1

    home_content += "\n### 2. ⚠️ Danh sách Hồ sơ Rủi ro (`RuiRo`)\n"
    for r_id, r_name, fname in risks_list:
        home_content += f"- [[{fname}|{r_id} - {r_name}]]\n"
        total_wikilinks += 1

    home_content += "\n### 3. 🚨 Danh sách Sự kiện Rủi ro (`SuKienRuiRo`)\n"
    for e_id, e_name, fname in events_list:
        home_content += f"- [[{fname}|{e_id} - {e_name}]]\n"
        total_wikilinks += 1

    home_path = WIKI_DIR / "Home.md"
    with open(home_path, "w", encoding="utf-8") as f:
        f.write(home_content)
    created_pages += 1

    print("\n" + "=" * 80)
    print("BÁO CÁO KẾT QUẢ SINH WIKI MARKDOWN")
    print("=" * 80)
    print(f"✅ Tổng số trang Wiki Markdown đã sinh: {created_pages} trang")
    print(f"  - wiki/Home.md: 1 trang")
    print(f"  - wiki/risks/: {len(risks_list)} trang")
    print(f"  - wiki/controls/: {len(controls_list)} trang")
    print(f"  - wiki/events/: {len(events_list)} trang")
    print(f"✅ Tổng số Wikilinks (Internal links): {total_wikilinks} wikilinks")

    # In ví dụ đường đi KiemSoat -> RuiRo -> SuKienRuiRo
    print("\n🌐 VÍ DỤ ĐƯỜNG ĐI CHI TIẾT (KiemSoat -> RuiRo -> SuKienRuiRo):")
    # Lấy ví dụ từ KS-001 -> RR-001 -> SK-001
    sample_ks = "KS-001"
    ks_name = id_to_filename.get(sample_ks, sample_ks)
    ks_rels = [r for r in rels_from_source.get(sample_ks, []) if r["relationship_type"] == "MITIGATES"]

    if ks_rels:
        sample_rr = ks_rels[0]["target_id"]
        rr_name = id_to_filename.get(sample_rr, sample_rr)
        rr_rels = [r for r in rels_from_source.get(sample_rr, []) if r["relationship_type"] == "OBSERVED_AS"]

        if rr_rels:
            sample_sk = rr_rels[0]["target_id"]
            sk_name = id_to_filename.get(sample_sk, sample_sk)

            print(f"  [KiemSoat] [[{ks_name}]] ({sample_ks})")
            print(f"       │")
            print(f"       │  -[:MITIGATES]-> (Confidence: {ks_rels[0]['confidence']})")
            print(f"       v")
            print(f"  [RuiRo]    [[{rr_name}]] ({sample_rr})")
            print(f"       │")
            print(f"       │  -[:OBSERVED_AS]-> (Confidence: {rr_rels[0]['confidence']})")
            print(f"       v")
            print(f"  [SuKien]   [[{sk_name}]] ({sample_sk})")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    build_wiki()
