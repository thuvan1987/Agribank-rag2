#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script chuẩn hóa dữ liệu thành Node (entities.csv) và Edge (relations.csv)
Phục vụ Wiki Risk Graph (Buổi 13 - Bước 2)
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Đảm bảo thư mục outputs tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FILE = DATA_DIR / "risk_profiles_seed.csv"
CONTROL_FILE = DATA_DIR / "controls_seed.csv"
EVENT_FILE = DATA_DIR / "risk_events_seed.csv"
REL_FILE = DATA_DIR / "relationships_seed.csv"

def build_entities_and_relations():
    print("=" * 80)
    print("BẮT ĐẦU CHUẨN HÓA DỮ LIỆU ENTITIES VÀ RELATIONS")
    print("=" * 80)

    # 1. Kiểm tra sự tồn tại của dữ liệu đầu vào
    for file_path in [RISK_FILE, CONTROL_FILE, EVENT_FILE, REL_FILE]:
        if not file_path.exists():
            print(f"❌ Error: File {file_path.name} không tồn tại tại {file_path}")
            sys.exit(1)

    # Đọc các file CSV gốc
    df_risk = pd.read_csv(RISK_FILE)
    df_control = pd.read_csv(CONTROL_FILE)
    df_event = pd.read_csv(EVENT_FILE)
    df_rel = pd.read_csv(REL_FILE)

    entities = []

    # 2. Mapping risk_profiles_seed.csv -> type = RuiRo
    for _, row in df_risk.iterrows():
        entities.append({
            "id": str(row["id"]),
            "type": "RuiRo",
            "name": str(row["name"]),
            "description": str(row["description"]),
            "source_file": "risk_profiles_seed.csv",
            "data_origin": str(row["data_origin"]),
            "verification_status": str(row["verification_status"]),
            # Các thuộc tính nghiệp vụ riêng cho RuiRo
            "category": str(row.get("category", "")),
            "cause": str(row.get("cause", "")),
            "event": str(row.get("event", "")),
            "impact": str(row.get("impact", "")),
            "inherent_level": str(row.get("inherent_level", "")),
            "residual_level": str(row.get("residual_level", "")),
            "owner_unit_id": str(row.get("owner_unit_id", "")),
            # Cột riêng của các loại entity khác để trống
            "control_type": "",
            "frequency": "",
            "owner_role_id": "",
            "effectiveness": "",
            "risk_id": "",
            "occurred_at": "",
            "discovered_at": "",
            "severity": "",
            "loss_amount_vnd": ""
        })

    # 3. Mapping controls_seed.csv -> type = KiemSoat
    for _, row in df_control.iterrows():
        entities.append({
            "id": str(row["id"]),
            "type": "KiemSoat",
            "name": str(row["name"]),
            "description": "", # controls_seed không có description, giữ nguyên chuẩn hóa
            "source_file": "controls_seed.csv",
            "data_origin": str(row["data_origin"]),
            "verification_status": str(row["verification_status"]),
            # Thuộc tính nghiệp vụ RuiRo
            "category": "",
            "cause": "",
            "event": "",
            "impact": "",
            "inherent_level": "",
            "residual_level": "",
            "owner_unit_id": "",
            # Thuộc tính nghiệp vụ riêng cho KiemSoat
            "control_type": str(row.get("control_type", "")),
            "frequency": str(row.get("frequency", "")),
            "owner_role_id": str(row.get("owner_role_id", "")),
            "effectiveness": str(row.get("effectiveness", "")),
            # Thuộc tính nghiệp vụ SuKienRuiRo
            "risk_id": "",
            "occurred_at": "",
            "discovered_at": "",
            "severity": "",
            "loss_amount_vnd": ""
        })

    # 4. Mapping risk_events_seed.csv -> type = SuKienRuiRo
    for _, row in df_event.iterrows():
        entities.append({
            "id": str(row["id"]),
            "type": "SuKienRuiRo",
            "name": str(row.get("description", ""))[:50] + "...", # Dùng mô tả ngắn cho name nếu không có cột name
            "description": str(row.get("description", "")),
            "source_file": "risk_events_seed.csv",
            "data_origin": str(row["data_origin"]),
            "verification_status": str(row["verification_status"]),
            # Thuộc tính nghiệp vụ RuiRo
            "category": "",
            "cause": "",
            "event": "",
            "impact": "",
            "inherent_level": "",
            "residual_level": "",
            "owner_unit_id": "",
            # Thuộc tính nghiệp vụ KiemSoat
            "control_type": "",
            "frequency": "",
            "owner_role_id": "",
            "effectiveness": "",
            # Thuộc tính nghiệp vụ riêng cho SuKienRuiRo
            "risk_id": str(row.get("risk_id", "")),
            "occurred_at": str(row.get("occurred_at", "")),
            "discovered_at": str(row.get("discovered_at", "")),
            "severity": str(row.get("severity", "")),
            "loss_amount_vnd": str(row.get("loss_amount_vnd", ""))
        })

    df_entities = pd.DataFrame(entities)
    entities_out_path = OUTPUT_DIR / "entities.csv"
    df_entities.to_csv(entities_out_path, index=False, encoding="utf-8")
    print(f"✅ Đã lưu {len(df_entities)} entities vào {entities_out_path}")

    # 5. Chuẩn hóa relations.csv từ relationships_seed.csv
    # Giữ nguyên verification_status gốc (không đổi PROPOSED -> VERIFIED)
    rel_cols = ["source_id", "relationship_type", "target_id", "source", "evidence_quote", "confidence", "verification_status", "data_origin"]
    df_relations = df_rel[rel_cols].copy()
    
    relations_out_path = OUTPUT_DIR / "relations.csv"
    df_relations.to_csv(relations_out_path, index=False, encoding="utf-8")
    print(f"✅ Đã lưu {len(df_relations)} relations vào {relations_out_path}")

    # 6. Kiểm tra Tham chiếu mồ côi (Orphan references)
    entity_ids = set(df_entities["id"].astype(str))
    orphan_errors = []

    for idx, row in df_relations.iterrows():
        s_id = str(row["source_id"])
        t_id = str(row["target_id"])

        if s_id not in entity_ids:
            orphan_errors.append(f"Dòng {idx+1}: source_id '{s_id}' không tồn tại trong entities.csv")
        if t_id not in entity_ids:
            orphan_errors.append(f"Dòng {idx+1}: target_id '{t_id}' không tồn tại trong entities.csv")

    print("\n" + "=" * 80)
    print("THỐNG KÊ KẾT QUẢ BƯỚC CHUẨN HÓA")
    print("=" * 80)

    # In số entity theo từng type
    type_counts = df_entities["type"].value_counts()
    print("\n📊 Số lượng Entity theo từng loại (Type):")
    for e_type, count in type_counts.items():
        print(f"  - {e_type}: {count} entities")

    # In số relation theo từng relationship_type
    rel_counts = df_relations["relationship_type"].value_counts()
    print("\n🔗 Số lượng Relation theo từng loại (Relationship Type):")
    for r_type, count in rel_counts.items():
        print(f"  - {r_type}: {count} relations")

    # In trạng thái kiểm tra Orphan
    print("\n🔎 Kiểm tra tham chiếu mồ côi (Orphan Reference Check):")
    if orphan_errors:
        print(f"❌ PHÁT HIỆN {len(orphan_errors)} LỖI ORPHAN REFERENCE:")
        for err in orphan_errors:
            print(f"  - {err}")
    else:
        print("✅ PASS: Không có orphan reference nào. Tất cả source_id và target_id đều tồn tại hợp lệ trong entities.csv!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    build_entities_and_relations()
