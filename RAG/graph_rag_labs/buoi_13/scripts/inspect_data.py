#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm tra và phân tích dữ liệu cho Wiki Risk Graph (Buổi 13 - Bước 1)
"""

import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CSV_FILES = {
    "risk_profiles_seed.csv": DATA_DIR / "risk_profiles_seed.csv",
    "controls_seed.csv": DATA_DIR / "controls_seed.csv",
    "risk_events_seed.csv": DATA_DIR / "risk_events_seed.csv",
    "relationships_seed.csv": DATA_DIR / "relationships_seed.csv"
}

def inspect_csv_files():
    print("=" * 80)
    print("BÁO CÁO KIỂM TRA VÀ PHÂN TÍCH DỮ LIỆU WIKI RISK GRAPH")
    print("=" * 80)

    dfs = {}
    for name, path in CSV_FILES.items():
        if not path.exists():
            print(f"❌ File {name} không tồn tại tại {path}")
            continue
        dfs[name] = pd.read_csv(path)
        print(f"\n📁 File: {name}")
        print(f"  - Số dòng: {len(dfs[name])}")
        print(f"  - Số cột: {len(dfs[name].columns)}")
        print(f"  - Danh sách cột: {list(dfs[name].columns)}")
        print("  - Số giá trị NULL theo từng cột:")
        null_counts = dfs[name].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                print(f"    + {col}: {count} nulls")
            else:
                print(f"    + {col}: 0 nulls")
        
        # Check duplicates
        dup_count = dfs[name].duplicated().sum()
        print(f"  - Số dòng trùng lặp (Duplicate rows): {dup_count}")
        
        # Unique IDs if 'id' in columns
        if 'id' in dfs[name].columns:
            id_dups = dfs[name]['id'].duplicated().sum()
            print(f"  - Khóa chính 'id': {dfs[name]['id'].nunique()} unique, {id_dups} duplicate IDs")

    print("\n" + "=" * 80)
    print("PHÂN TÍCH KHÓA THAM CHIẾU VÀ MỐI QUAN HỆ")
    print("=" * 80)

    # Risk Profiles
    risk_df = dfs.get("risk_profiles_seed.csv")
    control_df = dfs.get("controls_seed.csv")
    event_df = dfs.get("risk_events_seed.csv")
    rel_df = dfs.get("relationships_seed.csv")

    risk_ids = set(risk_df['id'].dropna().astype(str)) if risk_df is not None and 'id' in risk_df.columns else set()
    control_ids = set(control_df['id'].dropna().astype(str)) if control_df is not None and 'id' in control_df.columns else set()
    event_ids = set(event_df['id'].dropna().astype(str)) if event_df is not None and 'id' in event_df.columns else set()

    # Check risk_events foreign key: risk_id -> risk_profiles.id
    if event_df is not None and 'risk_id' in event_df.columns:
        event_risk_refs = set(event_df['risk_id'].dropna().astype(str))
        missing_risk_refs = event_risk_refs - risk_ids
        print(f"\n🔍 Tham chiếu từ risk_events_seed.csv ('risk_id') -> risk_profiles_seed.csv ('id'):")
        print(f"  - Tổng số risk_id duy nhất trong events: {len(event_risk_refs)}")
        print(f"  - Số risk_id thiếu/không tồn tại trong risk_profiles: {len(missing_risk_refs)}")
        if missing_risk_refs:
            print(f"    ⚠️ Mới/thiếu: {missing_risk_refs}")

    # Check relationships_seed.csv
    if rel_df is not None:
        print(f"\n🔍 Chi tiết quan hệ trong relationships_seed.csv:")
        rel_types = rel_df['relationship_type'].value_counts()
        print("  - Danh sách relationship_type:")
        for r_type, count in rel_types.items():
            print(f"    + {r_type}: {count} bản ghi")

        all_known_ids = risk_ids | control_ids | event_ids
        sources = rel_df['source_id'].astype(str)
        targets = rel_df['target_id'].astype(str)

        missing_sources = set(sources) - all_known_ids
        missing_targets = set(targets) - all_known_ids

        print(f"  - Tồn tại source_id không có trong node master: {len(missing_sources)}")
        if missing_sources:
            print(f"    ⚠️ Missing source_ids: {missing_sources}")

        print(f"  - Tồn tại target_id không có trong node master: {len(missing_targets)}")
        if missing_targets:
            print(f"    ⚠️ Missing target_ids: {missing_targets}")

    print("\n" + "=" * 80)
    print("KẾT THÚC BÁO CÁO")
    print("=" * 80)

if __name__ == "__main__":
    inspect_csv_files()
