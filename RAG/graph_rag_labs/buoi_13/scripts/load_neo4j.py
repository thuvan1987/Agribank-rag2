#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script import dữ liệu Wiki Risk Graph vào Neo4j Database (Buổi 13 - Bước 6)
Đọc từ outputs/entities.csv và outputs/relations.csv
Đảm bảo tính Idempotent (chạy lại không tạo duplicate) bằng MERGE và Parameterized Cypher.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
OUTPUTS_DIR = BASE_DIR / "outputs"

ENTITIES_FILE = OUTPUTS_DIR / "entities.csv"
RELATIONS_FILE = OUTPUTS_DIR / "relations.csv"
SCHEMA_FILE = BASE_DIR / "cypher" / "schema.cypher"

# Nạp file môi trường .env
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv(BASE_DIR.parent / "ner_kb" / ".env")

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
password = os.getenv("NEO4J_PASSWORD", "")
database = os.getenv("NEO4J_DATABASE", "neo4j")

def load_to_neo4j():
    print("=" * 80)
    print("BẮT ĐẦU IMPORT KNOWLEDGE GRAPH VÀO NEO4J")
    print("=" * 80)

    # 1. Kiểm tra thư viện neo4j
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("⚠️ Thư viện 'neo4j' chưa được cài đặt trong môi trường Python.")
        print("💡 Bạn có thể cài đặt bằng lệnh: pip install neo4j")
        print("ℹ️ Quá trình kiểm tra bỏ qua import Neo4j (Wiki Markdown đã được tạo hoàn chỉnh).")
        return

    # 2. Kiểm tra thông tin kết nối
    if not password:
        print("⚠️ Không tìm thấy NEO4J_PASSWORD trong file .env!")
        print("💡 HƯỚNG DẪN KẾT NỐI NEO4J:")
        print("   1. Đảm bảo Neo4j Desktop 2.0 hoặc Neo4j Docker đang chạy.")
        print("   2. Cấu hình mật khẩu trong file .env tại thư mục project.")
        return

    # 3. Kiểm tra file dữ liệu
    if not ENTITIES_FILE.exists() or not RELATIONS_FILE.exists():
        print(f"❌ Khôg tìm thấy {ENTITIES_FILE} hoặc {RELATIONS_FILE}")
        return

    print(f"📡 Đang kết nối tới Neo4j tại URI: {uri} (Database: {database})...")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ Kết nối Neo4j thành công!")
    except Exception as e:
        print(f"\n⚠️ Không thể kết nối tới Neo4j: {e}")
        print("\n💡 HƯỚNG DẪN XỬ LÝ:")
        print("   1. Kiểm tra xem Neo4j Server / Neo4j Desktop đã Khởi động (Start) chưa.")
        print("   2. Kiểm tra Cổng kết nối Bolt (mặc định 7687) trong file .env.")
        print("   3. Kiểm tra Username và Password trong file .env.")
        print("ℹ️ Hệ thống giữ nguyên tất cả dữ liệu Wiki Markdown và CSV đã tạo thành công trước đó.")
        return

    df_entities = pd.read_csv(ENTITIES_FILE).fillna("")
    df_relations = pd.read_csv(RELATIONS_FILE).fillna("")

    with driver.session(database=database) as session:
        # 4. Áp dụng Constraints & Indexes
        print("\n🛠️ 1. Khởi tạo Uniqueness Constraints và Indexes...")
        if SCHEMA_FILE.exists():
            with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
                cypher_statements = [stmt.strip() for stmt in f.read().split(";") if stmt.strip()]
            for stmt in cypher_statements:
                if stmt.startswith("//"):
                    continue
                try:
                    session.run(stmt)
                except Exception as ex:
                    # Ignore if constraint/index already exists
                    pass

        # 5. Import Nodes với MERGE (Parameterized Cypher)
        print("\n📦 2. Import Nodes vào Neo4j (MERGE)...")
        node_counts = {"RuiRo": 0, "KiemSoat": 0, "SuKienRuiRo": 0}

        for _, row in df_entities.iterrows():
            e_type = str(row["type"])
            props = row.to_dict()

            if e_type == "RuiRo":
                query = """
                MERGE (r:RuiRo {id: $id})
                SET r.name = $name,
                    r.description = $description,
                    r.category = $category,
                    r.cause = $cause,
                    r.event = $event,
                    r.impact = $impact,
                    r.inherent_level = $inherent_level,
                    r.residual_level = $residual_level,
                    r.owner_unit_id = $owner_unit_id,
                    r.data_origin = $data_origin,
                    r.verification_status = $verification_status,
                    r.source_file = $source_file
                """
                session.run(query, props)
                node_counts["RuiRo"] += 1

            elif e_type == "KiemSoat":
                query = """
                MERGE (k:KiemSoat {id: $id})
                SET k.name = $name,
                    k.control_type = $control_type,
                    k.frequency = $frequency,
                    k.owner_role_id = $owner_role_id,
                    k.effectiveness = $effectiveness,
                    k.data_origin = $data_origin,
                    k.verification_status = $verification_status,
                    k.source_file = $source_file
                """
                session.run(query, props)
                node_counts["KiemSoat"] += 1

            elif e_type == "SuKienRuiRo":
                query = """
                MERGE (s:SuKienRuiRo {id: $id})
                SET s.name = $name,
                    s.description = $description,
                    s.risk_id = $risk_id,
                    s.occurred_at = $occurred_at,
                    s.discovered_at = $discovered_at,
                    s.severity = $severity,
                    s.loss_amount_vnd = $loss_amount_vnd,
                    s.data_origin = $data_origin,
                    s.verification_status = $verification_status,
                    s.source_file = $source_file
                """
                session.run(query, props)
                node_counts["SuKienRuiRo"] += 1

        print(f"  - Node :RuiRo: {node_counts['RuiRo']}")
        print(f"  - Node :KiemSoat: {node_counts['KiemSoat']}")
        print(f"  - Node :SuKienRuiRo: {node_counts['SuKienRuiRo']}")

        # 6. Import Relationships với MERGE (Parameterized Cypher)
        print("\n🔗 3. Import Relationships vào Neo4j (MERGE)...")
        rel_counts = {"MITIGATES": 0, "OBSERVED_AS": 0}

        for _, row in df_relations.iterrows():
            rel_type = str(row["relationship_type"])
            props = row.to_dict()

            if rel_type == "MITIGATES":
                query = """
                MATCH (k:KiemSoat {id: $source_id})
                MATCH (r:RuiRo {id: $target_id})
                MERGE (k)-[rel:MITIGATES]->(r)
                SET rel.source = $source,
                    rel.evidence_quote = $evidence_quote,
                    rel.confidence = $confidence,
                    rel.verification_status = $verification_status,
                    rel.data_origin = $data_origin
                """
                session.run(query, props)
                rel_counts["MITIGATES"] += 1

            elif rel_type == "OBSERVED_AS":
                query = """
                MATCH (r:RuiRo {id: $source_id})
                MATCH (s:SuKienRuiRo {id: $target_id})
                MERGE (r)-[rel:OBSERVED_AS]->(s)
                SET rel.source = $source,
                    rel.evidence_quote = $evidence_quote,
                    rel.confidence = $confidence,
                    rel.verification_status = $verification_status,
                    rel.data_origin = $data_origin
                """
                session.run(query, props)
                rel_counts["OBSERVED_AS"] += 1

        print(f"  - Relationship [:MITIGATES]: {rel_counts['MITIGATES']}")
        print(f"  - Relationship [:OBSERVED_AS]: {rel_counts['OBSERVED_AS']}")

        # 7. Kiểm tra tổng số Node và Relationship thực tế từ DB
        res_nodes = session.run("MATCH (n) RETURN count(n) AS total_nodes").single()
        res_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS total_rels").single()

        print("\n📊 4. Xác nhận số lượng thực tế trong Neo4j Database:")
        print(f"  - Tổng số Node trong DB: {res_nodes['total_nodes']}")
        print(f"  - Tổng số Relationship trong DB: {res_rels['total_rels']}")

    driver.close()
    print("\n✅ Hoàn thành Import Knowledge Graph vào Neo4j thành công!")
    print("=" * 80)

if __name__ == "__main__":
    load_to_neo4j()
