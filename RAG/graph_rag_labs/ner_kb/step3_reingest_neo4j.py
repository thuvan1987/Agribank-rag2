#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Tái nạp dữ liệu Đồ thị mở rộng vào Neo4j (30 tài liệu)
Bao gồm:
1. HTML Chunking: Phân đoạn HTML 30 tài liệu và trích xuất cấu trúc cây phân cấp (Document, Chapter, Section, Article, Clause).
2. Embedding Generation: Tạo vector embedding (384-dim, vi-distilled-msmarco-MiniLM-L12-cos-v5) cho toàn bộ các chunks.
3. Neo4j Loading: Nạp toàn bộ Nút (:Document, :Chunk), Quan hệ (:CAN_CU, :THAY_THE, :SUA_DOI_BO_SUNG, :HOP_NHAT, :VAN_BAN_BO_SUNG, :PART_OF, :PARENT_OF, :NEXT) và khởi tạo Vector Index trên Neo4j.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Tăng giới hạn trường CSV
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
KB_HOPS_DIR = BASE_DIR.parent / "kb+hops"
ENV_FILE = BASE_DIR / ".env"

if not ENV_FILE.exists():
    ENV_FILE = KB_HOPS_DIR / ".env"

# Import HTML Chunker từ kb+hops
sys.path.insert(0, str(KB_HOPS_DIR))
from step1_html_chunker import parse_html_document, LEVEL_RANKS

METADATA_CSV = BASE_DIR / "metadata.csv"
CONTENT_CSV = BASE_DIR / "content.csv"
RELATIONSHIPS_CSV = BASE_DIR / "relationships.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHUNKS_FILE = OUTPUT_DIR / "chunks.json"
RELATIONS_FILE = OUTPUT_DIR / "relationships.json"
CHUNKS_EMBEDDINGS_FILE = OUTPUT_DIR / "chunks_with_embeddings.json"

MODEL_NAME = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"
BATCH_SIZE = 500


def run_html_chunking():
    print("\n--- 1. CHUNKING HTML & TRÍCH XUẤT CẤU TRÚC PHÂN CẤP ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if METADATA_CSV.exists():
        with open(METADATA_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["id"]] = row

    all_chunks = []
    all_relations = []

    with open(CONTENT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["id"]
            html_content = row["content_html"]
            meta = metadata.get(doc_id, {})
            chunks, relations = parse_html_document(doc_id, html_content, meta)
            all_chunks.extend(chunks)
            all_relations.extend(relations)

    with open(CHUNKS_FILE, mode="w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(RELATIONS_FILE, mode="w", encoding="utf-8") as f:
        json.dump(all_relations, f, ensure_ascii=False, indent=2)

    print(f"✓ Hoàn tất Chunking: Tạo {len(all_chunks)} chunks và {len(all_relations)} quan hệ nội bộ.")


def run_embedding_generation():
    print("\n--- 2. TẠO VECTOR EMBEDDINGS CHO CHUNKS ---")
    with open(CHUNKS_FILE, mode="r", encoding="utf-8") as f:
        chunks = json.load(f)

    device = "cpu"
    print(f"✓ Nạp mô hình SentenceTransformer '{MODEL_NAME}' trên {device.upper()}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [c.get("clean_text", "") for c in chunks]
    print(f"✓ Đang tạo embedding cho {len(texts)} chunks...")

    start_t = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"✓ Tạo embedding hoàn tất sau {time.time() - start_t:.2f}s.")

    for idx, c in enumerate(chunks):
        c["embedding"] = embeddings[idx].tolist()

    with open(CHUNKS_EMBEDDINGS_FILE, mode="w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"✓ Đã lưu file vector embeddings tại {CHUNKS_EMBEDDINGS_FILE}")


def try_create_database(driver, target_db):
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{target_db}` IF NOT EXISTS")
            return target_db
    except Exception:
        return target_db


def run_neo4j_ingestion():
    print("\n--- 3. TÁI NẠP ĐỒ THỊ VÀO NEO4J ---")
    load_dotenv(dotenv_path=ENV_FILE)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD", "password").strip()
    target_db = os.getenv("NEO4J_DATABASE", "kb-hops").strip()

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print(f"✓ Kết nối thành công Neo4j tại {uri}.")

    active_db = try_create_database(driver, target_db)
    try:
        with driver.session(database=active_db) as session:
            session.run("RETURN 1")
    except Exception:
        active_db = "neo4j"

    print(f"✓ Sử dụng Database: '{active_db}'")

    with driver.session(database=active_db) as session:
        # Xóa dữ liệu cũ để tái nạp toàn bộ 30 văn bản mở rộng
        print("🧹 Đang xóa dữ liệu đồ thị cũ để tái nạp...")
        session.run("MATCH (n) DETACH DELETE n")

        # Constraints & Vector Index
        session.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

        try:
            session.run("""
                CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            print("✓ Tạo Vector Index 'chunk_vector_index' (384 dims) THÀNH CÔNG.")
        except Exception as e:
            print(f"⚠️ Cảnh báo Vector Index: {e}")

        # Nạp Document nodes
        doc_list = []
        with open(METADATA_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_list.append(row)

        cypher_doc = """
        UNWIND $batch AS row
        MERGE (d:Document {id: row.id})
        SET d.title = row.title,
            d.so_ky_hieu = row.so_ky_hieu,
            d.ngay_ban_hanh = row.ngay_ban_hanh,
            d.loai_van_ban = row.loai_van_ban,
            d.ngay_co_hieu_luc = row.ngay_co_hieu_luc,
            d.ngay_het_hieu_luc = row.ngay_het_hieu_luc,
            d.nguon_thu_thap = row.nguon_thu_thap,
            d.nganh = row.nganh,
            d.linh_vuc = row.linh_vuc,
            d.co_quan_ban_hanh = row.co_quan_ban_hanh,
            d.nguoi_ky = row.nguoi_ky,
            d.tinh_trang_hieu_luc = row.tinh_trang_hieu_luc
        """
        session.run(cypher_doc, batch=doc_list)
        print(f"✓ Đã nạp {len(doc_list)} nút (:Document).")

        # Nạp Quan hệ giữa các Document từ relationships.csv
        doc_rels = []
        with open(RELATIONSHIPS_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_rels.append(row)

        for rel in doc_rels:
            rel_type = rel.get("relationship_type", "LIEN_KET").strip().upper()
            cypher_doc_rel = f"""
            MATCH (d1:Document {{id: $doc_id}}), (d2:Document {{id: $other_doc_id}})
            MERGE (d1)-[r:{rel_type}]->(d2)
            SET r.relationship = $relationship
            """
            session.run(cypher_doc_rel, doc_id=rel["doc_id"], other_doc_id=rel["other_doc_id"], relationship=rel.get("relationship", ""))

        print(f"✓ Đã nạp {len(doc_rels)} quan hệ giữa các nút Document.")

        # Nạp Chunks kèm Embedding
        with open(CHUNKS_EMBEDDINGS_FILE, mode="r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        cypher_chunk = """
        UNWIND $batch AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.doc_id = row.doc_id,
            c.level = row.level,
            c.title = row.title,
            c.clean_text = row.clean_text,
            c.parent_id = row.parent_id,
            c.embedding = row.embedding
        """
        for i in range(0, len(chunks_data), BATCH_SIZE):
            batch = chunks_data[i : i + BATCH_SIZE]
            session.run(cypher_chunk, batch=batch)

        print(f"✓ Đã nạp {len(chunks_data)} nút (:Chunk).")

        # Nạp Quan hệ cấp Chunk (PART_OF, PARENT_OF, NEXT)
        with open(RELATIONS_FILE, mode="r", encoding="utf-8") as f:
            rels_data = json.load(f)

        part_of_list = [r for r in rels_data if r["type"] == "PART_OF"]
        parent_of_list = [r for r in rels_data if r["type"] == "PARENT_OF"]
        next_list = [r for r in rels_data if r["type"] == "NEXT"]

        cypher_part_of = """
        UNWIND $batch AS row
        MATCH (c:Chunk {chunk_id: row.source})
        WITH c, row, replace(row.target, 'doc_', '') AS doc_id_clean
        MATCH (d:Document {id: doc_id_clean})
        MERGE (c)-[:PART_OF]->(d)
        """
        for i in range(0, len(part_of_list), BATCH_SIZE):
            session.run(cypher_part_of, batch=part_of_list[i : i + BATCH_SIZE])

        cypher_parent_of = """
        UNWIND $batch AS row
        MATCH (c1:Chunk {chunk_id: row.source}), (c2:Chunk {chunk_id: row.target})
        MERGE (c1)-[:PARENT_OF]->(c2)
        """
        for i in range(0, len(parent_of_list), BATCH_SIZE):
            session.run(cypher_parent_of, batch=parent_of_list[i : i + BATCH_SIZE])

        cypher_next = """
        UNWIND $batch AS row
        MATCH (c1:Chunk {chunk_id: row.source}), (c2:Chunk {chunk_id: row.target})
        MERGE (c1)-[:NEXT]->(c2)
        """
        for i in range(0, len(next_list), BATCH_SIZE):
            session.run(cypher_next, batch=next_list[i : i + BATCH_SIZE])

        print(f"✓ Đã nạp {len(rels_data)} quan hệ cấp Chunk (PART_OF, PARENT_OF, NEXT).")

    driver.close()
    print("\n🎉 HOÀN THÀNH TÁI NẠP ĐỒ THỊ TRI THỨC MỞ RỘNG (30 TÀI LIỆU) THÀNH CÔNG!")


def main():
    print("==================================================")
    print("🚀 BƯỚC 3: TÁI NẠP DỮ LIỆU ĐỒ THỊ MỞ RỘNG VÀO NEO4J")
    print("==================================================")
    run_html_chunking()
    run_embedding_generation()
    run_neo4j_ingestion()


if __name__ == "__main__":
    main()
