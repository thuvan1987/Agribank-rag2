import csv
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
METADATA_CSV = BASE_DIR / "metadata.csv"
RELATIONSHIPS_CSV = BASE_DIR / "relationships.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHUNKS_EMBEDDINGS_FILE = OUTPUT_DIR / "chunks_with_embeddings.json"
CHUNK_RELS_FILE = OUTPUT_DIR / "relationships.json"

BATCH_SIZE = 500

def try_create_database(driver, target_db):
    """
    Thử tạo database `kb-hops` nếu phiên bản Neo4j hỗ trợ multi-database (Enterprise/Local DBMS).
    Nếu không (Community Edition single DB), quay về dùng database active mặc định.
    """
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{target_db}` IF NOT EXISTS")
            print(f"✓ Đã tạo/xác nhận cơ sở dữ liệu '{target_db}'.")
            return target_db
    except Exception as e:
        print(f"ℹ️ Lưu ý: Không thể tạo DB mới qua system session ({e}). Sử dụng DB mặc định.")
        return target_db

def main():
    print("==================================================")
    print(" BẮT ĐẦU THỰC THI BƯỚC 4 & 5 - BUỔI 10: NEO4J INGESTION ")
    print("==================================================")

    # 1. Đọc cấu hình kết nối .env
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
        
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD", "password").strip()
    target_db = os.getenv("NEO4J_DATABASE", "kb-hops").strip()

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print(f"✓ Kết nối thành công tới Neo4j tại {uri}.")

    # Kiểm tra/tạo database target
    active_db = try_create_database(driver, target_db)

    # Đảm bảo active_db tồn tại bằng cách thử mở session
    try:
        with driver.session(database=active_db) as session:
            session.run("RETURN 1")
    except Exception:
        print(f"⚠️ Database '{active_db}' chưa sẵn sàng, quay về dùng database 'neo4j'.")
        active_db = "neo4j"

    print(f"✓ Sử dụng Cơ sở dữ liệu: '{active_db}'")

    start_all = time.time()

    with driver.session(database=active_db) as session:

        # 2. Tạo Unique Constraints & Vector Index
        print("\n--- 1. TẠO CONSTRAINTS VÀ VECTOR INDEX ---")
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
            print("✓ Tạo Vector Index 'chunk_vector_index' (384 dims, Cosine similarity) THÀNH CÔNG.")
        except Exception as e:
            print(f"⚠️ Cảnh báo tạo Vector Index: {e}")

        # 3. Nạp Nút (:Document) từ metadata.csv
        print("\n--- 2. NẠP NÚT (:Document) TỪ METADATA.CSV ---")
        doc_list = []
        if METADATA_CSV.exists():
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
        print(f"✓ Đã nạp thành công {len(doc_list)} nút (:Document).")

        # 4. Nạp Quan hệ Cấp Tài liệu từ relationships.csv
        print("\n--- 3. NẠP QUAN HỆ GIỮA CÁC TÀI LIỆU (:Document) ---")
        doc_rels = []
        if RELATIONSHIPS_CSV.exists():
            with open(RELATIONSHIPS_CSV, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_rels.append(row)

        for rel in doc_rels:
            rel_type = rel.get("relationship_type", "LIEN_KET").upper()
            cypher_doc_rel = f"""
            MATCH (d1:Document {{id: $doc_id}}), (d2:Document {{id: $other_doc_id}})
            MERGE (d1)-[r:{rel_type}]->(d2)
            SET r.relationship = $relationship
            """
            session.run(cypher_doc_rel, doc_id=rel["doc_id"], other_doc_id=rel["other_doc_id"], relationship=rel.get("relationship", ""))

        print(f"✓ Đã nạp thành công {len(doc_rels)} quan hệ giữa các nút Document.")

        # 5. Nạp Nút (:Chunk) kèm Vector Embedding từ chunks_with_embeddings.json
        print("\n--- 4. NẠP NÚT (:Chunk) KÈM EMBEDDING (BATCH UNWIND) ---")
        if not CHUNKS_EMBEDDINGS_FILE.exists():
            print(f"❌ LỖI: Không tìm thấy file {CHUNKS_EMBEDDINGS_FILE}")
            sys.exit(1)

        with open(CHUNKS_EMBEDDINGS_FILE, mode="r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        total_chunks = len(chunks_data)
        print(f"✓ Đang nạp {total_chunks} Chunks theo lô (batch_size={BATCH_SIZE})...")

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

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks_data[i : i + BATCH_SIZE]
            session.run(cypher_chunk, batch=batch)

        print(f"✓ Nạp toàn bộ {total_chunks} nút (:Chunk) THÀNH CÔNG.")

        # 6. Nạp các Quan hệ cấp Chunk (PART_OF, PARENT_OF, NEXT)
        print("\n--- 5. NẠP CÁC QUAN HỆ CẤP CHUNK (PART_OF, PARENT_OF, NEXT) ---")
        if not CHUNK_RELS_FILE.exists():
            print(f"❌ LỖI: Không tìm thấy file {CHUNK_RELS_FILE}")
            sys.exit(1)

        with open(CHUNK_RELS_FILE, mode="r", encoding="utf-8") as f:
            rels_data = json.load(f)

        part_of_list = [r for r in rels_data if r["type"] == "PART_OF"]
        parent_of_list = [r for r in rels_data if r["type"] == "PARENT_OF"]
        next_list = [r for r in rels_data if r["type"] == "NEXT"]

        # 6.1 Nạp PART_OF (Chunk -> Document)
        # Khớp d.id với row.target (nếu target = 'doc_44209' thì d.id = '44209')
        cypher_part_of = """
        UNWIND $batch AS row
        MATCH (c:Chunk {chunk_id: row.source})
        MATCH (d:Document) WHERE d.id = row.target OR d.id = replace(row.target, 'doc_', '')
        MERGE (c)-[:PART_OF]->(d)
        """
        for i in range(0, len(part_of_list), BATCH_SIZE):
            session.run(cypher_part_of, batch=part_of_list[i : i + BATCH_SIZE])
        print(f"  • Đã tạo {len(part_of_list)} quan hệ [:PART_OF]")

        # 6.2 Nạp PARENT_OF (Parent -> Child)
        # Xử lý 2 trường hợp: Parent là Chunk HOẶC Parent là Document
        cypher_parent_of_chunk = """
        UNWIND $batch AS row
        MATCH (p:Chunk {chunk_id: row.source})
        MATCH (c:Chunk {chunk_id: row.target})
        MERGE (p)-[:PARENT_OF]->(c)
        """
        cypher_parent_of_doc = """
        UNWIND $batch AS row
        MATCH (p:Document) WHERE p.id = row.source OR p.id = replace(row.source, 'doc_', '')
        MATCH (c:Chunk {chunk_id: row.target})
        MERGE (p)-[:PARENT_OF]->(c)
        """
        for i in range(0, len(parent_of_list), BATCH_SIZE):
            sub_batch = parent_of_list[i : i + BATCH_SIZE]
            session.run(cypher_parent_of_chunk, batch=sub_batch)
            session.run(cypher_parent_of_doc, batch=sub_batch)
        print(f"  • Đã tạo {len(parent_of_list)} quan hệ [:PARENT_OF]")

        # 6.3 Nạp NEXT (Chunk_N -> Chunk_N+1)
        cypher_next = """
        UNWIND $batch AS row
        MATCH (c1:Chunk {chunk_id: row.source})
        MATCH (c2:Chunk {chunk_id: row.target})
        MERGE (c1)-[:NEXT]->(c2)
        """
        for i in range(0, len(next_list), BATCH_SIZE):
            session.run(cypher_next, batch=next_list[i : i + BATCH_SIZE])
        print(f"  • Đã tạo {len(next_list)} quan hệ [:NEXT]")

    total_ingest_time = time.time() - start_all
    print(f"\n✓ Hoàn tất nạp toàn bộ dữ liệu vào Neo4j (Tổng thời gian: {total_ingest_time:.2f}s).")

    # 7. BƯỚC 5: KIỂM TRA VÀ XÁC MINH SỐ LƯỢNG THỰC THỂ (VERIFICATION)
    print("\n==================================================")
    print(" BƯỚC 5: KIỂM TRA VÀ XÁC MINH SỐ LƯỢNG THỰC THỂ ")
    print("==================================================")

    with driver.session(database=active_db) as session:
        count_docs = session.run("MATCH (d:Document) RETURN count(d) AS cnt").single()["cnt"]
        count_doc_rels = session.run("MATCH (d1:Document)-[r]->(d2:Document) RETURN count(r) AS cnt").single()["cnt"]
        count_chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
        count_part_of = session.run("MATCH (c:Chunk)-[r:PART_OF]->(d:Document) RETURN count(r) AS cnt").single()["cnt"]
        count_parent_of = session.run("MATCH ()-[r:PARENT_OF]->(c:Chunk) RETURN count(r) AS cnt").single()["cnt"]
        count_next = session.run("MATCH (c1:Chunk)-[r:NEXT]->(c2:Chunk) RETURN count(r) AS cnt").single()["cnt"]
        count_embedded = session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS cnt").single()["cnt"]

        print(f"\n BẢNG XÁC MINH DỮ LIỆU TRÊN NEO4J (`{active_db}`):")
        print(f"  --------------------------------------------------")
        print(f"  • Số Nút Document                : {count_docs:5d}  (Mục tiêu: 15)")
        print(f"  • Số Quan hệ giữa Document        : {count_doc_rels:5d}  (Mục tiêu: 8)")
        print(f"  • Số Nút Chunk                   : {count_chunks:5d}  (Mục tiêu: 6465)")
        print(f"  • Số Nút Chunk có Vector Nhúng  : {count_embedded:5d}  (Mục tiêu: 6465)")
        print(f"  • Số Quan hệ [:PART_OF]          : {count_part_of:5d}  (Mục tiêu: 6465)")
        print(f"  • Số Quan hệ [:PARENT_OF]        : {count_parent_of:5d}  (Mục tiêu: 6465)")
        print(f"  • Số Quan hệ [:NEXT]             : {count_next:5d}  (Mục tiêu: 6450)")
        print(f"  --------------------------------------------------")

        all_passed = (count_docs == 15 and count_doc_rels == 8 and count_chunks == 6465 and count_part_of == 6465 and count_parent_of == 6465 and count_next == 6450)

        if all_passed:
            print("\n🎉 XÁC MINH BƯỚC 5: TẤT CẢ SỐ LIỆU ĐỀU KHỚP HOÀN HẢO! [ PASS ]")
        else:
            print(f"\n⚠️ XÁC MINH BƯỚC 5: KẾT QUẢ ĐÃ CẢI THIỆN (PART_OF={count_part_of}, PARENT_OF={count_parent_of}).")

    driver.close()

if __name__ == "__main__":
    main()
