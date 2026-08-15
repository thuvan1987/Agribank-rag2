import os
import sys
import pandas as pd
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
except ImportError:
    logger.error("Vui lòng cài đặt thư viện neo4j: pip install neo4j")
    sys.exit(1)

def get_neo4j_driver():
    # Thử load .env từ thư mục hiện tại hoặc các thư mục cha
    load_dotenv()
    
    # Nếu chưa có, thử load từ thư mục buoi_13 (như cấu trúc khóa học)
    if not os.environ.get("NEO4J_URI"):
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', 'buoi_13', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info(f"Đã kết nối Neo4j thành công tại {uri}")
        return driver
    except Exception as e:
        logger.error(f"Không thể kết nối Neo4j. Chi tiết lỗi: {e}")
        logger.info("Vui lòng đảm bảo Neo4j Database đang chạy và thông tin trong .env là chính xác.")
        return None

def apply_schema(driver):
    logger.info("Áp dụng Schema Constraints...")
    with driver.session() as session:
        session.run("CREATE CONSTRAINT vanban_id_unique IF NOT EXISTS FOR (v:VanBan) REQUIRE v.id IS UNIQUE")
        session.run("CREATE CONSTRAINT dieukhoan_id_unique IF NOT EXISTS FOR (d:DieuKhoan) REQUIRE d.id IS UNIQUE")

def load_vanban(driver, metadata_path):
    logger.info("Đang nạp Node VanBan...")
    df = pd.read_csv(metadata_path)
    # Đảm bảo id là số nguyên hợp lệ
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype(int)
    df = df.fillna("")
    
    records = df.to_dict("records")
    
    query = """
    UNWIND $records AS row
    MERGE (v:VanBan {id: toInteger(row.id)})
    SET v.title = row.title,
        v.document_type = row.loai_van_ban,
        v.status = row.tinh_trang_hieu_luc,
        v.so_ky_hieu = row.so_ky_hieu,
        v.lab_session = 'buoi_14'
    """
    
    with driver.session() as session:
        result = session.run(query, records=records)
        summary = result.consume()
        logger.info(f"Đã xử lý VanBan: Tạo mới {summary.counters.nodes_created} nodes, thiết lập {summary.counters.properties_set} thuộc tính.")

def load_dieukhoan(driver, chunks_path):
    logger.info("Đang nạp Node DieuKhoan...")
    df = pd.read_csv(chunks_path)
    df = df.dropna(subset=['chunk_id'])
    df = df[df['chunk_id'].astype(str).str.strip() != ""]
    # Đảm bảo document_id là int
    df['document_id'] = pd.to_numeric(df['document_id'], errors='coerce')
    df['document_id'] = df['document_id'].fillna(0).astype(int)
    df = df.fillna("")
    
    records = df.to_dict("records")
    
    query = """
    UNWIND $records AS row
    MERGE (d:DieuKhoan {id: row.chunk_id})
    SET d.document_id = toInteger(row.document_id),
        d.text = row.text,
        d.article = row.article,
        d.chapter = row.chapter,
        d.lab_session = 'buoi_14'
    """
    
    with driver.session() as session:
        result = session.run(query, records=records)
        summary = result.consume()
        logger.info(f"Đã xử lý DieuKhoan: Tạo mới {summary.counters.nodes_created} nodes, thiết lập {summary.counters.properties_set} thuộc tính.")

def create_contains_relationships(driver):
    logger.info("Tạo quan hệ (:VanBan)-[:CONTAINS]->(:DieuKhoan)...")
    query = """
    MATCH (v:VanBan {lab_session: 'buoi_14'})
    MATCH (d:DieuKhoan {lab_session: 'buoi_14', document_id: v.id})
    MERGE (v)-[r:CONTAINS]->(d)
    SET r.lab_session = 'buoi_14'
    """
    with driver.session() as session:
        result = session.run(query)
        summary = result.consume()
        logger.info(f"Tạo được {summary.counters.relationships_created} quan hệ CONTAINS.")

def create_next_relationships(driver, chunks_path):
    logger.info("Tạo quan hệ (:DieuKhoan)-[:NEXT]->(:DieuKhoan)...")
    # Sử dụng Pandas để xác định thứ tự chunk an toàn
    df = pd.read_csv(chunks_path)
    
    # Đảm bảo các chunk của cùng văn bản được gom lại và theo đúng thứ tự file gốc
    relationships = []
    
    for doc_id, group in df.groupby("document_id", sort=False):
        chunk_ids = group["chunk_id"].tolist()
        for i in range(len(chunk_ids) - 1):
            relationships.append({
                "from_id": chunk_ids[i],
                "to_id": chunk_ids[i+1]
            })
            
    query = """
    UNWIND $rels AS rel
    MATCH (d1:DieuKhoan {id: rel.from_id, lab_session: 'buoi_14'})
    MATCH (d2:DieuKhoan {id: rel.to_id, lab_session: 'buoi_14'})
    MERGE (d1)-[r:NEXT]->(d2)
    SET r.lab_session = 'buoi_14'
    """
    
    with driver.session() as session:
        result = session.run(query, rels=relationships)
        summary = result.consume()
        logger.info(f"Tạo được {summary.counters.relationships_created} quan hệ NEXT.")

def load_custom_relationships(driver, relationships_path):
    logger.info("Đang nạp quan hệ từ relationships.csv...")
    if not os.path.exists(relationships_path):
        logger.warning(f"Không tìm thấy file {relationships_path}. Bỏ qua nạp quan hệ tùy chỉnh.")
        return
        
    df = pd.read_csv(relationships_path)
    df = df.fillna("")
    
    for _, row in df.iterrows():
        doc_id = row['doc_id']
        other_doc_id = row['other_doc_id']
        rel_text = row['relationship']
        rel_type = row['relationship_type'].strip().upper().replace(" ", "_").replace(",", "")
        
        if not rel_type:
            continue
            
        # Không được dùng parameter trực tiếp cho relationship TYPE trong Cypher, nên format string an toàn.
        # Chúng ta chỉ cho phép các kí tự [A-Z_] cho rel_type để tránh injection.
        sanitized_rel_type = "".join([c for c in rel_type if c.isalpha() or c == '_'])
        
        query = f"""
        MATCH (v1:VanBan {{id: toInteger($doc_id), lab_session: 'buoi_14'}})
        MATCH (v2:VanBan {{id: toInteger($other_doc_id), lab_session: 'buoi_14'}})
        MERGE (v1)-[r:{sanitized_rel_type}]->(v2)
        SET r.relationship = $rel_text,
            r.lab_session = 'buoi_14'
        """
        
        with driver.session() as session:
            session.run(query, doc_id=doc_id, other_doc_id=other_doc_id, rel_text=rel_text)
            
    logger.info("Đã nạp xong quan hệ tùy chỉnh từ relationships.csv.")

def generate_report(driver, output_path):
    logger.info("Khởi tạo báo cáo...")
    
    report_lines = [
        "# Báo cáo Xây dựng Mini Knowledge Graph (Buổi 14)",
        "",
        "## 1. Thống kê Node"
    ]
    
    with driver.session() as session:
        # Đếm Nodes
        res_nodes = session.run("MATCH (n) WHERE n.lab_session='buoi_14' RETURN labels(n)[0] AS label, count(n) AS c")
        for record in res_nodes:
            report_lines.append(f"- **{record['label']}**: {record['c']} nodes")
            
        report_lines.append("")
        report_lines.append("## 2. Thống kê Relationships")
        
        # Đếm Relationships
        res_rels = session.run("MATCH ()-[r]->() WHERE r.lab_session='buoi_14' RETURN type(r) AS type, count(r) AS c")
        for record in res_rels:
            report_lines.append(f"- **{record['type']}**: {record['c']} relationships")
            
        report_lines.append("")
        report_lines.append("## 3. Orphan Nodes (Không kết nối)")
        
        # Kiểm tra orphan
        res_orphans = session.run("MATCH (n {lab_session: 'buoi_14'}) WHERE NOT (n)-[]-() RETURN labels(n)[0] AS label, count(n) AS c")
        has_orphan = False
        for record in res_orphans:
            if record['c'] > 0:
                report_lines.append(f"- Phát hiện **{record['c']}** orphan nodes loại `{record['label']}`.")
                has_orphan = True
                
        if not has_orphan:
            report_lines.append("- Không phát hiện orphan node nào. Đồ thị được kết nối tốt.")
            
    # Ghi file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Đã ghi báo cáo tại: {output_path}")

def main():
    logger.info("=== BẮT ĐẦU XÂY DỰNG MINI KNOWLEDGE GRAPH ===")
    
    driver = get_neo4j_driver()
    if not driver:
        logger.error("Hủy quá trình do Neo4j không khả dụng.")
        return

    # Đường dẫn file
    base_dir = os.path.abspath(os.path.dirname(__file__) + '/..')
    metadata_csv = os.path.join(base_dir, '..', 'kb+hops', 'metadata.csv')
    relationships_csv = os.path.join(base_dir, '..', 'kb+hops', 'relationships.csv')
    chunks_csv = os.path.join(base_dir, 'data', 'processed', 'chunks_normalized.csv')
    report_file = os.path.join(base_dir, 'outputs', 'kg_build_report.md')

    if not all(os.path.exists(p) for p in [metadata_csv, chunks_csv]):
        logger.error("Thiếu file CSV nguồn. Vui lòng kiểm tra lại đường dẫn.")
        driver.close()
        return

    # Thực thi tuần tự
    apply_schema(driver)
    load_vanban(driver, metadata_csv)
    load_dieukhoan(driver, chunks_csv)
    create_contains_relationships(driver)
    create_next_relationships(driver, chunks_csv)
    load_custom_relationships(driver, relationships_csv)
    
    generate_report(driver, report_file)
    
    driver.close()
    logger.info("=== HOÀN TẤT XÂY DỰNG MINI KNOWLEDGE GRAPH ===")

if __name__ == "__main__":
    main()
