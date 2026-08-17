import os
import pandas as pd
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

def get_neo4j_driver():
    """Tạo kết nối với Neo4j từ cấu hình trong file .env."""
    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    return GraphDatabase.driver(uri, auth=(user, password))

def merge_security_tags(tx, chunk_id, document_id, allowed_roles):
    """
    Sử dụng MERGE để cập nhật thuộc tính allowed_roles vào đồ thị.
    Chỉ cập nhật/tạo mới, không DETACH DELETE.
    """
    # Chuyển chuỗi JSON string thành Python list để neo4j lưu thành List of Strings
    roles_list = json.loads(allowed_roles)
    
    query = """
    // 1. Xử lý node VanBan
    MERGE (v:VanBan {document_id: $document_id})
    SET v.allowed_roles = $roles_list,
        v.lab_session = "buoi_15"
        
    // 2. Xử lý node DieuKhoan (chunk)
    MERGE (d:DieuKhoan {chunk_id: $chunk_id})
    SET d.allowed_roles = $roles_list,
        d.lab_session = "buoi_15",
        d.document_id = $document_id
        
    // 3. Liên kết DieuKhoan với VanBan
    MERGE (d)-[:THUOC_VAN_BAN]->(v)
    """
    tx.run(query, 
           document_id=str(document_id), 
           chunk_id=str(chunk_id), 
           roles_list=roles_list)

def verify_graph(tx):
    """Kiểm tra đồ thị sau khi cập nhật."""
    print("\n[BẮT ĐẦU KIỂM TRA ĐỒ THỊ NEO4J]")
    
    # 1. Đếm số node có chứa thuộc tính allowed_roles
    count_query = """
    MATCH (n) 
    WHERE n.allowed_roles IS NOT NULL AND n.lab_session = "buoi_15"
    RETURN labels(n) AS node_labels, count(n) AS total_nodes
    """
    result = tx.run(count_query)
    print("\n1. Thống kê số lượng node có phân quyền (lab_session='buoi_15'):")
    for record in result:
        print(f"   - Labels {record['node_labels']}: {record['total_nodes']} nodes")
        
    # 2. Truy vấn thử 1 node VanBan và các node DieuKhoan tương ứng
    sample_query = """
    MATCH (v:VanBan)-[:THUOC_VAN_BAN]-(d:DieuKhoan)
    WHERE v.allowed_roles IS NOT NULL AND d.allowed_roles IS NOT NULL 
          AND v.lab_session = "buoi_15"
    RETURN v.document_id AS doc_id, v.allowed_roles AS doc_roles,
           d.chunk_id AS chunk_id, d.allowed_roles AS chunk_roles
    LIMIT 3
    """
    sample_result = tx.run(sample_query)
    print("\n2. Lấy mẫu kết quả liên kết (VanBan <-> DieuKhoan):")
    for record in sample_result:
        print(f"   >>> VanBan ID: {record['doc_id']} | Quyền: {record['doc_roles']}")
        print(f"       + DieuKhoan ID: {record['chunk_id']} | Quyền: {record['chunk_roles']}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    csv_file = os.path.join(project_dir, "data", "processed", "chunks_secure.csv")
    
    if not os.path.exists(csv_file):
        print(f"Lỗi: Không tìm thấy {csv_file}")
        return
        
    print(f"Đọc dữ liệu từ {csv_file}...")
    df = pd.read_csv(csv_file)
    
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            print("Đang chạy lệnh MERGE để nạp/cập nhật dữ liệu vào Neo4j...")
            total = len(df)
            for i, row in df.iterrows():
                chunk_id = row['chunk_id']
                document_id = row['document_id']
                allowed_roles = row['allowed_roles']
                
                session.execute_write(merge_security_tags, chunk_id, document_id, allowed_roles)
                
                if (i + 1) % 150 == 0:
                    print(f"   -> Đã xử lý {i + 1}/{total} chunks.")
                    
            print(f"   -> Đã xử lý xong {total}/{total} chunks. Hoàn tất!")
            
            # Chạy kiểm tra đồ thị
            session.execute_read(verify_graph)
    finally:
        driver.close()

if __name__ == "__main__":
    main()
