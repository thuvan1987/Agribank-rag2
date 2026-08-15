// buoi_14/cypher/demo_queries.cypher

// 1. Đếm số lượng Node và Relationship theo lab_session = 'buoi_14'
MATCH (n) WHERE n.lab_session = 'buoi_14'
RETURN labels(n) AS Label, count(n) AS SoLuongNode;

MATCH ()-[r]->() WHERE r.lab_session = 'buoi_14'
RETURN type(r) AS LoaiQuanHe, count(r) AS SoLuongQuanHe;

// 2. Tìm tất cả các điều khoản (chunk) thuộc một văn bản cụ thể (ví dụ ID: 44209)
MATCH (v:VanBan {id: 44209, lab_session: 'buoi_14'})-[:CONTAINS]->(d:DieuKhoan)
RETURN v.title, d.article, d.text
ORDER BY d.id;

// 3. Truy vết các văn bản có quan hệ [:CAN_CU] (hoặc bất kỳ quan hệ linh động nào từ CSV)
MATCH (v1:VanBan {lab_session: 'buoi_14'})-[r:CAN_CU]->(v2:VanBan {lab_session: 'buoi_14'})
RETURN v1.id, v1.title, type(r) AS QuanHe, v2.id, v2.title;

// 4. Lấy ngữ cảnh: Điều khoản trước và sau của một điều khoản cụ thể (ví dụ ID: '44209_chunk_5')
MATCH (prev:DieuKhoan)-[:NEXT]->(d:DieuKhoan {id: '44209_chunk_5'})-[:NEXT]->(next:DieuKhoan)
RETURN prev.article AS Truoc, d.article AS HienTai, next.article AS Sau;

// 5. Kiểm tra orphan nodes (Node không có kết nối nào) trong buoi_14
MATCH (n {lab_session: 'buoi_14'})
WHERE NOT (n)-[]-()
RETURN labels(n) AS Label, n.id AS ID, n.title AS Title;
