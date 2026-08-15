// ==============================================================================
// DEMO CYPHER QUERIES FOR WIKI RISK GRAPH (BUỔI 13)
// ==============================================================================

// Query A: Xem toàn bộ graph (Giới hạn 100 kết quả)
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100;

// Query B: Tìm kiểm soát giảm thiểu một rủi ro cụ thể (Ví dụ: RR-001)
MATCH (k:KiemSoat)-[r:MITIGATES]->(rui_ro:RuiRo {id: 'RR-001'})
RETURN k.id AS Control_ID, k.name AS Control_Name, r.evidence_quote AS Evidence, r.verification_status AS Status, rui_ro.name AS Risk_Name;

// Query C: Tìm sự kiện đã ghi nhận của một rủi ro cụ thể (Ví dụ: RR-001)
MATCH (rui_ro:RuiRo {id: 'RR-001'})-[r:OBSERVED_AS]->(s:SuKienRuiRo)
RETURN rui_ro.name AS Risk_Name, s.id AS Event_ID, s.description AS Event_Description, s.severity AS Severity, s.loss_amount_vnd AS Loss_VND;

// Query D: Tìm đường đi đầy đủ 3 bước (KiemSoat -> RuiRo -> SuKienRuiRo)
MATCH path = (k:KiemSoat)-[:MITIGATES]->(r:RuiRo)-[:OBSERVED_AS]->(s:SuKienRuiRo)
RETURN path
LIMIT 50;

// Query E: Tìm hồ sơ rủi ro chưa có bất kỳ biện pháp kiểm soát nào (MITIGATES)
MATCH (r:RuiRo)
WHERE NOT ( (:KiemSoat)-[:MITIGATES]->(r) )
RETURN r.id AS Unprotected_Risk_ID, r.name AS Risk_Name, r.category AS Category, r.inherent_level AS Inherent_Level;

// Query F: Tìm tất cả các quan hệ chưa được VERIFIED (Trạng thái PROPOSED)
MATCH (a)-[r]->(b)
WHERE r.verification_status <> 'VERIFIED'
RETURN type(r) AS Relationship_Type, a.id AS Source_ID, b.id AS Target_ID, r.evidence_quote AS Evidence, r.verification_status AS Status;
