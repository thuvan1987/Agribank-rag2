// ==============================================================================
// SCHEMA CYPHER FOR WIKI RISK GRAPH (BUỔI 13)
// ==============================================================================

// 1. Uniqueness Constraints (Khóa duy nhất trên 'id')
CREATE CONSTRAINT rui_ro_id_unique IF NOT EXISTS
FOR (r:RuiRo) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT kiem_soat_id_unique IF NOT EXISTS
FOR (k:KiemSoat) REQUIRE k.id IS UNIQUE;

CREATE CONSTRAINT su_kien_rui_ro_id_unique IF NOT EXISTS
FOR (s:SuKienRuiRo) REQUIRE s.id IS UNIQUE;

// 2. Indexes for fast lookup (Tối ưu truy vấn)
CREATE INDEX rui_ro_name_idx IF NOT EXISTS
FOR (r:RuiRo) ON (r.name);

CREATE INDEX kiem_soat_name_idx IF NOT EXISTS
FOR (k:KiemSoat) ON (k.name);

CREATE INDEX su_kien_severity_idx IF NOT EXISTS
FOR (s:SuKienRuiRo) ON (s.severity);
