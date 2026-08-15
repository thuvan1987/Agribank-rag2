// buoi_14/cypher/schema.cypher

// 1. UNIQUE constraint cho VanBan(id)
CREATE CONSTRAINT vanban_id_unique IF NOT EXISTS
FOR (v:VanBan) REQUIRE v.id IS UNIQUE;

// 2. UNIQUE constraint cho DieuKhoan(id)
CREATE CONSTRAINT dieukhoan_id_unique IF NOT EXISTS
FOR (d:DieuKhoan) REQUIRE d.id IS UNIQUE;

// (Neo4j sẽ tự động tạo index khi có UNIQUE constraint trên property đó, nên không cần CREATE INDEX cho `id` nữa)
