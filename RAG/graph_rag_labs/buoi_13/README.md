# 🛡️ Wiki Risk Graph - Hướng dẫn Vận hành Project (Buổi 13)

Dự án xây dựng **Wiki Tri thức Quản trị Rủi ro (Wiki Risk Graph)** từ dữ liệu rủi ro mô phỏng ngân hàng & tài chính, hỗ trợ duyệt Obsidian Wiki Markdown và nạp đồ thị tri thức vào cơ sở dữ liệu Neo4j.

---

## 📁 Cấu trúc Thư mục Project

```text
buoi_13/
├── .env                        # Cấu hình môi trường (URI, Password Neo4j)
├── .env.example                # File cấu hình mẫu
├── buoi_13.md                  # Tài liệu bài học chi tiết
├── README.md                   # Hướng dẫn sử dụng và thứ tự chạy
│
├── data/                       # Dữ liệu gốc (Seed Data)
│   ├── risk_profiles_seed.csv  # Hồ sơ 12 rủi ro
│   ├── controls_seed.csv       # Hồ sơ 10 biện pháp kiểm soát
│   ├── risk_events_seed.csv    # Hồ sơ 12 sự kiện rủi ro
│   └── relationships_seed.csv  # 22 quan hệ (MITIGATES, OBSERVED_AS)
│
├── outputs/                    # Dữ liệu chuẩn hóa & Báo cáo kiểm thử
│   ├── entities.csv            # 34 Nodes đã chuẩn hóa
│   ├── relations.csv           # 22 Edges đã chuẩn hóa
│   └── wiki_validation_report.md # Báo cáo nghiệm thu 9 hạng mục
│
├── scripts/                    # Scripts xử lý dữ liệu và kiểm thử
│   ├── inspect_data.py         # Bước 1: Kiểm tra dữ liệu gốc seed
│   ├── build_entities.py       # Bước 2: Chuẩn hóa thành Node & Edge
│   ├── build_wiki.py           # Bước 3: Sinh Wiki Markdown và Wikilinks
│   ├── validate_wiki.py        # Bước 4: Kiểm thử toàn vẹn Wiki
│   └── load_neo4j.py           # Bước 6: Import vào Neo4j Database
│
├── cypher/                     # Script Cypher cho Neo4j
│   ├── schema.cypher           # Uniqueness Constraints & Indexes
│   └── demo_queries.cypher     # 6 câu lệnh Cypher demo tra cứu đồ thị
│
└── wiki/                       # Kho lưu trữ Obsidian Wiki Markdown
    ├── Home.md                 # Cổng thông tin Dashboard chính
    ├── risks/                  # 12 trang hồ sơ rủi ro
    ├── controls/               # 10 trang biện pháp kiểm soát
    └── events/                 # 12 trang sự kiện rủi ro thực tế
```

---

## 🚀 Thứ tự Lệnh chạy Project (Execution Pipeline)

Thực hiện lần lượt các bước sau trong terminal từ thư mục `buoi_13/`:

### **Bước 1: Kiểm tra và phân tích Dữ liệu gốc Seed**
```bash
python3 scripts/inspect_data.py
```
*Kết quả*: In báo cáo tổng quan về 4 file CSV gốc, cấu trúc cột, số bản ghi và tính toàn vẹn của các khóa tham chiếu.

---

### **Bước 2: Chuẩn hóa Dữ liệu thành Entities và Relations**
```bash
python3 scripts/build_entities.py
```
*Kết quả*: Sinh file `outputs/entities.csv` (34 nodes) và `outputs/relations.csv` (22 edges).

---

### **Bước 3: Tự động Sinh Wiki Markdown cho Obsidian**
```bash
python3 scripts/build_wiki.py
```
*Kết quả*: Sinh 35 trang markdown chuẩn hóa kèm Obsidian Wikilinks trong thư mục `wiki/` (Home.md + risks + controls + events).

---

### **Bước 4: Kiểm thử Tính toàn vẹn của hệ thống Wiki**
```bash
python3 scripts/validate_wiki.py
```
*Kết quả*: Kiểm tra 9 hạng mục (broken link, orphan page, trùng ID...) và xuất báo cáo tại `outputs/wiki_validation_report.md`.

---

### **Bước 5: Trực quan hóa bằng Obsidian (Thao tác thủ công)**
1. Mở phần mềm **Obsidian**.
2. Chọn **Open folder as vault** -> Chọn thư mục `wiki/`.
3. Mở trang `Home.md` hoặc chuyển sang chế độ **Graph View** (Ctrl/Cmd + G) để xem đồ thị tri thức mạng lưới rủi ro.

---

### **Bước 6: Import Knowledge Graph vào Neo4j Database**

1. Khởi động instance Neo4j Server (hoặc Neo4j Desktop 2.0).
2. Tạo file `.env` với thông tin đăng nhập:
   ```text
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=abcd1234
   NEO4J_DATABASE=neo4j
   ```
3. Chạy script import:
   ```bash
   python3 scripts/load_neo4j.py
   ```
4. Truy cập **Neo4j Browser** (`http://localhost:7474`) và thực thi các câu lệnh Cypher kiểm thử trong file [cypher/demo_queries.cypher](cypher/demo_queries.cypher).

---

## 🔍 Danh sách Cypher Demo Queries

Các truy vấn Cypher trong `cypher/demo_queries.cypher`:
- **Query A**: Xem toàn bộ đồ thị (`MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100`)
- **Query B**: Tìm kiểm soát giảm thiểu rủi ro cụ thể (`:KiemSoat -[:MITIGATES]-> :RuiRo`)
- **Query C**: Tìm sự kiện rủi ro thực tế (`:RuiRo -[:OBSERVED_AS]-> :SuKienRuiRo`)
- **Query D**: Tìm đường đi 3 bước đầy đủ (`KiemSoat -> RuiRo -> SuKienRuiRo`)
- **Query E**: Tìm hồ sơ rủi ro chưa có biện pháp kiểm soát nào
- **Query F**: Tìm tất cả các quan hệ chưa được xác minh (`verification_status <> 'VERIFIED'`)
