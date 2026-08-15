# 📋 Báo cáo Kiểm thử Wiki Risk Graph (Wiki Validation Report)

*Thời điểm kiểm thử: Tự động khởi tạo bởi `scripts/validate_wiki.py`*

---

## 1. 📊 Tổng quan Hệ thống Wiki

| Chỉ số kiểm thử | Giá trị | Trạng thái |
|---|---|---|
| **Tổng số file Markdown** | 35 | ✅ PASS |
| **Tổng số Wikilink** | 78 | ✅ PASS |
| **Wikilink hỏng (Broken link)** | 0 | ✅ PASS |
| **Entity bị trùng ID** | 0 | ✅ PASS |
| **Trang có ID không khớp entities.csv** | 0 | ✅ PASS |
| **Relation có source/target không tồn tại** | 0 | ✅ PASS |
| **Rủi ro thiếu Kiểm soát (MITIGATES)** | 2 | ⚠️ DATA NOTICE |
| **Rủi ro thiếu Sự kiện (OBSERVED_AS)** | 0 | ✅ FULL COVERAGE |
| **Trang mồ côi (Orphan Pages)** | 0 | ✅ PASS |

---

## 2. 🔍 Chi tiết kết quả kiểm tra

### 2.1. Kiểm tra Wikilink hỏng (Broken Wikilinks)
✅ **Không phát hiện bất kỳ wikilink hỏng nào.** Tất cả wikilinks đều trỏ đúng tới file đích.

### 2.2. Kiểm tra Trùng lặp ID Entity
✅ **Không có ID entity nào bị trùng lặp.**

### 2.3. Kiểm tra Rủi ro thiếu Kiểm soát hoặc thiếu Sự kiện
- ⚠️ **Danh sách Rủi ro chưa có Kiểm soát (`MITIGATES`)**: ['RR-011', 'RR-012']
- ✅ **100% Rủi ro đều có tối thiểu 1 Sự kiện Rủi ro thực tế.**

### 2.4. Kiểm tra Trang mồ côi (Orphan Pages)
✅ **Không có trang mồ côi nào.** Tất cả các trang đều có liên kết hai chiều từ `Home.md` hoặc các trang thực thể liên quan.

---

## 3. 🎯 Kết luận Phân loại Lỗi (Lỗi Code vs Lỗi Dữ liệu)

- 🚀 **Lỗi Chương trình (Code Bugs)**: **0 lỗi**. Tất cả thuật toán tạo file, sinh wikilink và cấu trúc thư mục hoạt động chuẩn xác 100%.
- 📊 **Ghi chú Dữ liệu Seed (Data Coverage)**: **2 ghi chú**. Trong dữ liệu gốc `relationships_seed.csv`, rủi ro `['RR-011', 'RR-012']` chưa được gán biện pháp kiểm soát `MITIGATES`. Tuân thủ quy tắc không tự bịa thêm quan hệ.
