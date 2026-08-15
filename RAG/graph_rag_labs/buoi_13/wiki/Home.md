---
id: HOME
type: Dashboard
verification_status: VERIFIED
data_origin: system
---

# 🏠 Wiki Risk Graph - Cổng thông tin Quản trị Rủi ro

Được khởi tạo từ dữ liệu chuẩn hóa [entities.csv](../outputs/entities.csv) và [relations.csv](../outputs/relations.csv).

## 📊 Thống kê Hệ thống Đồ thị
- **Tổng số Nodes (Thực thể)**: 34
  - 🛡️ **Biện pháp Kiểm soát (`KiemSoat`)**: 10
  - ⚠️ **Hồ sơ Rủi ro (`RuiRo`)**: 12
  - 🚨 **Sự kiện Rủi ro (`SuKienRuiRo`)**: 12
- **Tổng số Edges (Mối quan hệ)**: 22
  - `MITIGATES` (`KiemSoat` -> `RuiRo`): 10
  - `OBSERVED_AS` (`RuiRo` -> `SuKienRuiRo`): 12
- **Tổng số Wikilinks liên kết nội bộ**: 44

---

## 🗂️ Danh mục Trang Chi tiết

### 1. 🛡️ Danh sách Biện pháp Kiểm soát (`KiemSoat`)
- [[Đối soát tự động giao dịch và sổ cái|KS-001 - Đối soát tự động giao dịch và sổ cái]]
- [[Kiểm tra hạn mức phê duyệt trên hệ thống|KS-002 - Kiểm tra hạn mức phê duyệt trên hệ thống]]
- [[Checklist điều kiện giải ngân bắt buộc|KS-003 - Checklist điều kiện giải ngân bắt buộc]]
- [[Rà soát quyền truy cập định kỳ|KS-004 - Rà soát quyền truy cập định kỳ]]
- [[Kiểm thử khả năng chịu tải và chuyển đổi dự phòng|KS-005 - Kiểm thử khả năng chịu tải và chuyển đổi dự phòng]]
- [[Xác thực hai kênh với lệnh chuyển tiền ngoại lệ|KS-006 - Xác thực hai kênh với lệnh chuyển tiền ngoại lệ]]
- [[Theo dõi SLA xử lý cảnh báo AML|KS-007 - Theo dõi SLA xử lý cảnh báo AML]]
- [[Rà soát độc lập định giá tài sản bảo đảm|KS-008 - Rà soát độc lập định giá tài sản bảo đảm]]
- [[Hiệu chỉnh luật phát hiện giao dịch gian lận|KS-009 - Hiệu chỉnh luật phát hiện giao dịch gian lận]]
- [[Đối chiếu dữ liệu nguồn trước khi phát hành báo cáo|KS-010 - Đối chiếu dữ liệu nguồn trước khi phát hành báo cáo]]

### 2. ⚠️ Danh sách Hồ sơ Rủi ro (`RuiRo`)
- [[Giao dịch chuyển tiền bị hạch toán sai|RR-001 - Giao dịch chuyển tiền bị hạch toán sai]]
- [[Phê duyệt tín dụng vượt thẩm quyền|RR-002 - Phê duyệt tín dụng vượt thẩm quyền]]
- [[Giải ngân thiếu hồ sơ bảo đảm|RR-003 - Giải ngân thiếu hồ sơ bảo đảm]]
- [[Lộ thông tin khách hàng|RR-004 - Lộ thông tin khách hàng]]
- [[Gián đoạn dịch vụ ngân hàng số|RR-005 - Gián đoạn dịch vụ ngân hàng số]]
- [[Gian lận giả mạo yêu cầu chuyển tiền|RR-006 - Gian lận giả mạo yêu cầu chuyển tiền]]
- [[Chậm báo cáo giao dịch đáng ngờ|RR-007 - Chậm báo cáo giao dịch đáng ngờ]]
- [[Định giá tài sản bảo đảm không chính xác|RR-008 - Định giá tài sản bảo đảm không chính xác]]
- [[Không phát hiện giao dịch bất thường|RR-009 - Không phát hiện giao dịch bất thường]]
- [[Sai lệch số liệu báo cáo quản trị|RR-010 - Sai lệch số liệu báo cáo quản trị]]
- [[Nhà cung cấp công nghệ không đáp ứng cam kết|RR-011 - Nhà cung cấp công nghệ không đáp ứng cam kết]]
- [[Xung đột lợi ích trong mua sắm|RR-012 - Xung đột lợi ích trong mua sắm]]

### 3. 🚨 Danh sách Sự kiện Rủi ro (`SuKienRuiRo`)
- [[Sai lệch trạng thái giao dịch được phát hiện khi đ...|SK-001 - Sai lệch trạng thái giao dịch được phát hiện khi đ...]]
- [[Hồ sơ tín dụng được phê duyệt vượt hạn mức của ngư...|SK-002 - Hồ sơ tín dụng được phê duyệt vượt hạn mức của ngư...]]
- [[Giải ngân trước khi hoàn thiện chứng từ bảo đảm...|SK-003 - Giải ngân trước khi hoàn thiện chứng từ bảo đảm...]]
- [[Tài khoản có quyền truy cập dữ liệu vượt phạm vi c...|SK-004 - Tài khoản có quyền truy cập dữ liệu vượt phạm vi c...]]
- [[Dịch vụ ngân hàng số gián đoạn trong giờ cao điểm...|SK-005 - Dịch vụ ngân hàng số gián đoạn trong giờ cao điểm...]]
- [[Yêu cầu chuyển tiền giả mạo được xử lý trước khi b...|SK-006 - Yêu cầu chuyển tiền giả mạo được xử lý trước khi b...]]
- [[Báo cáo giao dịch đáng ngờ nộp quá hạn nội bộ...|SK-007 - Báo cáo giao dịch đáng ngờ nộp quá hạn nội bộ...]]
- [[Rà soát phát hiện giá trị tài sản bảo đảm đã hết h...|SK-008 - Rà soát phát hiện giá trị tài sản bảo đảm đã hết h...]]
- [[Giao dịch bất thường chỉ bị phát hiện sau khi khác...|SK-009 - Giao dịch bất thường chỉ bị phát hiện sau khi khác...]]
- [[Báo cáo quản trị sử dụng dữ liệu nguồn chưa đối ch...|SK-010 - Báo cáo quản trị sử dụng dữ liệu nguồn chưa đối ch...]]
- [[Nhà cung cấp chậm khôi phục dịch vụ so với SLA...|SK-011 - Nhà cung cấp chậm khôi phục dịch vụ so với SLA...]]
- [[Kiểm tra sau mua sắm phát hiện thiếu kê khai xung ...|SK-012 - Kiểm tra sau mua sắm phát hiện thiếu kê khai xung ...]]
