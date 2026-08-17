# Security Audit Report

**Tổng số test cases:** 5

## Test 1: Admin HR Document (Tuyển dụng)
- **Query:** `đào tạo ban đầu và thi cấp chứng chỉ đại lý, chi đào tạo nâng cao kiến thức cho đại lý, chi tuyển dụng đại lý`
- **Target Doc ID:** `163441`
- **Kết quả:** ✅ PASS
- **Bằng chứng:** Tài liệu `163441` KHÔNG bị rò rỉ cho quyền ['Guest', 'Staff'] và TRUY XUẤT THÀNH CÔNG bởi quyền ['Admin'].

## Test 2: Admin HR Document (Ban trù bị)
- **Query:** `Ban trù bị là một nhóm người do thành viên sáng lập lựa chọn`
- **Target Doc ID:** `177271`
- **Kết quả:** ❌ FAIL (DATA LEAKAGE)
  - Lỗi nghiêm trọng: Tài liệu cấm `177271` đã bị rò rỉ cho quyền ['Guest', 'Staff']!

## Test 3: Staff Document (Vận chuyển tiền mặt)
- **Query:** `quy định việc giao nhận, bảo quản, vận chuyển; kiểm tra, kiểm kê, bàn giao, xử lý thừa thiếu tiền mặt, tài sản quý`
- **Target Doc ID:** `44209`
- **Kết quả:** ❌ FAIL (DATA LEAKAGE)
  - Lỗi nghiêm trọng: Tài liệu cấm `44209` đã bị rò rỉ cho quyền ['Guest']!

## Test 4: Staff Document (Admin also access)
- **Query:** `Ngân hàng Nhà nước Việt Nam (sau đây gọi tắt là Ngân hàng Nhà nước). Tổ chức tín dụng, chi nhánh ngân hàng nước ngoài`
- **Target Doc ID:** `44209`
- **Kết quả:** ✅ PASS
- **Bằng chứng:** Tài liệu `44209` KHÔNG bị rò rỉ cho quyền ['Guest'] và TRUY XUẤT THÀNH CÔNG bởi quyền ['Admin'].

## Test 5: Phê duyệt duyệt vay
- **Query:** `hạn mức và phê duyệt duyệt vay theo quy định`
- **Target Doc ID:** `44209`
- **Kết quả:** ⚠️ PASS (No Leak) nhưng FAIL (Recall)
- **Bằng chứng:** Tài liệu `44209` KHÔNG bị rò rỉ cho quyền ['Guest'] NHƯNG cũng không lọt top tìm kiếm với quyền ['Staff'].

## Kết luận
⚠️ **Hệ thống KHÔNG đạt chuẩn.** Phát hiện lỗ hổng kiểm soát truy cập! Cần rà soát lại script phân quyền.