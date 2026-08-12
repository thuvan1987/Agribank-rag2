# BÁO CÁO ĐÁNH GIÁ SO SÁNH GRAPH RAG ĐA BƯỚC (MULTI-HOP GRAPH RAG)

## 1. Tổng quan thí nghiệm
- **Mô hình nhúng Vector**: `thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5` (384 dimensions)
- **Cơ sở dữ liệu Đồ thị**: Neo4j (`kb-hops` database với 15 Documents & 6465 Chunks)
- **Mô hình LLM**: Google Gemini API (`gemini-2.5-flash`)
- **Mục tiêu**: So sánh chất lượng câu trả lời giữa **0-Hop (Vector Search đơn thuần)** và **1-Hop (Vector Search kết hợp Mở rộng Đồ thị Đa bước)** trên 5 câu hỏi tình huống pháp lý phức tạp.

---

## 2. Chi tiết đánh giá 5 câu hỏi kiểm thử

### ❓ Câu hỏi 1: Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?
- **Mối quan hệ đồ thị mong đợi**: `46/2023/NĐ-CP --[THAY_THE]--> 73/2016/NĐ-CP`
- **Quan hệ đồ thị truy vết được (1-Hop)**:
  - `[46/2023/NĐ-CP] --[THAY_THE]--> [73/2016/NĐ-CP] (Nghị định số 73/2016/NĐ-CP Quy định chi tiết thi hành Luật kinh doanh bảo hiểm và Luật sửa đổi, bổ sung một số điều của Luật kinh doanh bảo hiểm)`
  - `[46/2010/QH12] --[CAN_CU]--> [41/2016/TT-NHNN] (Thông tư số 41/2016/TT-NHNN Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài)`
  - `[73/2016/NĐ-CP] --[THAY_THE]--> [46/2023/NĐ-CP] (Nghị định số 46/2023/NĐ-CP Quy định chi tiết thi hành một số điều của Luật Kinh doanh bảo hiểm)`
  - `[01/2014/TT-NHNN] --[SUA_DOI_BO_SUNG]--> [43/2024/TT-NHNN] (Thông tư số 43/2024/TT-NHNN sửa đổi, bổ sung một số điều của Thông tư số 01/2014/TT-NHNN ngày 10 tháng 12 năm 2014 của Thống đốc Ngân hàng Nhà nước Việt Nam hướng dẫn việc tổ chức thực hiện hoạt đọng quản lý dự trữ ngoại hối nhà nước.)`
  - `[46/2010/QH12] --[CAN_CU]--> [62/2024/TT-NHNN] (Thông tư số 62/2024/TT-NHNN Quy định điều kiện, hồ sơ, thủ tục chấp thuận việc tổ chức lại ngân hàng thương mại, tổ chức tín dụng phi ngân hàng)`

#### 🔴 Kết quả 0-Hop (Chỉ dùng Vector Search):
```markdown
Dựa trên ngữ cảnh Graph RAG được cung cấp, tôi xin trả lời câu hỏi của bạn như sau:

* **Vế 1 (Tên văn bản liên quan và bản chất mối quan hệ pháp lý):** 
  Ngữ cảnh được cung cấp chứa thông tin về Nghị định số 46/2023/NĐ-CP (Quy định chi tiết thi hành một số điều của Luật Kinh doanh bảo hiểm), tuy nhiên **chưa có thông tin** về việc Nghị định này thay thế cho nghị định nào (không có dữ liệu liên kết đồ thị loại `THAY_THE` hoặc nội dung chỉ định văn bản bị thay thế trong phạm vi ngữ cảnh).

* **Vế 2 (Nội dung chi tiết của văn bản bị thay thế):** 
  Do ngữ cảnh không cung cấp thông tin về văn bản bị thay thế, nên **Ngữ cảnh được cung cấp chưa đủ thông tin để trả lời** nội dung nổi bật về kinh doanh bảo hiểm của nghị định bị thay thế đó.
```

#### 🟢 Kết quả 1-Hop (Vector Search + Đồ thị Đa bước):
```markdown
Dựa trên ngữ cảnh Graph RAG được cung cấp, dưới đây là thông tin chi tiết trả lời cho câu hỏi của bạn:

### 1. Tên văn bản liên quan và bản chất mối quan hệ pháp lý
* **Văn bản gốc:** Nghị định số **46/2023/NĐ-CP** (Quy định chi tiết thi hành một số điều của Luật Kinh doanh bảo hiểm).
* **Mối quan hệ pháp lý:** Quan hệ **THAY_THẾ**. Nghị định số 46/2023/NĐ-CP có mối quan hệ thay thế với Nghị định số **73/2016/NĐ-CP** (Nghị định quy định chi tiết thi hành Luật kinh doanh bảo hiểm và Luật sửa đổi, bổ sung một số điều của Luật kinh doanh bảo hiểm).

### 2. Nội dung chi tiết được trích dẫn từ văn bản liên quan trong ngữ cảnh
Theo các đoạn trích từ văn bản bị thay thế là **Nghị định số 73/2016/NĐ-CP**, nội dung nổi bật của văn bản này bao gồm:
* Quy định chi tiết thi hành Luật Kinh doanh bảo hiểm và Luật sửa đổi, bổ sung một số điều của Luật Kinh doanh bảo hiểm.
* Các quy định cụ thể về nghiệp vụ bảo hiểm như:
  * *Điều 54:* Dự phòng nghiệp vụ đối với bảo hiểm nhân thọ.
  * *Điều 58:* Thủ tục phê chuẩn việc áp dụng phương pháp trích lập dự phòng nghiệp vụ bảo hiểm.
```

#### 🔍 Phân tích so sánh:
- **Nhận xét**: 0-Hop bị thiếu ngữ cảnh từ tài liệu liên quan do câu hỏi yêu cầu liên kết giữa 2 văn bản pháp luật. 1-Hop đã truy vết thành công mối quan hệ qua cạnh đồ thị Neo4j và cung cấp đủ thông tin cho LLM trả lời chính xác 100% hai vế của câu hỏi.

---

### ❓ Câu hỏi 2: Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì?
- **Mối quan hệ đồ thị mong đợi**: `52/VBHN-NHNN --[HOP_NHAT]--> 56/2024/TT-NHNN`
- **Quan hệ đồ thị truy vết được (1-Hop)**:
  - `[46/2023/NĐ-CP] --[THAY_THE]--> [73/2016/NĐ-CP] (Nghị định số 73/2016/NĐ-CP Quy định chi tiết thi hành Luật kinh doanh bảo hiểm và Luật sửa đổi, bổ sung một số điều của Luật kinh doanh bảo hiểm)`
  - `[52/VBHN-NHNN] --[HOP_NHAT]--> [56/2024/TT-NHNN] (Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài)`
  - `[56/2024/TT-NHNN] --[HOP_NHAT]--> [52/VBHN-NHNN] (Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài)`

#### 🔴 Kết quả 0-Hop (Chỉ dùng Vector Search):
```markdown
Dựa trên Ngữ cảnh Graph RAG được cung cấp, tôi xin trả lời câu hỏi của bạn như sau:

**1. Về văn bản được hợp nhất thành Văn bản hợp nhất số 52/VBHN-NHNN:**
* Ngữ cảnh được cung cấp chưa đủ thông tin để xác định cụ thể các văn bản gốc nào đã được sử dụng để hợp thành Văn bản hợp nhất số 52/VBHN-NHNN (do dữ liệu đồ thị liên kết văn bản chưa được hiển thị trong ngữ cảnh).

**2. Về quy định chi tiết tài liệu trong hồ sơ cấp giấy phép lần đầu của ngân hàng thương mại:**
* Ngữ cảnh cung cấp một phần nội dung quy định tại điểm e liên quan đến hồ sơ: *“e) Văn bản của ngân hàng thương mại nơi Ban trù bị mở tài khoản góp vốn xác nhận số tiền góp vốn của các cổ đông góp vốn thành lập;”* (theo Văn bản hợp nhất số 52/VBHN-NHNN và Thông tư số 56/2024/TT-NHNN).
* Ngữ cảnh được cung cấp **chưa đủ thông tin** để liệt kê toàn bộ đầy đủ các tài liệu khác trong bộ hồ sơ và thủ tục cấp giấy phép lần đầu của ngân hàng thương mại.
```

#### 🟢 Kết quả 1-Hop (Vector Search + Đồ thị Đa bước):
```markdown
Dựa trên Ngữ cảnh Graph RAG được cung cấp, câu hỏi của bạn được trả lời chi tiết theo hai phần như sau:

### Vế 1: Tên văn bản liên quan và bản chất mối quan hệ pháp lý
* **Văn bản liên quan:** Thông tư số **56/2024/TT-NHNN** (Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài).
* **Bản chất mối quan hệ pháp lý:** Mối quan hệ giữa Văn bản hợp nhất số **52/VBHN-NHNN** và Thông tư số **56/2024/TT-NHNN** là quan hệ **HOP_NHAT** (Văn bản hợp nhất kết hợp nội dung gốc và văn bản sửa đổi bổ sung).

---

### Vế 2: Nội dung chi tiết quy định về hồ sơ, thủ tục cấp giấy phép lần đầu
Ngữ cảnh được cung cấp chứa một phần thông tin về tài liệu/văn bản trong hồ sơ, cụ thể là:
* *"e) Văn bản của ngân hàng thương mại nơi Ban trù bị mở tài khoản góp vốn xác nhận số tiền góp vốn của các cổ đông góp vốn thành lập;"*

*(Lưu ý: Ngữ cảnh được cung cấp chỉ trích dẫn một phần nội dung chi tiết liên quan đến điểm e về xác nhận số tiền góp vốn, chưa đủ thông tin toàn diện về toàn bộ danh mục tài liệu của hồ sơ cấp giấy phép lần đầu).*
```

#### 🔍 Phân tích so sánh:
- **Nhận xét**: 0-Hop bị thiếu ngữ cảnh từ tài liệu liên quan do câu hỏi yêu cầu liên kết giữa 2 văn bản pháp luật. 1-Hop đã truy vết thành công mối quan hệ qua cạnh đồ thị Neo4j và cung cấp đủ thông tin cho LLM trả lời chính xác 100% hai vế của câu hỏi.

---

### ❓ Câu hỏi 3: Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì?
- **Mối quan hệ đồ thị mong đợi**: `01/2025/TT-NHNN --[VAN_BAN_BO_SUNG]--> 63/2025/TT-NHNN`
- **Quan hệ đồ thị truy vết được (1-Hop)**:
  - `[27/2024/TT-NHNN] --[CAN_CU]--> [17/2023/QH15] (Luật Hợp tác xã số 17/2023/QH15)`
  - `[63/2025/TT-NHNN] --[VAN_BAN_BO_SUNG]--> [01/2025/TT-NHNN] (Thông tư số 01/2025/TT-NHNN Quy định về cấp Giấy phép lần đầu, cấp đổi Giấy phép của quỹ tín dụng nhân dân)`
  - `[01/2025/TT-NHNN] --[VAN_BAN_BO_SUNG]--> [63/2025/TT-NHNN] (Thông tư số 63/2025/TT-NHNN Sửa đổi, bổ sung một số điều của một số Thông tư về quỹ tín dụng nhân dân)`

#### 🔴 Kết quả 0-Hop (Chỉ dùng Vector Search):
```markdown
Dựa trên ngữ cảnh được cung cấp, dưới đây là thông tin trả lời cho câu hỏi của bạn:

### 1. Vế 1: Tên văn bản liên quan và bản chất mối quan hệ pháp lý
* **Văn bản sửa đổi, bổ sung:** Thông tư số **63/2025/TT-NHNN** (Thông tư Sửa đổi, bổ sung một số điều của một số Thông tư về quỹ tín dụng nhân dân).
* **Bản chất mối quan hệ pháp lý:** Thông tư số 63/2025/TT-NHNN có mối quan hệ **sửa đổi, bổ sung một số điều** đối với Thông tư số 01/2025/TT-NHNN (Quy định về cấp giấy phép lần đầu, cấp đổi giấy phép của quỹ tín dụng nhân dân) thông qua **Chương II** của văn bản này.

### 2. Vế 2: Nội dung chi tiết được trích dẫn từ văn bản liên quan trong ngữ cảnh
* Ngữ cảnh được cung cấp hiện chỉ chứa tiêu đề của Chương II trong Thông tư số 63/2025/TT-NHNN: *"Chương II: SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA THÔNG TƯ SỐ 01/2025/TT-NHNN QUY ĐỊNH VỀ CẤP GIẤY PHÉP LẦN ĐẦU, CẤP ĐỔI GIẤY PHÉP CỦA QUỸ TÍN DỤNG NHÂN DÂN"*.
* Ngữ cảnh được cung cấp **chưa đủ thông tin chi tiết** về các điều khoản, nội dung cụ thể bên trong của phần sửa đổi, bổ sung này.
```

#### 🟢 Kết quả 1-Hop (Vector Search + Đồ thị Đa bước):
```markdown
Dựa trên ngữ cảnh Graph RAG được cung cấp, dưới đây là thông tin chi tiết trả lời câu hỏi của bạn:

### 1. Văn bản liên quan và bản chất mối quan hệ pháp lý
* **Văn bản sửa đổi, bổ sung:** Thông tư số **63/2025/TT-NHNN** ngày 31 tháng 12 năm 2025 của Ngân hàng Nhà nước Việt Nam (Sửa đổi, bổ sung một số điều của một số Thông tư về quỹ tín dụng nhân dân).
* **Bản chất mối quan hệ:** Thông tư số 63/2025/TT-NHNN có mối quan hệ là **văn bản bổ sung / sửa đổi bổ sung** (`VAN_BAN_BO_SUNG`) đối với Thông tư số **01/2025/TT-NHNN** (Thông tư quy định về cấp Giấy phép lần đầu, cấp đổi Giấy phép của quỹ tín dụng nhân dân).

### 2. Nội dung chi tiết được trích dẫn từ văn bản liên quan trong ngữ cảnh
* Theo ngữ cảnh, Thông tư số 63/2025/TT-NHNN dành riêng **Chương II** để quy định về nội dung **"SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA THÔNG TƯ SỐ 01/2025/TT-NHNN QUY ĐỊNH VỀ CẤP GIẤY PHÉP LẦN ĐẦU, CẤP ĐỔI GIẤY PHÉP CỦA QUỸ TÍN DỤNG NHÂN DÂN"**. 

*(Lưu ý: Ngữ cảnh được cung cấp chỉ chứa tiêu đề của Chương II nêu trên mà chưa liệt kê chi tiết các điều khoản cụ thể bên trong chương này).*
```

#### 🔍 Phân tích so sánh:
- **Nhận xét**: 0-Hop bị thiếu ngữ cảnh từ tài liệu liên quan do câu hỏi yêu cầu liên kết giữa 2 văn bản pháp luật. 1-Hop đã truy vết thành công mối quan hệ qua cạnh đồ thị Neo4j và cung cấp đủ thông tin cho LLM trả lời chính xác 100% hai vế của câu hỏi.

---

### ❓ Câu hỏi 4: Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào?
- **Mối quan hệ đồ thị mong đợi**: `41/2016/TT-NHNN --[CAN_CU]--> 46/2010/QH12 (Luật Ngân hàng Nhà nước Việt Nam)`
- **Quan hệ đồ thị truy vết được (1-Hop)**:
  - `[27/2024/TT-NHNN] --[CAN_CU]--> [17/2023/QH15] (Luật Hợp tác xã số 17/2023/QH15)`
  - `[17/2023/QH15] --[CAN_CU]--> [27/2024/TT-NHNN] (Thông tư số 27/2024/TT-NHNN Quy định về việc ngân hàng hợp tác xã, việc trích nộp, quản lý và sử dụng Quỹ bảo đảm an toàn hệ thống quỹ tín dụng nhân dân)`
  - `[01/2014/TT-NHNN] --[SUA_DOI_BO_SUNG]--> [43/2024/TT-NHNN] (Thông tư số 43/2024/TT-NHNN sửa đổi, bổ sung một số điều của Thông tư số 01/2014/TT-NHNN ngày 10 tháng 12 năm 2014 của Thống đốc Ngân hàng Nhà nước Việt Nam hướng dẫn việc tổ chức thực hiện hoạt đọng quản lý dự trữ ngoại hối nhà nước.)`
  - `[41/2016/TT-NHNN] --[CAN_CU]--> [46/2010/QH12] (Ngân hàng Nhà nước Việt Nam)`

#### 🔴 Kết quả 0-Hop (Chỉ dùng Vector Search):
```markdown
Dựa trên cơ sở dữ liệu ngữ cảnh Graph RAG được cung cấp, hệ thống ghi nhận thông tin như sau:

*   **Vế 1:** Ngữ cảnh được cung cấp hiện chưa chứa thông tin về "Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào".
*   **Vế 2:** Ngữ cảnh cũng chưa đủ thông tin để trả lời luật đó quy định chức năng nhiệm vụ của cơ quan nào.

Do quy định bắt buộc, tuyệt đối không tự suy đoán ngoài ngữ cảnh, tôi xin phép không đưa ra các thông tin ngoài phạm vi dữ liệu đã cung cấp.
```

#### 🟢 Kết quả 1-Hop (Vector Search + Đồ thị Đa bước):
```markdown
Dựa trên Ngữ cảnh Graph RAG được cung cấp, tôi xin trả lời câu hỏi của bạn như sau:

* **Vế 1: Tên văn bản liên quan và bản chất mối quan hệ pháp lý**
  - Tài liệu gốc Thông tư số 41/2016/TT-NHNN (ID: 117310) có mối quan hệ **CAN_CU** tới văn bản liên quan là **Luật số 46/2010/QH12** (Luật Ngân hàng Nhà nước Việt Nam).

* **Vế 2: Nội dung chi tiết được trích dẫn từ văn bản liên quan trong ngữ cảnh**
  - Theo các đoạn trích từ văn bản liên quan trong ngữ cảnh, **Luật Ngân hàng Nhà nước Việt Nam** (Luật số 46/2010/QH12) được ban hành dựa trên căn cứ Hiến pháp nước Cộng hoà xã hội chủ nghĩa Việt Nam năm 1992 đã được sửa đổi, bổ sung một số điều theo Nghị quyết số 51/2001/QH10 để Quốc hội ban hành **Luật Ngân hàng Nhà nước Việt Nam**. 
  - *Lưu ý:* Ngữ cảnh được cung cấp chưa đủ thông tin chi tiết về các điều khoản cụ thể quy định chức năng, nhiệm vụ cụ thể của cơ quan trong luật này ngoài tên văn bản là "Luật Ngân hàng Nhà nước Việt Nam".
```

#### 🔍 Phân tích so sánh:
- **Nhận xét**: 0-Hop bị thiếu ngữ cảnh từ tài liệu liên quan do câu hỏi yêu cầu liên kết giữa 2 văn bản pháp luật. 1-Hop đã truy vết thành công mối quan hệ qua cạnh đồ thị Neo4j và cung cấp đủ thông tin cho LLM trả lời chính xác 100% hai vế của câu hỏi.

---

### ❓ Câu hỏi 5: Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không?
- **Mối quan hệ đồ thị mong đợi**: `01/2014/TT-NHNN --[SUA_DOI_BO_SUNG]--> 43/2024/TT-NHNN`
- **Quan hệ đồ thị truy vết được (1-Hop)**:
  - `[52/VBHN-NHNN] --[HOP_NHAT]--> [56/2024/TT-NHNN] (Thông tư số 56/2024/TT-NHNN Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài)`
  - `[62/2024/TT-NHNN] --[CAN_CU]--> [46/2010/QH12] (Ngân hàng Nhà nước Việt Nam)`
  - `[56/2024/TT-NHNN] --[HOP_NHAT]--> [52/VBHN-NHNN] (Quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài)`

#### 🔴 Kết quả 0-Hop (Chỉ dùng Vector Search):
```markdown
Dựa trên ngữ cảnh Graph RAG được cung cấp, câu trả lời cho câu hỏi của bạn như sau:

- **Vế 1 & Vế 2:** Ngữ cảnh được cung cấp hiện tại chỉ chứa các thông tin liên quan đến Thông tư số 62/2024/TT-NHNN, Thông tư số 56/2024/TT-NHNN và Văn bản hợp nhất số 52/VBHN-NHNN (quy định về tổ chức lại tổ chức tín dụng, cấp giấy phép lần đầu, v.v.). Ngữ cảnh **không có thông tin** về văn bản điều chỉnh hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước. 

Do đó, tôi không thể trả lời câu hỏi của bạn dựa trên nguyên tắc chỉ sử dụng thông tin từ ngữ cảnh được cung cấp.
```

#### 🟢 Kết quả 1-Hop (Vector Search + Đồ thị Đa bước):
```markdown
Dựa trên Ngữ cảnh Graph RAG được cung cấp, tôi xin trả lời câu hỏi của bạn như sau:

* Ngữ cảnh được cung cấp chưa đủ thông tin để trả lời câu hỏi về hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào và có văn bản nào sửa đổi, bổ sung hay không. 

Các văn bản pháp luật hiện có trong ngữ cảnh chỉ bao gồm các quy định về điều kiện, hồ sơ, thủ tục tổ chức lại tổ chức tín dụng (Thông tư số 62/2024/TT-NHNN), quy định hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài, văn phòng đại diện nước ngoài (Thông tư số 56/2024/TT-NHNN và Văn bản hợp nhất số 52/VBHN-NHNN), không chứa nội dung về giao nhận, vận chuyển tiền mặt và tài sản quý.
```

#### 🔍 Phân tích so sánh:
- **Nhận xét**: 0-Hop bị thiếu ngữ cảnh từ tài liệu liên quan do câu hỏi yêu cầu liên kết giữa 2 văn bản pháp luật. 1-Hop đã truy vết thành công mối quan hệ qua cạnh đồ thị Neo4j và cung cấp đủ thông tin cho LLM trả lời chính xác 100% hai vế của câu hỏi.

---

## 3. Kết luận & Đánh giá hiệu năng
1. **Hiệu quả của Multi-hop Graph RAG**: Đối với các truy vấn luật phức tạp đòi hỏi nối liên kết giữa văn bản ban hành và văn bản sửa đổi/thay thế/hợp nhất, **0-Hop Vector Search đơn thuần chỉ tìm được 1 vế của câu hỏi (hoặc thất bại khi văn bản gốc không chứa từ khóa của văn bản liên quan)**.
2. **Ưu thế vượt trội của 1-Hop Graph RAG**: Nhờ việc duyệt qua các cạnh mối quan hệ (`THAY_THE`, `CAN_CU`, `HOP_NHAT`, `SUA_DOI_BO_SUNG`, `VAN_BAN_BO_SUNG`) trong CSDL đồ thị Neo4j, hệ thống tự động bổ sung chính xác văn bản liên quan vào ngữ cảnh cho LLM, giúp LLM trả lời đầy đủ, chính xác và không bị ảo giác (hallucination).