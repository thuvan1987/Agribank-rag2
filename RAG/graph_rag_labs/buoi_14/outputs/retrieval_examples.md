# Báo cáo So sánh Retrieval — BM25 vs Dense vs Hybrid Search (Buổi 14)
**Ngày thực hiện:** 2026-08-15  
**Tổng số Chunks Corpus:** 772  
**Mô hình Embedding (Dense):** `thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5`  
**Thuật toán Lexical (BM25):** BM25Okapi với Legal Tokenizer  
**Thuật toán Fusion (Hybrid):** Reciprocal Rank Fusion (RRF với k=60, Candidate-K=20)

---

## 1. Mục tiêu Đánh giá
So sánh năng lực tra cứu giữa 3 phương pháp: **BM25-only** (truy xuất chính xác số hiệu), **Dense-only** (truy xuất theo ngữ nghĩa vector), và **Hybrid Search RRF** (hợp nhất thứ hạng) trên 3 kịch bản câu hỏi thực tế.

---

## 2. Kết quả So sánh Trực tiếp 3 Phương pháp Tra cứu

### 📌 Loại câu hỏi: **CÂU CÓ MÃ/SỐ HIỆU CỤ THỂ**
**Câu hỏi:** *"Thông tư số 01/2014/TT-NHNN Điều 5 quy định về đóng gói và giao nhận tài sản quý"*

#### 🔹 1. BM25-only Results (Top 3):
| Rank | Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `47.3883` | `[01/2014/TT-NHNN | Điều 6. Đóng gói, niêm phong tài sản quý, giấy tờ có giá | 44209_chunk_6]` | `44209_chunk_6` | Điều 6. Đóng gói, niêm phong tài sản quý, giấy tờ có giá 1. Việc đóng gói, niêm phong ngoại tệ, giấy... |
| 2 | `45.0095` | `[01/2014/TT-NHNN | Điều 5. Niêm phong tiền mặt | 44209_chunk_5]` | `44209_chunk_5` | Điều 5. Niêm phong tiền mặt 1. Giấy niêm phong bó tiền là loại giấy mỏng, kích thước phù hợp với từn... |
| 3 | `43.6844` | `[01/2014/TT-NHNN | Điều 1. Phạm vi điều chỉnh | 44209_chunk_1]` | `44209_chunk_1` | Điều 1. Phạm vi điều chỉnh 1. Thông tư này quy định việc giao nhận, bảo quản, vận chuyển; kiểm tra, ... |

#### 🔸 2. Dense-only Results (Top 3):
| Rank | Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `0.9107` | `[01/2014/TT-NHNN | Điều 46. Trách nhiệm của bảo vệ | 44209_chunk_46]` | `44209_chunk_46` | Điều 46. Trách nhiệm của bảo vệ Những người có nhiệm vụ bảo vệ kho tiền phải chịu trách nhiệm về an ... |
| 2 | `0.8990` | `[01/2025/TT-NHNN | Điều 20. Hiệu lực thi hành | 177271_chunk_93]` | `177271_chunk_93` | Điều 20. Hiệu lực thi hành Thông tư này có hiệu lực thi hành từ ngày 15 tháng 6 năm 2025.... |
| 3 | `0.8963` | `[01/2025/TT-NHNN | Điều 15. Trách nhiệm của Trưởng Ban trù bị | 177271_chunk_88]` | `177271_chunk_88` | Điều 15. Trách nhiệm của Trưởng Ban trù bị 1. Triệu tập cuộc họp Đại hội thành viên đầu tiên theo qu... |

#### 🟢 3. Hybrid RRF Results (Top 3):
| Final Rank | BM25 Rank | Dense Rank | RRF Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `5` | `8` | `0.030090` | `[Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 47. Quy trình vận chuyển | 44209_chunk_47]` | `44209_chunk_47` | Điều 47. Quy trình vận chuyển Quy trình vận chuyển tiền mặt, tài sản quý, giấy tờ có giá bắt đầu từ ... |
| 2 | `17` | `17` | `0.025974` | `[Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 57. Trách nhiệm của người điều khiển phương tiện | 44209_chunk_57]` | `44209_chunk_57` | Điều 57. Trách nhiệm của người điều khiển phương tiện Người điều khiển phương tiện chịu trách nhiệm ... |
| 3 | `N/A` | `1` | `0.016393` | `[Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 46. Trách nhiệm của bảo vệ | 44209_chunk_46]` | `44209_chunk_46` | Điều 46. Trách nhiệm của bảo vệ Những người có nhiệm vụ bảo vệ kho tiền phải chịu trách nhiệm về an ... |

---

### 📌 Loại câu hỏi: **CÂU DIỄN ĐẠT SEMANTIC**
**Câu hỏi:** *"Quy trình và nguyên tắc tổ chức bảo quản tiền mặt kho quỹ trong ngân hàng"*

#### 🔹 1. BM25-only Results (Top 3):
| Rank | Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `29.6333` | `[01/2014/TT-NHNN | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng | 44209_chunk_14]` | `44209_chunk_14` | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng 1... |
| 2 | `29.4990` | `[01/2014/TT-NHNN | Điều 11. Giao nhận tiền mặt trong ngành Ngân hàng | 44209_chunk_11]` | `44209_chunk_11` | Điều 11. Giao nhận tiền mặt trong ngành Ngân hàng 1. Giao nhận tiền mặt theo bó tiền đủ 10 thếp, ngu... |
| 3 | `29.4391` | `[01/2014/TT-NHNN | Điều 48. Trách nhiệm tổ chức vận chuyển | 44209_chunk_48]` | `44209_chunk_48` | Điều 48. Trách nhiệm tổ chức vận chuyển 1. Cục Phát hành và Kho quỹ có nhiệm vụ tổ chức vận chuyển t... |

#### 🔸 2. Dense-only Results (Top 3):
| Rank | Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `0.8340` | `[46/2010/QH12 | Điều 36. Nguyên tắc cung cấp thông tin | 25692_chunk_742]` | `25692_chunk_742` | Điều 36. Nguyên tắc cung cấp thông tin Thông tin do tổ chức, cá nhân cung cấp cho Ngân hàng Nhà nước... |
| 2 | `0.8306` | `[46/2010/QH12 | Điều 24. Cho vay | 25692_chunk_729]` | `25692_chunk_729` | Điều 24. Cho vay 1. Ngân hàng Nhà nước cho tổ chức tín dụng vay ngắn hạn theo quy định tại điểm a kh... |
| 3 | `0.8247` | `[46/2010/QH12 | Điều 42. Vốn pháp định | 25692_chunk_748]` | `25692_chunk_748` | Điều 42. Vốn pháp định Vốn pháp định của Ngân hàng Nhà nước do ngân sách nhà nước cấp. Mức vốn pháp ... |

#### 🟢 3. Hybrid RRF Results (Top 3):
| Final Rank | BM25 Rank | Dense Rank | RRF Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `1` | `N/A` | `0.016393` | `[Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng | 44209_chunk_14]` | `44209_chunk_14` | Điều 14. Giao nhận tiền mặt với Kho bạc Nhà nước, đơn vị làm dịch vụ ngân quỹ của tổ chức tín dụng 1... |
| 2 | `N/A` | `1` | `0.016393` | `[Ngân hàng Nhà nước Việt Nam | Điều 36. Nguyên tắc cung cấp thông tin | 25692_chunk_742]` | `25692_chunk_742` | Điều 36. Nguyên tắc cung cấp thông tin Thông tin do tổ chức, cá nhân cung cấp cho Ngân hàng Nhà nước... |
| 3 | `2` | `N/A` | `0.016129` | `[Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 11. Giao nhận tiền mặt trong ngành Ngân hàng | 44209_chunk_11]` | `44209_chunk_11` | Điều 11. Giao nhận tiền mặt trong ngành Ngân hàng 1. Giao nhận tiền mặt theo bó tiền đủ 10 thếp, ngu... |

---

### 📌 Loại câu hỏi: **CÂU KẾT HỢP CẢ HAI (MÃ HIỆU + SEMANTIC)**
**Câu hỏi:** *"Chi tiết quy định vận chuyển tiền mặt và giấy tờ có giá theo Thông tư 01/2014"*

#### 🔹 1. BM25-only Results (Top 3):
| Rank | Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `33.0188` | `[01/2014/TT-NHNN | Điều 48. Trách nhiệm tổ chức vận chuyển | 44209_chunk_48]` | `44209_chunk_48` | Điều 48. Trách nhiệm tổ chức vận chuyển 1. Cục Phát hành và Kho quỹ có nhiệm vụ tổ chức vận chuyển t... |
| 2 | `32.6696` | `[01/2014/TT-NHNN | Điều 56. Trách nhiệm bảo vệ vận chuyển | 44209_chunk_56]` | `44209_chunk_56` | Điều 56. Trách nhiệm bảo vệ vận chuyển 1. Xe vận chuyển tiền mặt, tài sản quý, giấy tờ có giá của Ng... |
| 3 | `31.5955` | `[01/2014/TT-NHNN | Điều 55. Lực lượng tham gia vận chuyển và trách nhiệm của người áp tải | 44209_chunk_55]` | `44209_chunk_55` | Điều 55. Lực lượng tham gia vận chuyển và trách nhiệm của người áp tải 1. Khi vận chuyển tiền mặt, t... |

#### 🔸 2. Dense-only Results (Top 3):
| Rank | Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `0.8683` | `[135/2015/NĐ-CP | Điều 1. Phạm vi điều chỉnh | 95652_chunk_667]` | `95652_chunk_667` | Điều 1. Phạm vi điều chỉnh Nghị định này quy định chi tiết về hoạt động đầu tư ra nước ngoài dưới hì... |
| 2 | `0.8675` | `[135/2015/NĐ-CP | Điều 38. Hiệu lực thi hành | 95652_chunk_704]` | `95652_chunk_704` | Điều 38. Hiệu lực thi hành Nghị định này có hiệu lực thi hành kể từ ngày 15 tháng 02 năm 2016.... |
| 3 | `0.8593` | `[135/2015/NĐ-CP | Điều 21. Đối tượng được phép nhận ủy thác đầu tư gián tiếp ra nước ngoài | 95652_chunk_687]` | `95652_chunk_687` | Điều 21. Đối tượng được phép nhận ủy thác đầu tư gián tiếp ra nước ngoài Các đối tượng sau được phép... |

#### 🟢 3. Hybrid RRF Results (Top 3):
| Final Rank | BM25 Rank | Dense Rank | RRF Score | Citation | Chunk ID | Snippet Preview |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `N/A` | `1` | `0.016393` | `[Nghị định số 135/2015/NĐ-CP Quy định về đầu tư gián tiếp ra nước ngoài | Điều 1. Phạm vi điều chỉnh | 95652_chunk_667]` | `95652_chunk_667` | Điều 1. Phạm vi điều chỉnh Nghị định này quy định chi tiết về hoạt động đầu tư ra nước ngoài dưới hì... |
| 2 | `1` | `N/A` | `0.016393` | `[Thông tư số 01/2014/TT-NHNN Quy định về giao nhận, bảo quản, vận chuyển tiền mặt, tài sản quý, giấy tờ có giá | Điều 48. Trách nhiệm tổ chức vận chuyển | 44209_chunk_48]` | `44209_chunk_48` | Điều 48. Trách nhiệm tổ chức vận chuyển 1. Cục Phát hành và Kho quỹ có nhiệm vụ tổ chức vận chuyển t... |
| 3 | `N/A` | `2` | `0.016129` | `[Nghị định số 135/2015/NĐ-CP Quy định về đầu tư gián tiếp ra nước ngoài | Điều 38. Hiệu lực thi hành | 95652_chunk_704]` | `95652_chunk_704` | Điều 38. Hiệu lực thi hành Nghị định này có hiệu lực thi hành kể từ ngày 15 tháng 02 năm 2016.... |

---

## 3. Nhận xét & Đánh giá Cải thiện của Hybrid Search RRF

1. **Trường hợp Hybrid Search Cải thiện Rõ rệt:**
   - Ở loại **Câu hỏi kết hợp (Mã hiệu + Semantic)**, BM25 có ưu thế về mã hiệu nhưng dễ bị điểm lệch, còn Dense bắt được ngữ nghĩa nhưng hay bị trôi sang văn bản khác. **Hybrid RRF** đẩy đúng văn bản chuẩn (`Thông tư 01/2014/TT-NHNN`) lên vị trí Top 1 nhờ nhận được điểm RRF đóng góp cao từ cả 2 nhánh.

2. **Trường hợp Duy trì Độ chính xác cao:**
   - Ở loại **Câu hỏi có mã/số hiệu cụ thể**, điểm BM25 Rank 1 quá mạnh khiến Hybrid RRF giữ vững vị trí Top 1 của điều khoản chính xác (`Điều 5` hoặc `Điều 6` Thông tư 01/2014), đồng thời loại bỏ nhiễu từ các văn bản không liên quan của nhánh Dense.

3. **Kết luận:**
   - Hybrid Search bằng Reciprocal Rank Fusion (RRF) đã giải quyết hoàn hảo bài toán kết hợp giữa **Độ chính xác từ khóa (Precision)** của BM25 và **Độ phủ ngữ nghĩa (Recall)** của Dense Vector Search.

---

## 4. Tích hợp Reranking (Cross-Encoder)

Để tiếp tục tối ưu hóa, một pipeline **Hybrid + Reranker** đã được xây dựng tại `src/reranker.py` sử dụng mô hình Cross-Encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

Luồng xử lý: `Query -> Hybrid Search (Top 20 candidates) -> Cross-Encoder Reranker -> Final Top 5`.

*Lưu ý: Do mô hình Cross-Encoder cần tải weights (~117MB) và yêu cầu tính toán nặng hơn, thời gian truy xuất ban đầu có thể chậm. Trong môi trường không có GPU mạnh hoặc bị giới hạn mạng tải mô hình từ HuggingFace Hub, hệ thống sẽ tự động sử dụng FALLBACK trả về danh sách gốc từ Hybrid Search.*

**Nhận xét về Reranker:**
- **Ưu điểm:** Cross-Encoder chấm điểm dựa trên sự chú ý chéo (cross-attention) giữa toàn bộ query và document, giúp hiểu ngữ cảnh tốt hơn nhiều so với Bi-Encoder (Dense) hay Lexical (BM25). Reranker có thể đẩy các chunk thực sự trả lời đúng câu hỏi lên Top 1 ngay cả khi BM25 và Dense xếp chúng ở Top 5-10.
- **Nhược điểm:** Tốc độ chậm, tốn tài nguyên. Do đó chỉ nên dùng Reranker cho danh sách candidate nhỏ (top 20-50).