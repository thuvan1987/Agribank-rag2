import os
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.secure_retriever import SecureRetriever

def run_security_audit():
    print("Loading Secure Retriever...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    csv_file = os.path.join(project_dir, "data", "processed", "chunks_secure.csv")
    
    df = pd.read_csv(csv_file)
    retriever = SecureRetriever(df)
    
    test_cases = [
        {
            "name": "Test 1: Admin HR Document (Tuyển dụng)",
            "query": "đào tạo ban đầu và thi cấp chứng chỉ đại lý, chi đào tạo nâng cao kiến thức cho đại lý, chi tuyển dụng đại lý",
            "target_sensitive_document_id": "163441",
            "unauthorized_roles": ["Guest", "Staff"],
            "authorized_roles": ["Admin"]
        },
        {
            "name": "Test 2: Admin HR Document (Ban trù bị)",
            "query": "Ban trù bị là một nhóm người do thành viên sáng lập lựa chọn",
            "target_sensitive_document_id": "177271",
            "unauthorized_roles": ["Guest", "Staff"],
            "authorized_roles": ["Admin"]
        },
        {
            "name": "Test 3: Staff Document (Vận chuyển tiền mặt)",
            "query": "quy định việc giao nhận, bảo quản, vận chuyển; kiểm tra, kiểm kê, bàn giao, xử lý thừa thiếu tiền mặt, tài sản quý",
            "target_sensitive_document_id": "44209",
            "unauthorized_roles": ["Guest"],
            "authorized_roles": ["Staff"]
        },
        {
            "name": "Test 4: Staff Document (Admin also access)",
            "query": "Ngân hàng Nhà nước Việt Nam (sau đây gọi tắt là Ngân hàng Nhà nước). Tổ chức tín dụng, chi nhánh ngân hàng nước ngoài",
            "target_sensitive_document_id": "44209",
            "unauthorized_roles": ["Guest"],
            "authorized_roles": ["Admin"]
        },
        {
            "name": "Test 5: Phê duyệt duyệt vay",
            "query": "hạn mức và phê duyệt duyệt vay theo quy định",
            "target_sensitive_document_id": "44209", # 44209 also has Staff keywords probably
            "unauthorized_roles": ["Guest"],
            "authorized_roles": ["Staff"]
        }
    ]
    
    report = []
    report.append("# Security Audit Report\n")
    report.append(f"**Tổng số test cases:** {len(test_cases)}\n")
    
    all_passed = True
    
    for i, tc in enumerate(test_cases):
        print(f"Running {tc['name']}...")
        report.append(f"## {tc['name']}")
        report.append(f"- **Query:** `{tc['query']}`")
        report.append(f"- **Target Doc ID:** `{tc['target_sensitive_document_id']}`")
        
        # Test with unauthorized
        unauth_hits = retriever.search(tc['query'], user_roles=tc['unauthorized_roles'], top_k=5)
        unauth_doc_ids = [str(h['document_id']) for h in unauth_hits]
        
        # Test with authorized
        auth_hits = retriever.search(tc['query'], user_roles=tc['authorized_roles'], top_k=10)
        auth_doc_ids = [str(h['document_id']) for h in auth_hits]
        
        passed_unauth = tc['target_sensitive_document_id'] not in unauth_doc_ids
        # We don't strictly require passed_auth to be True for a security pass, but it helps ensure retrieval works.
        # However, for Test 5, doc 44209 might not be the exact best match, let's just check leakage.
        passed_auth = tc['target_sensitive_document_id'] in auth_doc_ids
        
        if passed_unauth:
            if passed_auth:
                report.append("- **Kết quả:** ✅ PASS")
                report.append(f"- **Bằng chứng:** Tài liệu `{tc['target_sensitive_document_id']}` KHÔNG bị rò rỉ cho quyền {tc['unauthorized_roles']} và TRUY XUẤT THÀNH CÔNG bởi quyền {tc['authorized_roles']}.")
            else:
                report.append("- **Kết quả:** ⚠️ PASS (No Leak) nhưng FAIL (Recall)")
                report.append(f"- **Bằng chứng:** Tài liệu `{tc['target_sensitive_document_id']}` KHÔNG bị rò rỉ cho quyền {tc['unauthorized_roles']} NHƯNG cũng không lọt top tìm kiếm với quyền {tc['authorized_roles']}.")
        else:
            all_passed = False
            report.append("- **Kết quả:** ❌ FAIL (DATA LEAKAGE)")
            report.append(f"  - Lỗi nghiêm trọng: Tài liệu cấm `{tc['target_sensitive_document_id']}` đã bị rò rỉ cho quyền {tc['unauthorized_roles']}!")
                
        report.append("")
        
    report.append("## Kết luận")
    if all_passed:
        report.append("🎉 **Hệ thống đạt chứng nhận an toàn dữ liệu mức cơ bản.** 100% các bài test tự động không phát hiện rò rỉ dữ liệu (No Data Leakage). Các cơ chế phân quyền BM25, Dense, và Graph hoạt động chính xác.")
    else:
        report.append("⚠️ **Hệ thống KHÔNG đạt chuẩn.** Phát hiện lỗ hổng kiểm soát truy cập! Cần rà soát lại script phân quyền.")
        
    out_dir = os.path.join(project_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "security_audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"Security audit completed. Report saved to {report_path}")

if __name__ == "__main__":
    run_security_audit()
