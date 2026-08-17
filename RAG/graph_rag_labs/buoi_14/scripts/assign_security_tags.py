import pandas as pd
import json
import os

# Định nghĩa các từ khóa cho từng cấp độ phân quyền
admin_keywords = ["nhân sự", "lương thưởng", "tuyển dụng", "bổ nhiệm"]
staff_keywords = ["tín dụng", "rủi ro", "hạn mức", "phê duyệt", "duyệt vay"]

def assign_roles(text):
    """
    Hàm phân loại bảo mật dựa trên từ khóa trong nội dung văn bản.
    """
    if not isinstance(text, str):
        text = str(text)
    
    text_lower = text.lower()
    
    # Tài liệu nhạy cảm nhất (Chỉ Admin)
    if any(keyword in text_lower for keyword in admin_keywords):
        return json.dumps(["Admin"])
    
    # Tài liệu nghiệp vụ (Admin và Staff)
    if any(keyword in text_lower for keyword in staff_keywords):
        return json.dumps(["Admin", "Staff"])
    
    # Tài liệu quy định chung (Ai cũng đọc được)
    return json.dumps(["Admin", "Staff", "Guest"])

def validate_data(df):
    """
    Hàm kiểm tra tính hợp lệ của dữ liệu đã phân quyền.
    """
    print("\n[BẮT ĐẦU KIỂM TRA DỮ LIỆU]")
    
    # 1. Kiểm tra roles không bị trống
    print("\n1. Kiểm tra không có dòng nào bị trống (null) allowed_roles:")
    null_count = df['allowed_roles'].isnull().sum()
    empty_list_count = (df['allowed_roles'] == "[]").sum()
    print(f"   - Số dòng bị null: {null_count}")
    print(f"   - Số dòng là mảng rỗng '[]': {empty_list_count}")
    
    if null_count == 0 and empty_list_count == 0:
        print("   -> THÀNH CÔNG: Mọi dòng đều đã được phân quyền!")
    else:
        print("   -> LỖI: Phát hiện dòng chưa được phân quyền!")
        
    # 2. Thống kê số lượng theo nhóm phân quyền
    print("\n2. Thống kê số lượng chunk theo nhóm phân quyền:")
    role_counts = df['allowed_roles'].value_counts()
    for role, count in role_counts.items():
        print(f"   - {role}: {count} chunks")
        
    # 3. Hiển thị 3 dòng mẫu
    print("\n3. Mẫu dữ liệu cho 3 cấp độ bảo mật:")
    
    admin_sample = df[df['allowed_roles'] == '["Admin"]']
    if not admin_sample.empty:
        print("\n   >>> MẪU ADMIN (Bảo mật cao nhất):")
        print(f"   - Document ID: {admin_sample.iloc[0].get('document_id', 'N/A')}")
        print(f"   - Roles: {admin_sample.iloc[0]['allowed_roles']}")
        print(f"   - Text snippet: {admin_sample.iloc[0]['text'][:150]}...")
        
    staff_sample = df[df['allowed_roles'] == '["Admin", "Staff"]']
    if not staff_sample.empty:
        print("\n   >>> MẪU STAFF (Nghiệp vụ nội bộ):")
        print(f"   - Document ID: {staff_sample.iloc[0].get('document_id', 'N/A')}")
        print(f"   - Roles: {staff_sample.iloc[0]['allowed_roles']}")
        print(f"   - Text snippet: {staff_sample.iloc[0]['text'][:150]}...")
        
    guest_sample = df[df['allowed_roles'] == '["Admin", "Staff", "Guest"]']
    if not guest_sample.empty:
        print("\n   >>> MẪU GUEST (Quy định chung):")
        print(f"   - Document ID: {guest_sample.iloc[0].get('document_id', 'N/A')}")
        print(f"   - Roles: {guest_sample.iloc[0]['allowed_roles']}")
        print(f"   - Text snippet: {guest_sample.iloc[0]['text'][:150]}...")

def main():
    # Xây dựng đường dẫn tuyệt đối để tránh lỗi relative path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    input_file = os.path.join(project_dir, "data", "processed", "chunks_normalized.csv")
    output_file = os.path.join(project_dir, "data", "processed", "chunks_secure.csv")
    
    print(f"Đọc dữ liệu từ: {input_file}")
    if not os.path.exists(input_file):
        print("Lỗi: Không tìm thấy file đầu vào!")
        return
        
    df = pd.read_csv(input_file)
    print(f"Đã nạp {len(df)} dòng dữ liệu.")
    
    print("\nĐang gán thẻ phân loại bảo mật...")
    df['allowed_roles'] = df['text'].apply(assign_roles)
    
    print(f"\nGhi file kết quả ra: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Ghi file thành công!")
    
    # Thực thi các kiểm tra
    validate_data(df)

if __name__ == "__main__":
    main()
