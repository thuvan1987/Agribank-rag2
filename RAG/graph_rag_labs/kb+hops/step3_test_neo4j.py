import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

def main():
    print("==================================================")
    print(" BẮT ĐẦU THỰC THI BƯỚC 3 - BUỔI 10: KIỂM TRA KẾT NỐI NEO4J ")
    print("==================================================")

    # 1. Đọc cấu hình từ file .env
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
        print(f"✓ Đã nạp cấu hình từ {ENV_FILE.name}")
    else:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy {ENV_FILE.name}, sử dụng cấu hình mặc định.")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    http_uri = os.getenv("NEO4J_HTTP_URI", "http://localhost:7474").strip()
    username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD", "password").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()

    # Che mật khẩu khi hiển thị cấu hình
    masked_password = "*" * len(password) if password else "(Rỗng)"

    print("\n--- THÔNG TIN CẤU HÌNH KẾT NỐI ---")
    print(f"  • Bolt URI    : {uri}")
    print(f"  • HTTP Browser: {http_uri}")
    print(f"  • Username    : {username}")
    print(f"  • Password    : {masked_password}")
    print(f"  • Database    : {database}")
    print("-----------------------------------\n")

    # 2. Thực hiện kết nối tới Neo4j Instance
    print(f"✓ Đang thử kết nối tới Neo4j qua giao thức Bolt ({uri})...")

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connectivity
        driver.verify_connectivity()
        print("✓ Xác thực kết nối mạng (verify_connectivity) THÀNH CÔNG!")

        # 3. Chạy truy vấn Cypher thử nghiệm
        print(f"✓ Đang thực thi truy vấn Cypher test trên database '{database}'...")
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 AS test_val, datetime() AS server_time")
            record = result.single()
            test_val = record["test_val"]
            server_time = record["server_time"]

        print(f"✓ Truy vấn Cypher thành công! Giá trị phản hồi: {test_val}, Thời gian server: {server_time}")
        print("\n==================================================")
        print(" TRẠNG THÁI KẾT NỐI NEO4J: [ PASS ]")
        print("==================================================\n")
        return True

    except exceptions.AuthError:
        print("\n❌ LỖI XÁC THỰC (AUTH ERROR): Sai Username hoặc Password!")
        print("👉 HƯỚNG DẪN BẮC CẦU:")
        print("   1. Mở Neo4j Desktop.")
        print("   2. Kiểm tra mật khẩu của Instance Neo4j.")
        print(f"   3. Cập nhật lại NEO4J_PASSWORD trong file:\n      {ENV_FILE}")
        print("\n==================================================")
        print(" TRẠNG THÁI KẾT NỐI NEO4J: [ FAIL ]")
        print("==================================================\n")
        return False

    except exceptions.ServiceUnavailable as e:
        print("\n❌ LỖI DỊCH VỤ KHÔNG KHẢ DỤNG (SERVICE UNAVAILABLE): Không kết nối được cổng Bolt 7687!")
        print("👉 HƯỚNG DẪN SỬA LỖI:")
        print("   1. Mở ứng dụng Neo4j Desktop.")
        print("   2. Đảm bảo Database Instance đã được nhấn 'START' (đang ở trạng thái Active / Running).")
        print("   3. Kiểm tra cổng kết nối Bolt (mặc định 7687).")
        print("\n==================================================")
        print(" TRẠNG THÁI KẾT NỐI NEO4J: [ FAIL ]")
        print("==================================================\n")
        return False

    except Exception as e:
        print(f"\n❌ LỖI KHÔNG XÁC ĐỊNH: {str(e)}")
        print("\n==================================================")
        print(" TRẠNG THÁI KẾT NỐI NEO4J: [ FAIL ]")
        print("==================================================\n")
        return False

    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    main()
