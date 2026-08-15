import os
import subprocess

queries = [
    "Thông tư số 01/2014/TT-NHNN Điều 5 quy định về đóng gói và giao nhận tài sản quý",
    "Quy trình và nguyên tắc tổ chức bảo quản tiền mặt kho quỹ trong ngân hàng",
    "Chi tiết quy định vận chuyển tiền mặt và giấy tờ có giá theo Thông tư 01/2014"
]

def main():
    print("=== ĐÁNH GIÁ 3 QUERIES VỚI RERANKING ===")
    
    with open("outputs/rerank_results_raw.txt", "w") as f:
        for i, q in enumerate(queries):
            print(f"Running query {i+1}: {q}")
            f.write(f"\n\n=========================================\n")
            f.write(f"QUERY {i+1}: {q}\n")
            f.write(f"=========================================\n")
            
            # Chạy script rerank.py
            cmd = ["python3", "scripts/rerank.py", "--query", q, "--candidate-k", "20", "--top-k", "5"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
                
    print("Done! Check outputs/rerank_results_raw.txt")

if __name__ == "__main__":
    main()
