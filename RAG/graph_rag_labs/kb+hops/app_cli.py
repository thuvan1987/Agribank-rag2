import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
sys.path.insert(0, str(BASE_DIR))

from graph_rag_retriever import GraphRAGRetriever
from step6_llm_qa_eval import call_gemini_llm


def ask_graph_rag(question: str, top_k: int = 5, n_hops: int = 1):
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=True)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("❌ LỖI: Thiếu GEMINI_API_KEY trong file .env")
        return

    gemini_client = genai.Client(api_key=api_key)
    retriever = GraphRAGRetriever(env_path=ENV_FILE)

    try:
        print(f"\n🔎 Đang truy vấn Ngữ cảnh Graph RAG (top_k={top_k}, n_hops={n_hops})...")
        context_data = retriever.get_multi_hop_context(question, top_k=top_k, n_hops=n_hops)
        
        print("\n=================== NGỮ CẢNH TRUY VẤN ĐƯỢC ===================")
        print(context_data["formatted_context"])
        print("=============================================================\n")

        print("🤖 Đang gửi ngữ cảnh tới Gemini LLM...")
        answer = call_gemini_llm(gemini_client, question, context_data["formatted_context"])

        print("\n=================== CÂU TRẢ LỜI CỦA LLM ===================")
        print(answer)
        print("=============================================================\n")

    finally:
        retriever.close()


def main():
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        ask_graph_rag(q)
    else:
        print("=== CHƯƠNG TRÌNH HỎI ĐÁP GRAPH RAG TƯƠNG TÁC (BUỔI 11) ===")
        print("Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát):\n")
        while True:
            try:
                user_q = input("❓ Câu hỏi: ").strip()
                if not user_q or user_q.lower() in ["exit", "quit", "q"]:
                    print("Tạm biệt!")
                    break
                
                hops_str = input("⚙️  Số bước nhảy Đồ thị (n_hops: 0 hoặc 1 hoặc 2, Mặc định: 1): ").strip()
                n_hops = int(hops_str) if hops_str.isdigit() else 1

                ask_graph_rag(user_q, top_k=5, n_hops=n_hops)
            except KeyboardInterrupt:
                print("\nThoát chương trình.")
                break


if __name__ == "__main__":
    main()
