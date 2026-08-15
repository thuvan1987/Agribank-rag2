import os
import sys
import argparse
import pandas as pd
import logging
from dotenv import load_dotenv

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Tắt log quá dài của neo4j driver
logging.getLogger("neo4j").setLevel(logging.WARNING)

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

def get_neo4j_driver():
    if not GraphDatabase:
        return None
    load_dotenv()
    if not os.environ.get("NEO4J_URI"):
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', 'buoi_13', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None

def retrieve(question: str, method: str, top_k: int, corpus_path: str):
    logger.info("⏳ Đang nạp dữ liệu...")
    try:
        df = pd.read_csv(corpus_path)
        chunks = df.to_dict('records')
    except Exception as e:
        logger.error(f"Lỗi đọc corpus: {e}")
        return []

    results = []
    
    if method == "bm25":
        retriever = BM25Retriever(chunks)
        results = retriever.search(question, top_k=top_k)
        for r in results:
            r["retrieval_method"] = "bm25"
            r["score"] = r.pop("score", 0.0)
            
    elif method == "dense":
        retriever = DenseRetriever(chunks)
        results = retriever.search(question, top_k=top_k)
        for r in results:
            r["retrieval_method"] = "dense"
            r["score"] = r.pop("score", 0.0)
            
    elif method == "hybrid":
        retriever = HybridRetriever(chunks=chunks)
        results = retriever.search(question=question, candidate_k=20, top_k=top_k)
        for r in results:
            r["retrieval_method"] = "hybrid"
            r["score"] = r.get("rrf_score", 0.0)
            
    elif method == "hybrid_rerank":
        hybrid_retriever = HybridRetriever(chunks=chunks)
        candidates = hybrid_retriever.search(question=question, candidate_k=20, top_k=20)
        reranker = Reranker()
        results = reranker.rerank(query=question, candidates=candidates, top_k=top_k)
        for r in results:
            r["retrieval_method"] = "hybrid_rerank"
            r["score"] = r.get("rerank_score", 0.0)
            
    else:
        logger.error(f"Method '{method}' không được hỗ trợ.")
        return []

    # Format output schema
    formatted_results = []
    for i, res in enumerate(results):
        formatted_results.append({
            "rank": i + 1,
            "chunk_id": res.get("chunk_id"),
            "document_id": res.get("document_id"),
            "text": res.get("text"),
            "score": round(float(res.get("score", 0.0)), 4),
            "citation": res.get("citation", ""),
            "retrieval_method": res.get("retrieval_method")
        })

    return formatted_results

def print_graph_hints(driver, results):
    print("\n" + "="*50)
    print("GRAPH HINTS")
    print("="*50)
    
    if not driver:
        print("Trạng thái Neo4j: NOT READY")
        print("Hãy khởi động Neo4j và cung cấp biến môi trường để xem Graph Hints.")
        return

    print("Trạng thái Neo4j: READY\n")
    
    # Gom nhóm document_id và chunk_id
    doc_ids = list(set([r["document_id"] for r in results if pd.notna(r.get("document_id"))]))
    chunk_ids = list(set([r["chunk_id"] for r in results if r.get("chunk_id")]))
    
    print(f"- Các văn bản liên quan (document_id): {doc_ids}")
    print(f"- Các đoạn trích (chunk_id): {chunk_ids}")
    
    print("\n- Mối quan hệ pháp lý trực tiếp (từ Neo4j):")
    with driver.session() as session:
        # Lấy quan hệ của Văn bản
        if doc_ids:
            query = """
            MATCH (v1:VanBan {lab_session: 'buoi_14'})-[r]->(v2:VanBan {lab_session: 'buoi_14'})
            WHERE v1.id IN $doc_ids OR v2.id IN $doc_ids
            RETURN v1.id AS v1_id, type(r) AS rel_type, v2.id AS v2_id, v2.title AS v2_title
            """
            try:
                res = session.run(query, doc_ids=[int(d) for d in doc_ids])
                rel_count = 0
                for record in res:
                    rel_count += 1
                    print(f"  * Văn bản {record['v1_id']} --[{record['rel_type']}]--> {record['v2_id']} ({record['v2_title'][:40]}...)")
                if rel_count == 0:
                    print("  * (Không có quan hệ văn bản nào)")
            except Exception as e:
                print(f"  * Lỗi query Neo4j: {e}")

        # Có thể query chunk :NEXT nếu muốn
        print("\n- Ngữ cảnh tuyến tính của Chunk (từ Neo4j):")
        if chunk_ids:
            query_next = """
            MATCH (d1:DieuKhoan {lab_session: 'buoi_14'})-[r:NEXT]->(d2:DieuKhoan {lab_session: 'buoi_14'})
            WHERE d1.id IN $chunk_ids
            RETURN d1.id AS from_chunk, d2.id AS to_chunk
            LIMIT 5
            """
            try:
                res = session.run(query_next, chunk_ids=chunk_ids)
                rel_count = 0
                for record in res:
                    rel_count += 1
                    print(f"  * Chunk {record['from_chunk']} --[NEXT]--> Chunk {record['to_chunk']}")
                if rel_count == 0:
                    print("  * (Không có chunk kế tiếp)")
            except Exception as e:
                print(f"  * Lỗi query Neo4j: {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified Retrieval Demo (Buổi 14)")
    parser.add_argument('--query', type=str, required=True, help="Câu truy vấn")
    parser.add_argument('--method', type=str, choices=['bm25', 'dense', 'hybrid', 'hybrid_rerank'], required=True, help="Phương pháp truy xuất")
    parser.add_argument('--top-k', type=int, default=5, help="Số lượng kết quả cuối cùng")
    parser.add_argument('--corpus', type=str, default="data/processed/chunks_normalized.csv")
    args = parser.parse_args()

    print("┌──────────────────────────────────────────────┐")
    print("│         RAG HYBRID SEARCH — BUỔI 14          │")
    print("├──────────────────────────────────────────────┤")
    print(f"│ App:           Unified Query Demo")
    print(f"│ Method:        {args.method.upper()}")
    print(f"│ Top-k:         {args.top_k}")
    print("└──────────────────────────────────────────────┘\n")

    print(f"QUERY: \"{args.query}\"\n")

    # 1. Retrieval
    results = retrieve(question=args.query, method=args.method, top_k=args.top_k, corpus_path=args.corpus)

    if not results:
        print("Không tìm thấy kết quả nào.")
        return

    # 2. In Kết quả
    for r in results:
        print(f"[{r['rank']}] Score: {r['score']} | Chunk: {r['chunk_id']}")
        print(f"Citation: {r['citation']}")
        snippet = r['text'].replace('\n', ' ')
        print(f"Text: {snippet[:150]}...\n")

    # 3. Graph Hints
    driver = get_neo4j_driver()
    print_graph_hints(driver, results)
    
    if driver:
        driver.close()

if __name__ == "__main__":
    main()
