import os
import json
import sqlite3
import psycopg
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(DOTENV_PATH)

_client_cache = None
_db_backend_cache = None

def _get_genai_client():
    """Lấy cached Gemini client nếu có GEMINI_API_KEY."""
    global _client_cache
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    if _client_cache is None:
        try:
            _client_cache = genai.Client(api_key=api_key)
        except Exception:
            return None
    return _client_cache


# --- Database Storage Helpers ---

def _get_db_connection():
    """Tạo kết nối tới PostgreSQL (rag_db) hoặc fallback SQLite local disk (.db)."""
    global _db_backend_cache

    if _db_backend_cache == "sqlite":
        storage_dir = os.path.join(BASE_DIR, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        sqlite_path = os.path.join(storage_dir, "local_storage.db")
        return sqlite3.connect(sqlite_path), "sqlite"

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    db_name = os.getenv("POSTGRES_DB", "rag_db")

    try:
        conn = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=db_name,
            connect_timeout=1
        )
        _db_backend_cache = "postgres"
        return conn, "postgres"
    except Exception:
        _db_backend_cache = "sqlite"
        storage_dir = os.path.join(BASE_DIR, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        sqlite_path = os.path.join(storage_dir, "local_storage.db")
        return sqlite3.connect(sqlite_path), "sqlite"


def _init_db_table():
    """Khởi tạo bảng chunks trong database."""
    conn, backend = _get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_name TEXT,
                text TEXT
            );
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()


def _save_chunks_to_db(chunks):
    """Lưu danh sách chunks vào PostgreSQL hoặc SQLite."""
    _init_db_table()
    conn, backend = _get_db_connection()
    cur = conn.cursor()
    try:
        data_tuples = [(c["chunk_id"], c["doc_name"], c["text"]) for c in chunks]
        if backend == "postgres":
            cur.executemany("""
                INSERT INTO chunks (chunk_id, doc_name, text)
                VALUES (%s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_name = EXCLUDED.doc_name,
                    text = EXCLUDED.text;
            """, data_tuples)
        else:
            cur.executemany("""
                INSERT OR REPLACE INTO chunks (chunk_id, doc_name, text)
                VALUES (?, ?, ?);
            """, data_tuples)
        conn.commit()
    finally:
        cur.close()
        conn.close()


def _get_chunks_by_ids(chunk_ids):
    """Lấy danh sách text của chunks từ database dựa trên chunk_ids."""
    if not chunk_ids:
        return []
    conn, backend = _get_db_connection()
    cur = conn.cursor()
    results = []
    try:
        if backend == "postgres":
            cur.execute("SELECT chunk_id, doc_name, text FROM chunks WHERE chunk_id = ANY(%s);", (chunk_ids,))
            rows = cur.fetchall()
        else:
            placeholders = ",".join(["?"] * len(chunk_ids))
            cur.execute(f"SELECT chunk_id, doc_name, text FROM chunks WHERE chunk_id IN ({placeholders});", chunk_ids)
            rows = cur.fetchall()
        
        row_dict = {row[0]: {"chunk_id": row[0], "doc_name": row[1], "text": row[2]} for row in rows}
        for cid in chunk_ids:
            if cid in row_dict:
                results.append(row_dict[cid])
    finally:
        cur.close()
        conn.close()
    return results


def _get_db_counts():
    """Đếm số lượng documents và chunks trong database."""
    conn, backend = _get_db_connection()
    cur = conn.cursor()
    num_chunks = 0
    num_docs = 0
    try:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT doc_name) FROM chunks;")
        row = cur.fetchone()
        if row:
            num_chunks = row[0] or 0
            num_docs = row[1] or 0
    except Exception:
        pass
    finally:
        cur.close()
        conn.close()
    return num_docs, num_chunks, backend


# --- ChromaDB Helper ---

def _get_chroma_collection():
    """Khởi tạo ChromaDB client (Server hoặc Embedded Persistent Client tại storage/chroma/)."""
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
    except Exception:
        storage_dir = os.path.join(BASE_DIR, "storage", "chroma")
        os.makedirs(storage_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=storage_dir)

    return client.get_or_create_collection(name="buoi_06_rag")


# --- Gemini Helper ---

def _get_embeddings_batch(texts: list):
    """Tạo embeddings theo batch với Gemini (gemini-embedding-2, dimensionality=384)."""
    client = _get_genai_client()
    if not client:
        return [[0.0] * 384 for _ in texts]

    embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            res = client.models.embed_content(
                model="gemini-embedding-2",
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=384)
            )
            if hasattr(res, "embeddings") and res.embeddings:
                for item in res.embeddings:
                    embeddings.append(list(item.values))
            elif hasattr(res, "embedding") and res.embedding:
                embeddings.append(list(res.embedding.values))
            else:
                embeddings.extend([[0.0] * 384] * len(batch))
        except Exception:
            embeddings.extend([[0.0] * 384] * len(batch))

    if len(embeddings) < len(texts):
        embeddings.extend([[0.0] * 384] * (len(texts) - len(embeddings)))

    return embeddings


def _get_single_embedding(text: str):
    """Tạo 1 embedding cho câu hỏi."""
    embs = _get_embeddings_batch([text])
    return embs[0] if embs else [0.0] * 384


# --- Core Functions ---

def index():
    """
    1. Đọc JSON từ RAG/rag_foundation/buoi_05/output/chunks/
    2. Tạo embedding với Gemini (gemini-embedding-2, dimensionality=384)
    3. Lưu text vào PostgreSQL (hoặc SQLite local disk .db)
    4. Lưu embedding vào ChromaDB
    """
    chunks_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "buoi_05", "output", "chunks"))
    if not os.path.exists(chunks_dir):
        return {"status": "error", "message": f"Thư mục không tồn tại: {chunks_dir}"}

    all_chunks = []
    seen_ids = set()
    
    # Quét tất cả các file JSON trong thư mục chunks
    for root, _, files in os.walk(chunks_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            raw_id = item.get("chunk_id") or item.get("id") or f"{idx}"
                            c_id = f"{file}_{raw_id}"
                            if c_id in seen_ids:
                                c_id = f"{c_id}_{len(seen_ids)}"
                            seen_ids.add(c_id)

                            doc = item.get("source") or item.get("document_id") or file
                            txt = item.get("text") or item.get("content") or ""
                            if txt.strip():
                                all_chunks.append({"chunk_id": str(c_id), "doc_name": str(doc), "text": str(txt)})
                    elif isinstance(data, dict):
                        if "documents" in data:
                            for d in data.get("documents", []):
                                doc_name = d.get("document", file)
                                for p in d.get("pages", []):
                                    page_num = p.get("page", 0)
                                    txt = p.get("text", "")
                                    c_id = f"{file}_{doc_name}_p{page_num}"
                                    if c_id in seen_ids:
                                        c_id = f"{c_id}_{len(seen_ids)}"
                                    seen_ids.add(c_id)

                                    if txt.strip():
                                        all_chunks.append({"chunk_id": str(c_id), "doc_name": str(doc_name), "text": str(txt)})
                except Exception:
                    continue

    if not all_chunks:
        return {"status": "warning", "message": "Không tìm thấy dữ liệu chunk hợp lệ."}

    # 1. Lưu text vào PostgreSQL/SQLite
    _save_chunks_to_db(all_chunks)

    # 2. Tạo embeddings & Lưu vào ChromaDB
    texts = [item["text"] for item in all_chunks]
    embeddings = _get_embeddings_batch(texts)
    ids = [item["chunk_id"] for item in all_chunks]
    metadatas = [{"doc_name": item["doc_name"]} for item in all_chunks]

    collection = _get_chroma_collection()
    
    # Upsert vào ChromaDB theo từng batch (200 items/batch)
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    unique_docs = len(set(c["doc_name"] for c in all_chunks))
    return {
        "status": "success",
        "documents": unique_docs,
        "chunks": len(all_chunks)
    }


def ask(question: str, top_k: int = 3, k: int = None):
    """
    1. Embedding câu hỏi với Gemini (dimensionality=384)
    2. Tìm top-k gần nhất trong ChromaDB
    3. Lấy text tương ứng từ PostgreSQL (hoặc SQLite local disk .db)
    4. Gửi ngữ cảnh + câu hỏi cho Gemini (gemini-flash-lite-latest) sinh câu trả lời
    Ràng buộc: Nếu thiếu GEMINI_API_KEY: Vẫn truy xuất retrieval, không gọi LLM.
    """
    if k is not None:
        top_k = k

    collection = _get_chroma_collection()
    total_in_chroma = collection.count()

    if total_in_chroma == 0:
        return {
            "answer": "Chưa có dữ liệu index trong ChromaDB. Vui lòng thực hiện index() trước.",
            "sources": [],
            "context": ""
        }

    # 1. Embedding câu hỏi
    q_emb = _get_single_embedding(question)

    # 2. Tìm kiếm top-k trong ChromaDB
    n_results = min(top_k, total_in_chroma)
    query_res = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results
    )

    retrieved_ids = []
    if query_res and "ids" in query_res and query_res["ids"]:
        retrieved_ids = query_res["ids"][0]

    # 3. Lấy text từ Database (PostgreSQL / SQLite)
    chunks_data = _get_chunks_by_ids(retrieved_ids)

    context_blocks = []
    sources = []
    for item in chunks_data:
        doc = item.get("doc_name", "Unknown")
        txt = item.get("text", "")
        context_blocks.append(f"[Tài liệu: {doc}]\n{txt}")
        sources.append({"doc_name": doc, "chunk_id": item.get("chunk_id")})

    context_text = "\n\n".join(context_blocks)

    client = _get_genai_client()

    # Ràng buộc: Nếu thiếu GEMINI_API_KEY -> Trả về kết quả Retrieval, không gọi LLM
    if not client:
        return {
            "answer": f"[KẾT QUẢ RETRIEVAL - Thiếu GEMINI_API_KEY]\n\nNgữ cảnh trích xuất:\n{context_text}",
            "sources": sources,
            "context": context_text
        }

    # 4. Gửi ngữ cảnh + câu hỏi cho Gemini (gemini-flash-lite-latest)
    try:
        prompt = f"Dựa vào thông tin ngữ cảnh bên dưới để trả lời câu hỏi một cách ngắn gọn, chính xác:\n\n{context_text}\n\nCâu hỏi: {question}\nTrả lời:"
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt
        )
        answer_text = response.text
    except Exception as e:
        answer_text = f"Không thể sinh câu trả lời LLM: {str(e)}\n\n[Dữ liệu trích xuất]\n{context_text}"

    return {
        "answer": answer_text,
        "sources": sources,
        "context": context_text
    }


def status():
    """
    Trả về số lượng document và số lượng chunk.
    """
    collection = _get_chroma_collection()
    chroma_chunks = collection.count()
    
    num_docs, num_chunks, backend = _get_db_counts()
    
    return {
        "documents": num_docs,
        "chunks": max(num_chunks, chroma_chunks),
        "db_backend": backend,
        "chroma_chunks": chroma_chunks
    }
