import re
from typing import List

def tokenize_legal_text(text: str) -> List[str]:
    """
    Hàm tokenize tối ưu cho văn bản pháp lý Việt Nam dùng cho BM25:
    - Giữ nguyên mã số hiệu văn bản (vd: 01/2014/TT-NHNN, 39/2016/TT-NHNN).
    - Giữ nguyên cụm số điều khoản (vd: điều 1, điều 5, chương 2).
    - Tách từ tiếng Việt và loại bỏ ký tự rác.
    """
    if not text:
        return []

    # 1. Lowercase text
    text_lower = text.lower().strip()

    # 2. Extract legal codes and article terms before general tokenization
    tokens = []

    # Pattern nhận diện mã văn bản (vd: 01/2014/tt-nhnn)
    doc_code_pattern = r"\b\d+/\d+/[a-z0-9\-]+(?:-[a-z0-9]+)*\b"
    doc_codes = re.findall(doc_code_pattern, text_lower)
    tokens.extend(doc_codes)

    # Pattern nhận diện số điều khoản (vd: điều 1, điều 25)
    article_pattern = r"\bđiều\s+\d+\b"
    articles = re.findall(article_pattern, text_lower)
    tokens.extend([art.replace(" ", "_") for art in articles])

    # Pattern nhận diện số chương (vd: chương 1, chương i)
    chapter_pattern = r"\bchương\s+[ivxlcdm0-9]+\b"
    chapters = re.findall(chapter_pattern, text_lower)
    tokens.extend([ch.replace(" ", "_") for ch in chapters])

    # 3. Chuẩn hóa khoảng trắng và loại bỏ các ký tự đặc biệt không phải từ/số/dấu gạch
    clean_str = re.sub(r"[^\w\s/]", " ", text_lower)
    words = clean_str.split()

    # Thêm các từ thông thường
    tokens.extend(words)

    return tokens
