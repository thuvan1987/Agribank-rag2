#!/usr/bin/env python3
"""Kiểm tra môi trường OCR / RAG Buổi 5.

Chạy từ thư mục `src`:
    python check_ocr_env.py
"""

import importlib
import importlib.util
import os
import sys

REQUIRED_PACKAGES = [
    ("fitz", "PyMuPDF"),
    ("PIL", "Pillow"),
    ("llama_cloud", "llama_cloud"),
    ("pydantic", "pydantic"),
    ("streamlit", "streamlit"),
    ("dotenv", "python-dotenv"),
]


def check_module(import_name: str):
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return False, ""

    try:
        module = importlib.import_module(import_name)
    except Exception:
        return False, ""

    version = getattr(module, "__version__", "")
    if not version:
        try:
            from importlib.metadata import version as pkg_version
            version = pkg_version(import_name)
        except Exception:
            version = ""

    return True, version


def print_header(title: str):
    print(title)
    print("=" * len(title))


def main():
    print_header("Kiểm tra môi trường OCR Buổi 5")

    py_ver = sys.version_info
    print(f"Python version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 11):
        print("⚠️  Khuyến nghị: dùng Python 3.11+ để tương thích tốt hơn với các thư viện modern.")

    missing = []
    print("\nCác gói cần thiết:")
    for import_name, pkg_name in REQUIRED_PACKAGES:
        ok, version = check_module(import_name)
        if ok:
            label = f"{pkg_name}"
            if version:
                label += f" (version {version})"
            print(f"  ✅ {label}")
        else:
            print(f"  ❌ {pkg_name} chưa cài")
            missing.append(pkg_name)

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    print("\nKiểm tra file cấu hình môi trường:")
    if os.path.isfile(env_path):
        print(f"  ✅ Đã tìm thấy {env_path}")
    else:
        print(f"  ❌ Thiếu file .env tại {env_path}")
        print("    Tạo file .env với nội dung: LLAMA_CLOUD_API_KEY='KEY CỦA BẠN'")
        missing.append(".env")

    print("\nKết quả chung:")
    if not missing:
        print("  ✅ PASS: môi trường OCR Buổi 5 đã sẵn sàng.")
        print("  Bạn có thể tiếp tục với xử lý PDF và OCR trong buổi học.")
        return 0

    print("  ❌ FAIL: môi trường chưa đầy đủ.")
    print("  Vui lòng cài thêm các gói sau:")
    print(f"    python -m pip install {' '.join(REQUIRED_PACKAGES[i][1] for i in range(len(REQUIRED_PACKAGES)))}")
    print("  Hoặc cài riêng từng gói nếu cần:")
    for pkg in [pkg_name for _, pkg_name in REQUIRED_PACKAGES]:
        if pkg in missing:
            print(f"    python -m pip install {pkg}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
