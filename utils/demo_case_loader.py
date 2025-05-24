# utils/demo_case_loader.py
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image # MIMEタイプ取得のため
from io import BytesIO

DEMO_CASES_DIR = Path(__file__).parent.parent / "demo_cases"

# 画像ファイルの一般的な拡張子
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')

def get_available_demo_cases() -> List[str]:
    """demo_cases ディレクトリ内の question_id (ディレクトリ名) のリストを返す。"""
    if not DEMO_CASES_DIR.is_dir():
        return []
    return sorted([d.name for d in DEMO_CASES_DIR.iterdir() if d.is_dir()])

def load_demo_case_data(question_id: str) -> Optional[Dict[str, Any]]:
    """
    指定された question_id のデモケースデータを読み込む。

    Returns:
        A dictionary containing:
        - "question_id": str
        - "question_text": Optional[str]
        - "style_text": Optional[str]
        - "understood_text": Optional[str]
        - "images": List[Dict[str, Any]]  # [{"filename": str, "path": Path, "bytes": bytes, "mime_type": str}]
        - "error": Optional[str] (if any error occurs)
    """
    case_dir = DEMO_CASES_DIR / question_id
    if not case_dir.is_dir():
        return {"error": f"Demo case directory not found: {case_dir}"}

    data: Dict[str, Any] = {
        "question_id": question_id,
        "question_text": None,
        "style_text": None,
        "understood_text": None,
        "images": [],
    }

    # テキストファイルの読み込み
    try:
        q_file = case_dir / f"{question_id}_question.txt"
        if q_file.exists():
            data["question_text"] = q_file.read_text(encoding="utf-8").strip()

        style_file = case_dir / f"{question_id}_style.txt"
        if style_file.exists():
            data["style_text"] = style_file.read_text(encoding="utf-8").strip()

        understood_file = case_dir / f"{question_id}_understood.txt"
        if understood_file.exists():
            data["understood_text"] = understood_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        return {"error": f"Error reading text files for case {question_id}: {e}"}

    # 画像ファイルの読み込み
    try:
        for item in case_dir.iterdir():
            if item.is_file() and item.name.lower().endswith(IMAGE_EXTENSIONS):
                # question_id を含まない元のファイル名で画像が保存されている場合も考慮
                # ここでは、拡張子で画像ファイルかどうかを判断する
                filename = item.name
                file_path = item
                try:
                    with open(file_path, "rb") as f_img:
                        img_bytes = f_img.read()
                    
                    # MIMEタイプを推測 (Pillowを使用)
                    pil_img_temp = Image.open(BytesIO(img_bytes))
                    mime_type = Image.MIME.get(pil_img_temp.format)
                    if not mime_type: # PillowでMIMEが取れない場合、拡張子から簡易的に
                        ext = filename.split('.')[-1].lower()
                        if ext == "jpg" or ext == "jpeg": mime_type = "image/jpeg"
                        elif ext == "png": mime_type = "image/png"
                        elif ext == "webp": mime_type = "image/webp"
                        else: mime_type = "application/octet-stream" # 不明な場合

                    data["images"].append({
                        "filename": filename,
                        "path": file_path,
                        "bytes": img_bytes,
                        "mime_type": mime_type
                    })
                except Exception as e_img:
                    print(f"Warning: Could not load image {filename} for case {question_id}: {e_img}")
                    # エラーがあっても他のファイルのロードは続ける
    except Exception as e:
        return {"error": f"Error listing or reading image files for case {question_id}: {e}"}
    
    # 画像リストをファイル名でソート (任意)
    data["images"] = sorted(data["images"], key=lambda x: x["filename"])

    return data

if __name__ == '__main__':
    print("Available demo cases:", get_available_demo_cases())
    test_case_id = "785926" # 存在するケースIDでテスト
    if test_case_id in get_available_demo_cases():
        case_data = load_demo_case_data(test_case_id)
        if case_data and "error" not in case_data:
            print(f"\nData for case '{test_case_id}':")
            print(f"  Question: {case_data.get('question_text')}")
            print(f"  Style: {case_data.get('style_text')}")
            print(f"  Understood: {case_data.get('understood_text')}")
            for img_info in case_data.get("images", []):
                print(f"  Image: {img_info['filename']} (MIME: {img_info['mime_type']}, Size: {len(img_info['bytes'])} bytes)")
        elif case_data:
            print(f"Error loading case '{test_case_id}': {case_data['error']}")
    else:
        print(f"Test case ID '{test_case_id}' not found in available cases.")