# utils/image_processing_parts/ocr_processors.py
from PIL import Image
from io import BytesIO  # trim_image_by_ocr_text_bounds で使う可能性のあるデバッグロード用
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import cv2  # デバッグ描画用

try:
    import pytesseract
    from pytesseract import Output as PytesseractOutput  # image_to_data を使うなら必要
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("[ocr_processors] Warning: pytesseract is not installed. OCR-dependent features will be limited.")

def get_text_orientation_with_tesseract(img_pil: Image.Image, lang: str = 'eng+jpn') -> int:
    """
    Pytesseractを使用して画像のテキストの向き（0, 90, 180, 270度）を推定する。
    最も多くの文字が検出された向きを返す。
    戻り値: 最適と思われる回転角度 (0, 90, 180, 270)。エラー時は0を返す。
    """
    if not PYTESSERACT_AVAILABLE:
        print("[ocr_processors] Pytesseract not available, cannot determine text orientation via OCR.")
        return 0

    best_angle = 0
    max_chars = -1

    print("[ocr_processors] Determining text orientation using Tesseract OCR (0, 90, 180, 270 degrees)...")
    for angle in [0, 90, 180, 270]:
        try:
            if angle == 0:
                img_rotated = img_pil.copy()
            else:
                img_rotated = img_pil.rotate(angle, expand=True)
            
            text = pytesseract.image_to_string(img_rotated, lang=lang, timeout=5) 
            num_chars = len(text.strip())
            print(f"[ocr_processors] Angle {angle}: Detected {num_chars} characters.")

            if num_chars > max_chars:
                max_chars = num_chars
                best_angle = angle
        except pytesseract.TesseractError as e: # type: ignore # PytesseractError が未定義と出る場合
            print(f"[ocr_processors] Tesseract OCR error at angle {angle}: {e}")
        except RuntimeError as e: 
            print(f"[ocr_processors] Tesseract runtime error (possibly timeout) at angle {angle}: {e}")
        except Exception as e:
            print(f"[ocr_processors] Unexpected error during OCR at angle {angle}: {e}")
            
    print(f"[ocr_processors] Best OCR angle determined: {best_angle} degrees with {max_chars} characters.")
    return best_angle

def trim_image_by_ocr_text_bounds(
    img_pil_input: Image.Image,
    padding: int = 0,  # デフォルトパディングを0に変更
    lang: str = 'eng+jpn',
    min_confidence: int = 30,
    min_box_height: int = 5,      # 有効とみなす最小のボックス高さ
    max_box_height_ratio: float = 0.3, # 画像高さに対する最大ボックス高さの割合
    min_box_width: int = 5,       # 有効とみなす最小のボックス幅
    max_box_width_ratio: float = 0.8,  # 画像幅に対する最大ボックス幅の割合
    min_aspect_ratio: float = 0.1, # 有効とみなす最小アスペクト比 (幅/高さ)
    max_aspect_ratio: float = 10.0,# 有効とみなす最大アスペクト比 (幅/高さ)
    tesseract_config: str = '--psm 6', # Tesseractの設定オプション
    return_debug_cv_image: bool = False
) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """
    Pytesseract OCRを使用してテキスト領域を検出し、それに基づいて画像をトリミングする。

    Args:
        img_pil_input (PIL.Image.Image): 入力画像（Pillow形式）
        padding (int): トリミング領域の外側に追加する余白（ピクセル単位）
        lang (str): OCRの言語設定（デフォルト: 'eng+jpn'）
        min_confidence (int): テキストボックスを有効とみなす最小信頼度
        min_box_height (int): 有効とみなす最小のボックス高さ
        max_box_height_ratio (float): 画像高さに対する最大ボックス高さの割合
        min_box_width (int): 有効とみなす最小のボックス幅
        max_box_width_ratio (float): 画像幅に対する最大ボックス幅の割合
        min_aspect_ratio (float): 有効とみなす最小アスペクト比 (幅/高さ)
        max_aspect_ratio (float): 有効とみなす最大アスペクト比 (幅/高さ)
        tesseract_config (str): Tesseractの追加設定
        return_debug_cv_image (bool): デバッグ用にOpenCV画像を返すかどうか

    Returns:
        Tuple[Optional[PIL.Image.Image], Optional[np.ndarray]]: (トリミング後のPillow画像, デバッグ用OpenCV画像 or None)
    """
    if not PYTESSERACT_AVAILABLE:
        print("[ocr_processors] Pytesseract not available for OCR-based trimming.")
        return None, None
    if not isinstance(img_pil_input, Image.Image):
        print(f"[ocr_processors] Error: Input must be a PIL Image for OCR trimming.")
        return None, None

    img_cv_for_debug_output: Optional[np.ndarray] = None
    img_pil_width, img_pil_height = img_pil_input.size

    try:
        print(f"[ocr_processors] Attempting OCR-based trimming (lang: {lang}, min_conf: {min_confidence}, tesseract_config: '{tesseract_config}')...")
        ocr_data = pytesseract.image_to_data(
            img_pil_input,
            lang=lang,
            config=tesseract_config,
            output_type=PytesseractOutput.DICT,
            timeout=15
        )
        n_boxes = len(ocr_data['level'])
        if n_boxes == 0:
            print("[ocr_processors] OCR did not detect any text boxes.")
            return None, None

        all_x, all_y, all_x_plus_w, all_y_plus_h = [], [], [], []
        num_valid_boxes_after_filter = 0

        img_cv_for_internal_debug_draw = None
        if return_debug_cv_image:
            img_cv_for_internal_debug_draw = cv2.cvtColor(np.array(img_pil_input.copy().convert('RGB')), cv2.COLOR_RGB2BGR)

        for i in range(n_boxes):
            if ocr_data['level'][i] == 5 and ocr_data['text'][i].strip() != "" and int(float(ocr_data['conf'][i])) >= min_confidence:
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                # --- ▼ バウンディングボックスのフィルタリング ▼ ---
                if not (w > 0 and h > 0):
                    continue
                if w < min_box_width or h < min_box_height:
                    if img_cv_for_internal_debug_draw is not None:
                        cv2.rectangle(img_cv_for_internal_debug_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 赤
                    continue
                if h > img_pil_height * max_box_height_ratio or w > img_pil_width * max_box_width_ratio:
                    if img_cv_for_internal_debug_draw is not None:
                        cv2.rectangle(img_cv_for_internal_debug_draw, (x, y), (x + w, y + h), (0, 165, 255), 1)  # オレンジ
                    continue
                aspect_ratio = w / h
                if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                    if img_cv_for_internal_debug_draw is not None:
                        cv2.rectangle(img_cv_for_internal_debug_draw, (x, y), (x + w, y + h), (255, 0, 255), 1)  # マゼンタ
                    continue
                # --- ▲ フィルタリングここまで ▲ ---
                all_x.append(x)
                all_y.append(y)
                all_x_plus_w.append(x + w)
                all_y_plus_h.append(y + h)
                num_valid_boxes_after_filter += 1
                if img_cv_for_internal_debug_draw is not None:
                    cv2.rectangle(img_cv_for_internal_debug_draw, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 緑

        if num_valid_boxes_after_filter == 0:
            print("[ocr_processors] OCR did not find enough valid text boxes after filtering.")
            img_cv_for_debug_output = img_cv_for_internal_debug_draw
            return None, img_cv_for_debug_output

        min_x, min_y = min(all_x), min(all_y)
        max_x_plus_w, max_y_plus_h = max(all_x_plus_w), max(all_y_plus_h)

        crop_x1 = max(0, min_x - padding)
        crop_y1 = max(0, min_y - padding)
        crop_x2 = min(img_pil_input.width, max_x_plus_w + padding)
        crop_y2 = min(img_pil_input.height, max_y_plus_h + padding)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            print(f"[ocr_processors] OCR Trim: Invalid crop region after filtering/padding.")
            img_cv_for_debug_output = img_cv_for_internal_debug_draw
            return None, img_cv_for_debug_output
        
        if img_cv_for_internal_debug_draw is not None:
            cv2.rectangle(img_cv_for_internal_debug_draw, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 2)  # 青
            img_cv_for_debug_output = img_cv_for_internal_debug_draw
        
        trimmed_image_pil = img_pil_input.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        print(f"[ocr_processors] OCR-based trimming successful. Cropped region: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
        return trimmed_image_pil, img_cv_for_debug_output
    except pytesseract.TesseractError as e:  # type: ignore
        print(f"[ocr_processors] Tesseract OCR error during OCR-based trimming: {e}")
    except RuntimeError as e:
        print(f"[ocr_processors] Tesseract runtime error (possibly timeout) during OCR-based trimming: {e}")
    except Exception as e:
        print(f"[ocr_processors] Unexpected error during OCR-based trimming: {e}")
        import traceback
        traceback.print_exc()
        img_cv_for_debug_output = None
        if return_debug_cv_image and 'img_cv_for_internal_debug_draw' in locals() and img_cv_for_internal_debug_draw is not None:
            img_cv_for_debug_output = img_cv_for_internal_debug_draw
        return None, img_cv_for_debug_output