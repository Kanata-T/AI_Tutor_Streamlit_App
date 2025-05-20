# utils/image_processor.py

from PIL import Image, ExifTags
from io import BytesIO
from typing import Optional, Dict, Literal, Tuple, List, Any
from utils import config_loader  # config_loader をインポート
import cv2  # ★ OpenCV をインポート ★
import numpy as np  # ★ Numpy をインポート ★
from scipy.signal import argrelmax # ★ scipyからargrelmaxをインポート ★

# --- 設定値読み込み ---
# config_loader.get_config() を使用し、image_processing セクションを取得
_full_config = config_loader.get_config()
IMG_PROC_CONFIG = _full_config.get("image_processing", {})

# Pillowの最大ピクセル数設定
pillow_max_pixels_cfg = IMG_PROC_CONFIG.get("pillow_max_image_pixels", 225000000)
if pillow_max_pixels_cfg is not None:
    try:
        Image.MAX_IMAGE_PIXELS = int(pillow_max_pixels_cfg)
    except (ValueError, TypeError):
        print(f"[image_processor] Warning: Invalid 'pillow_max_image_pixels' in config: {pillow_max_pixels_cfg}. Using Pillow default.")
        Image.MAX_IMAGE_PIXELS = None
else:
    Image.MAX_IMAGE_PIXELS = None

SUPPORTED_MIME_TYPES = IMG_PROC_CONFIG.get("supported_mime_types", ["image/png", "image/jpeg", "image/webp"])
CONVERTABLE_MIME_TYPES = IMG_PROC_CONFIG.get("convertable_mime_types", {"image/gif": "image/png", "image/bmp": "image/png"})
DEFAULT_MAX_PIXELS = IMG_PROC_CONFIG.get("default_max_pixels_for_resizing", 4000000)
DEFAULT_OUTPUT_FORMAT = IMG_PROC_CONFIG.get("default_output_format", "JPEG").upper()
DEFAULT_JPEG_QUALITY = IMG_PROC_CONFIG.get("default_jpeg_quality", 85)
APPLY_GRAYSCALE_DEFAULT = IMG_PROC_CONFIG.get("apply_grayscale", True)

# OpenCVトリミングのデフォルト設定 (configから取得、ネスト構造を意識)
DEFAULT_OPENCV_TRIM_SETTINGS = IMG_PROC_CONFIG.get("opencv_trimming", {})
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("apply", True)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("padding", 10)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("adaptive_thresh_block_size", 11)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("adaptive_thresh_c", 5)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("min_contour_area_ratio", 0.001)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("gaussian_blur_kernel", [5, 5])
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("morph_open_apply", False)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("morph_open_kernel_size", 3)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("morph_open_iterations", 1)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("morph_close_apply", False)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("morph_close_kernel_size", 3)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("morph_close_iterations", 1)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("haar_apply", False) # ★ Haar-Like
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("haar_rect_h", 20)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("haar_peak_threshold", 10.0)
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("h_proj_apply", False) # ★ 水平射影
DEFAULT_OPENCV_TRIM_SETTINGS.setdefault("h_proj_threshold_ratio", 0.01)

def get_image_orientation(img: Image.Image) -> Optional[int]:
    """
    画像のEXIF情報から向きタグを取得する。

    Args:
        img: PIL Imageオブジェクト。

    Returns:
        EXIFの向きタグの値 (int)、または見つからない/エラーの場合はNone。
    """
    try:
        exif = img.getexif()
        if not exif: return None
        for k, v in exif.items():
            if k in ExifTags.TAGS and ExifTags.TAGS[k] == 'Orientation': return v
    except Exception as e:
        print(f"[image_processor] Warning: Could not get EXIF data: {e}")
    return None

def correct_image_orientation_from_exif(img: Image.Image) -> Image.Image:
    """
    画像のEXIF情報に基づいて向きを補正する。

    Args:
        img: PIL Imageオブジェクト。

    Returns:
        向きが補正されたPIL Imageオブジェクト。補正不要/失敗時は元画像を返す。
    """
    orientation = get_image_orientation(img)
    if orientation == 2: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3: img = img.rotate(180)
    elif orientation == 4: img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5: img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6: img = img.rotate(-90, expand=True)
    elif orientation == 7: img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8: img = img.rotate(90, expand=True)
    return img

# === Haar-Like 行検出関数 ===
def calc_haarlike_vertical(img_gray: np.ndarray, rect_h: int) -> Optional[np.ndarray]:
    if rect_h <= 0 or rect_h % 2 != 0:
        print(f"[image_processor_cv] Error: rect_h for Haar-Like must be positive even, got {rect_h}"); return None
    if img_gray.ndim != 2: print(f"[image_processor_cv] Error: Haar-Like input must be grayscale."); return None
    pattern_h, height = rect_h // 2, img_gray.shape[0]
    if height <= rect_h: print(f"[image_processor_cv] Warn: Image height <= rect_h. Skipping Haar."); return np.array([])
    out = np.zeros(height - rect_h)
    try:
        for i in range(height - rect_h):
            s1_e, s2_s, s2_e = min(height,i+pattern_h), min(height,i+pattern_h), min(height,i+rect_h)
            if s1_e <= i or s2_e <= s2_s: out[i]=0; continue
            out[i] = np.mean(img_gray[i:s1_e, :]) - np.mean(img_gray[s2_s:s2_e, :])
    except Exception as e: print(f"[image_processor_cv] Error in calc_haarlike_vertical: {e}"); return None
    return out

def peak_detection_vertical(data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if data is None or len(data) == 0: return np.array([]), np.array([])
    peaks_start = argrelmax(data)[0]
    peaks_end = argrelmax(-data)[0]
    valid_starts = peaks_start[np.where(data[peaks_start] > threshold)]
    valid_ends = peaks_end[np.where(np.abs(data[peaks_end]) > threshold)]
    return valid_starts, valid_ends

# === 水平射影プロファイル関数 ===
def get_horizontal_projection(img_binary: np.ndarray, threshold_ratio: float = 0.01) -> Optional[Tuple[int, int]]:
    if img_binary is None or img_binary.ndim != 2: return None
    try:
        projection = np.sum(img_binary, axis=0)
        if np.max(projection) == 0: return None
        threshold = np.max(projection) * threshold_ratio
        text_cols = np.where(projection > threshold)[0]
        if len(text_cols) == 0: return None
        return np.min(text_cols), np.max(text_cols)
    except Exception as e: print(f"[image_processor_cv] Error in get_horizontal_projection: {e}"); return None

# === メインのトリミング関数 (trim_whitespace_opencv) ===
def trim_whitespace_opencv(
    img_pil: Image.Image,
    padding: int, adaptive_thresh_block_size: int, adaptive_thresh_c: int,
    min_contour_area_ratio: float, gaussian_blur_kernel: Tuple[int, int],
    haar_apply: bool, haar_rect_h: int, haar_peak_threshold: float,
    h_proj_apply: bool, h_proj_threshold_ratio: float,
    morph_open_apply: bool, morph_open_kernel_size: int, morph_open_iterations: int,
    morph_close_apply: bool, morph_close_kernel_size: int, morph_close_iterations: int,
    debug_image_collector: Optional[List[Dict[str, any]]] = None
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    OpenCVで余白トリミングし、(トリミング後PIL画像, 二値化トリミングPIL画像)を返す。
    """
    img_pil_original_for_crop = img_pil.copy()
    img_binary_for_ocr_crop = None
    try:
        # 1. グレースケール化
        img_cv_gray = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
        original_height, original_width = img_cv_gray.shape[:2]
        if debug_image_collector is not None: debug_image_collector.append({"label": "CV Trim - 0. 入力(Gray)", "data": cv_to_pil_bytes(img_cv_gray.copy()), "mode": "L", "size": (original_width, original_height)})

        # A-1. Haar-Likeで行の垂直範囲を推定
        y_min_haar_final, y_max_haar_final = 0, original_height # 最終的に使うHaarのY範囲
        if haar_apply:
            print(f"[image_processor_cv] Applying Haar-Like: rect_h={haar_rect_h}, peak_thresh={haar_peak_threshold}")
            haar_response = calc_haarlike_vertical(img_cv_gray, haar_rect_h)
            if haar_response is not None and len(haar_response) > 0:
                p_start, p_end = peak_detection_vertical(haar_response, haar_peak_threshold)
                if len(p_start) > 0 and len(p_end) > 0:
                    temp_y_min = np.min(p_start)
                    valid_ends = p_end[p_end > temp_y_min]
                    if len(valid_ends) > 0:
                        y_min_haar_final = max(0, temp_y_min)
                        y_max_haar_final = min(original_height, np.max(valid_ends) + haar_rect_h)
                        print(f"[image_processor_cv] Haar Y-Range determined: {y_min_haar_final}-{y_max_haar_final}")
                        if debug_image_collector is not None:
                            viz_h = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
                            cv2.rectangle(viz_h, (0,y_min_haar_final),(original_width-1,y_max_haar_final),(0,255,255),2)
                            debug_image_collector.append({"label": "CV Trim - A1. Haar Y-Range", "data": cv_to_pil_bytes(viz_h), "mode":"RGB", "size":(original_width,original_height)})

        # 2. ブラー
        img_blurred = img_cv_gray
        if gaussian_blur_kernel[0] > 0 and gaussian_blur_kernel[1] > 0 and gaussian_blur_kernel[0] % 2 != 0 and gaussian_blur_kernel[1] % 2 != 0:
            img_blurred = cv2.GaussianBlur(img_cv_gray, gaussian_blur_kernel, 0)
            if debug_image_collector is not None:
                debug_image_collector.append({"label": f"CV Trim - 1. ブラー(K{gaussian_blur_kernel})", "data": cv_to_pil_bytes(img_blurred.copy()), "mode": "L", "size": (original_width, original_height)})

        # 3. 二値化
        current_block_size = adaptive_thresh_block_size
        if current_block_size < 3: current_block_size = 3
        if current_block_size % 2 == 0: current_block_size += 1
        img_thresh = cv2.adaptiveThreshold(
            img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            current_block_size, adaptive_thresh_c
        )
        if debug_image_collector is not None:
            debug_image_collector.append({"label": f"CV Trim - 2. 二値化(B{current_block_size},C{adaptive_thresh_c})", "data": cv_to_pil_bytes(img_thresh.copy()), "mode": "L", "size": (original_width, original_height)})

        # A-2. 水平射影でテキストのX範囲を推定
        x_min_hproj_final, x_max_hproj_final = 0, original_width # 最終的に使うH-ProjのX範囲
        if h_proj_apply:
            print(f"[image_processor_cv] Applying H-Projection: thresh_ratio={h_proj_threshold_ratio}")
            proj_res = get_horizontal_projection(img_thresh, h_proj_threshold_ratio) # 二値化画像を使う
            if proj_res:
                x_min_hproj_final, x_max_hproj_final = proj_res
                print(f"[image_processor_cv] H-Proj X-Range determined: {x_min_hproj_final}-{x_max_hproj_final}")
                if debug_image_collector is not None:
                    viz_hp = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(viz_hp, (x_min_hproj_final,0),(x_max_hproj_final,original_height-1),(255,255,0),2)
                    debug_image_collector.append({"label": "CV Trim - A2. H-Proj X-Range", "data": cv_to_pil_bytes(viz_hp), "mode":"RGB", "size":(original_width,original_height)})

        # 4. モルフォロジー
        img_morphed = img_thresh
        if morph_open_apply and morph_open_kernel_size > 0:
            k_open = morph_open_kernel_size if morph_open_kernel_size % 2 != 0 else morph_open_kernel_size + 1
            kernel_open = np.ones((k_open, k_open), np.uint8)
            img_morphed = cv2.morphologyEx(img_morphed, cv2.MORPH_OPEN, kernel_open, iterations=morph_open_iterations)
            if debug_image_collector is not None:
                debug_image_collector.append({"label": f"CV Trim - MorphOPEN(K{k_open},I{morph_open_iterations})", "data": cv_to_pil_bytes(img_morphed.copy()), "mode": "L", "size": (original_width, original_height)})
        if morph_close_apply and morph_close_kernel_size > 0:
            k_close = morph_close_kernel_size if morph_close_kernel_size % 2 != 0 else morph_close_kernel_size + 1
            kernel_close = np.ones((k_close, k_close), np.uint8)
            img_morphed = cv2.morphologyEx(img_morphed, cv2.MORPH_CLOSE, kernel_close, iterations=morph_close_iterations)
            if debug_image_collector is not None:
                debug_image_collector.append({"label": f"CV Trim - MorphCLOSE(K{k_close},I{morph_close_iterations})", "data": cv_to_pil_bytes(img_morphed.copy()), "mode": "L", "size": (original_width, original_height)})
        contours_img_input = img_morphed

        # 5. 輪郭検出
        contours, hierarchy = cv2.findContours(contours_img_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if debug_image_collector is not None and contours:
            img_all_contours_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_all_contours_drawn, contours, -1, (0,0,255), 1)
            debug_image_collector.append({"label": "CV Trim - 3a. 全輪郭(フィルタ前)", "data": cv_to_pil_bytes(img_all_contours_drawn), "mode": "RGB", "size": (original_width, original_height)})
        if not contours:
            return img_pil_original_for_crop, None

        # 6. 有効な輪郭の選択
        min_abs_area = original_width * original_height * min_contour_area_ratio
        valid_contours_list = []
        for c_item in contours:
            if cv2.contourArea(c_item) >= min_abs_area:
                # ★ オプション: Haar/H-Proj範囲でさらにフィルタリング ★
                apply_spatial_filter = haar_apply or h_proj_apply
                if apply_spatial_filter:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(c_item)
                    cx, cy = x_c + w_c//2, y_c + h_c//2
                    in_haar_y = not haar_apply or (y_min_haar_final <= cy <= y_max_haar_final)
                    in_hproj_x = not h_proj_apply or (x_min_hproj_final <= cx <= x_max_hproj_final)
                    if in_haar_y and in_hproj_x:
                        valid_contours_list.append(c_item)
                else: # 空間フィルタなし
                    valid_contours_list.append(c_item)
        if debug_image_collector is not None and valid_contours_list:
            img_valid_contours_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_valid_contours_drawn, valid_contours_list, -1, (0,255,0), 1)
            debug_image_collector.append({"label": "CV Trim - 3b. 有効輪郭(フィルタ後)", "data": cv_to_pil_bytes(img_valid_contours_drawn), "mode": "RGB", "size": (original_width, original_height)})
        if not valid_contours_list:
            return img_pil_original_for_crop, None

        # 7. 輪郭結合 & BBox計算
        merged_x_min, merged_y_min = original_width, original_height
        merged_x_max, merged_y_max = 0, 0
        has_valid_bbox_from_contours = False
        for contour_item in valid_contours_list:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour_item)
            merged_x_min = min(merged_x_min, x_c)
            merged_y_min = min(merged_y_min, y_c)
            merged_x_max = max(merged_x_max, x_c + w_c)
            merged_y_max = max(merged_y_max, y_c + h_c)
            has_valid_bbox_from_contours = True
        if not has_valid_bbox_from_contours:
            return img_pil_original_for_crop, None
        x, y, w, h = merged_x_min, merged_y_min, merged_x_max - merged_x_min, merged_y_max - merged_y_min
        if debug_image_collector is not None:
            img_final_bbox_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img_final_bbox_drawn, (x,y), (x+w,y+h), (255,0,0), 2)
            debug_image_collector.append({"label": "CV Trim - 3c. 最終BoundingBox", "data": cv_to_pil_bytes(img_final_bbox_drawn), "mode": "RGB", "size": (original_width, original_height)})

        # ★ 7b. 最終バウンディングボックスの補正 (Haar-Likeと水平射影の結果を適用) ★
        final_x, final_y, final_w, final_h = x, y, w, h # 輪郭検出ベースのBBoxを初期値に

        if haar_apply and y_max_haar_final > y_min_haar_final: # Haarの結果が有効なら
            # HaarのY範囲を現在のBBoxにマージ（拡張）する
            # ただし、輪郭検出の結果も尊重し、完全にHaarの結果で上書きはしない
            # ここでは、輪郭検出で見つかったy, y+h と、Haarで見つかったy範囲を包含するようにする
            new_y_min_merged = min(final_y, y_min_haar_final)
            new_y_max_merged = max(final_y + final_h, y_max_haar_final)
            final_y = new_y_min_merged
            final_h = new_y_max_merged - final_y
            print(f"[image_processor_cv] Applied Haar Y-correction to BBox: y={final_y}, h={final_h}")

        if h_proj_apply and x_max_hproj_final > x_min_hproj_final: # 水平射影の結果が有効なら
            # 水平射影のX範囲を現在のBBoxにマージ（拡張）する
            new_x_min_merged = min(final_x, x_min_hproj_final)
            new_x_max_merged = max(final_x + final_w, x_max_hproj_final)
            final_x = new_x_min_merged
            final_w = new_x_max_merged - final_x
            print(f"[image_processor_cv] Applied H-Proj X-correction to BBox: x={final_x}, w={final_w}")

        if debug_image_collector is not None and (haar_apply or h_proj_apply):
            img_corrected_bbox_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            # 輪郭検出のBBox (青) と、補正後のBBox (オレンジ) を両方描画して比較
            cv2.rectangle(img_corrected_bbox_drawn, (x,y), (x+w,y+h), (255,0,0), 3) # 元の輪郭BBox (青)
            cv2.rectangle(img_corrected_bbox_drawn, (final_x,final_y), (final_x+final_w,final_y+final_h), (0,165,255), 2) # 補正後BBox (オレンジ)
            debug_image_collector.append({"label": "CV Trim - 3d. 補正後BBox", "data": cv_to_pil_bytes(img_corrected_bbox_drawn), "mode":"RGB", "size":(original_width,original_height)})

        # 8. パディングと切り出し (補正後の final_x, final_y, final_w, final_h を使用)
        x1, y1 = max(0, final_x - padding), max(0, final_y - padding)
        x2, y2 = min(original_width, final_x + final_w + padding), min(original_height, final_y + final_h + padding)

        if x2 <= x1 or y2 <= y1:
            print(f"[image_processor_cv] Warning: Invalid crop region after padding/correction. Skipping trim.")
            return img_pil_original_for_crop, None

        cropped_img_pil = img_pil_original_for_crop.crop((x1, y1, x2, y2))
        if debug_image_collector is not None:
            debug_image_collector.append({"label": "CV Trim - 4a. トリミング結果(元)", "data": pil_to_bytes(cropped_img_pil.copy()), "mode": cropped_img_pil.mode, "size": cropped_img_pil.size})
        # 二値化画像の同一範囲スライス
        cropped_binary_np = contours_img_input[y1:y2, x1:x2]
        if cropped_binary_np.size > 0:
            img_binary_for_ocr_crop = Image.fromarray(cropped_binary_np, mode='L')
            if debug_image_collector is not None:
                debug_image_collector.append({"label": "CV Trim - 4b. トリミング結果(二値OCR用)", "data": pil_to_bytes(img_binary_for_ocr_crop.copy(), format_str="PNG"), "mode": "L", "size": img_binary_for_ocr_crop.size})
        else:
            print("[image_processor_cv] Warning: Cropped binary for OCR is empty.")
        return cropped_img_pil, img_binary_for_ocr_crop
    except Exception as e:
        print(f"[image_processor_cv] Error in trim: {e}")
        import traceback; traceback.print_exc()
        return img_pil_original_for_crop, None

# === バイト変換ヘルパー関数 (pil_to_bytes, cv_to_pil_bytes) ===
def pil_to_bytes(img_pil: Image.Image, format_str: str = "PNG") -> Optional[bytes]: # format引数名を変更
    if not img_pil: return None
    try:
        bio = BytesIO()
        save_fmt = format_str.upper()
        if save_fmt not in ["PNG", "JPEG"]: save_fmt = "PNG"
        temp_img = img_pil.copy() # 元画像に影響を与えない
        if save_fmt == "JPEG" and temp_img.mode not in ['RGB', 'L']:
            if temp_img.mode == 'RGBA' or temp_img.mode == 'LA': # アルファがある場合は白背景でRGB化
                bg = Image.new("RGB", temp_img.size, (255,255,255))
                bg.paste(temp_img, mask=temp_img.split()[-1])
                temp_img = bg
            else:
                temp_img = temp_img.convert('RGB')
        temp_img.save(bio, format=save_fmt, quality=DEFAULT_JPEG_QUALITY if save_fmt == "JPEG" else None) # JPEG品質適用
        return bio.getvalue()
    except Exception as e: print(f"[image_processor] Error in pil_to_bytes: {e}"); return None

def cv_to_pil_bytes(img_cv: np.ndarray, format_str: str = "PNG") -> Optional[bytes]:
    if img_cv is None: return None
    try:
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3: # BGRカラー
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        elif len(img_cv.shape) == 2: # グレースケール
            img_pil = Image.fromarray(img_cv, mode='L')
        else: print(f"[image_processor] Error: cv_to_pil_bytes unexpected shape {img_cv.shape}"); return None
        return pil_to_bytes(img_pil, format_str)
    except Exception as e: print(f"[image_processor] Error in cv_to_pil_bytes: {e}"); return None

# === メインの前処理関数 (preprocess_uploaded_image) ===
def preprocess_uploaded_image(
    uploaded_file_data: bytes, mime_type: str,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    output_format: Literal["JPEG", "PNG"] = DEFAULT_OUTPUT_FORMAT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    grayscale: bool = APPLY_GRAYSCALE_DEFAULT,
    apply_trimming_opencv_override: Optional[bool] = None,
    trim_params_override: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, any]]:
    debug_images: List[Dict[str, Any]] = []
    if not uploaded_file_data:
        print("[image_processor] Error: No image data provided for preprocessing.")
        return None

    original_mime_type = mime_type.lower()
    target_output_format = output_format.upper()  # 大文字に統一
    processed_mime_type = ""

    if original_mime_type in CONVERTABLE_MIME_TYPES:
        processed_mime_type = CONVERTABLE_MIME_TYPES[original_mime_type]
        if processed_mime_type == "image/png":
            target_output_format = "PNG"
        elif processed_mime_type == "image/jpeg":
            target_output_format = "JPEG"
        print(f"[image_processor] Info: Converting {original_mime_type} to {processed_mime_type} (output as {target_output_format}).")
    elif original_mime_type in SUPPORTED_MIME_TYPES:
        if target_output_format == "JPEG":
            processed_mime_type = "image/jpeg"
        elif target_output_format == "PNG":
            processed_mime_type = "image/png"
        else:
            print(f"[image_processor] Warning: Unsupported target_output_format '{target_output_format}' for {original_mime_type}. Defaulting to {DEFAULT_OUTPUT_FORMAT}.")
            target_output_format = DEFAULT_OUTPUT_FORMAT  # configのデフォルトを使用
            processed_mime_type = "image/jpeg" if target_output_format == "JPEG" else "image/png"
    else:
        print(f"[image_processor] Error: Unsupported original MIME type '{original_mime_type}' and not in convertable list.")
        return {"error": f"サポートされていない画像形式です: {original_mime_type}。"}

    effective_cv_trim_settings = DEFAULT_OPENCV_TRIM_SETTINGS.copy()
    if trim_params_override: effective_cv_trim_settings.update(trim_params_override)
    should_apply_opencv_trim = effective_cv_trim_settings.get("apply", False)
    if apply_trimming_opencv_override is not None: should_apply_opencv_trim = apply_trimming_opencv_override
    
    print(f"[image_processor] Final decision for should_apply_opencv_trim: {should_apply_opencv_trim}")
    print(f"[image_processor] Effective OpenCV trim settings to be used: {effective_cv_trim_settings}")

    try:
        img_initial_pil = Image.open(BytesIO(uploaded_file_data))
        debug_images.append({
            "label": "0. 初期読み込み (Pillow)",
            "data": pil_to_bytes(img_initial_pil.copy()),
            "mode": img_initial_pil.mode,
            "size": img_initial_pil.size
        })
        img_oriented_pil = correct_image_orientation_from_exif(img_initial_pil.copy())
        if img_oriented_pil.size != img_initial_pil.size or get_image_orientation(img_initial_pil) not in [None, 1]:
            debug_images.append({
                "label": "1. 向き補正後 (Pillow)",
                "data": pil_to_bytes(img_oriented_pil.copy()),
                "mode": img_oriented_pil.mode,
                "size": img_oriented_pil.size
            })
        img_for_final_processing = img_oriented_pil.copy()
        img_for_ocr_input_pil = None

        if should_apply_opencv_trim:
            trimmed_pil, trimmed_binary_pil = trim_whitespace_opencv(
                img_oriented_pil,
                padding=int(effective_cv_trim_settings.get("padding")),
                adaptive_thresh_block_size=int(effective_cv_trim_settings.get("adaptive_thresh_block_size")),
                adaptive_thresh_c=int(effective_cv_trim_settings.get("adaptive_thresh_c")),
                min_contour_area_ratio=float(effective_cv_trim_settings.get("min_contour_area_ratio")),
                gaussian_blur_kernel=tuple(effective_cv_trim_settings.get("gaussian_blur_kernel")),
                haar_apply=bool(effective_cv_trim_settings.get("haar_apply")),
                haar_rect_h=int(effective_cv_trim_settings.get("haar_rect_h")),
                haar_peak_threshold=float(effective_cv_trim_settings.get("haar_peak_threshold")),
                h_proj_apply=bool(effective_cv_trim_settings.get("h_proj_apply")),
                h_proj_threshold_ratio=float(effective_cv_trim_settings.get("h_proj_threshold_ratio")),
                morph_open_apply=bool(effective_cv_trim_settings.get("morph_open_apply")),
                morph_open_kernel_size=int(effective_cv_trim_settings.get("morph_open_kernel_size")),
                morph_open_iterations=int(effective_cv_trim_settings.get("morph_open_iterations")),
                morph_close_apply=bool(effective_cv_trim_settings.get("morph_close_apply")),
                morph_close_kernel_size=int(effective_cv_trim_settings.get("morph_close_kernel_size")),
                morph_close_iterations=int(effective_cv_trim_settings.get("morph_close_iterations")),
                debug_image_collector=debug_images
            )
            if trimmed_pil:
                img_for_final_processing = trimmed_pil
                if trimmed_binary_pil:
                    img_for_ocr_input_pil = trimmed_binary_pil
                else:
                    img_for_ocr_input_pil = trimmed_pil.convert('L') if grayscale else trimmed_pil
            else:
                img_for_ocr_input_pil = img_oriented_pil.convert('L') if grayscale else img_oriented_pil
        else:
            img_for_ocr_input_pil = img_oriented_pil.convert('L') if grayscale else img_oriented_pil.copy()

        # --- 表示/Vision用画像の処理 ---
        img_display_vision = img_for_final_processing.copy()
        img_before_rgb_conversion = img_for_final_processing.copy()
        if img_for_final_processing.mode == 'RGBA' or img_for_final_processing.mode == 'LA' or (img_for_final_processing.mode == 'P' and 'transparency' in img_for_final_processing.info):
            try:
                img_rgb = Image.new("RGB", img_for_final_processing.size, (255, 255, 255))
                img_rgb.paste(img_for_final_processing, mask=img_for_final_processing.split()[-1])
                img_for_final_processing = img_rgb
            except Exception:
                img_for_final_processing = img_for_final_processing.convert('RGB')
        elif img_for_final_processing.mode != 'RGB':
            img_for_final_processing = img_for_final_processing.convert('RGB')
        if img_for_final_processing.mode != img_before_rgb_conversion.mode or img_for_final_processing.size != img_before_rgb_conversion.size:
            debug_images.append({
                "label": "3. RGB変換後 (Pillow)",
                "data": pil_to_bytes(img_for_final_processing.copy()),
                "mode": img_for_final_processing.mode,
                "size": img_for_final_processing.size
            })
        if grayscale:
            img_before_grayscale = img_for_final_processing.copy()
            img_for_final_processing = img_for_final_processing.convert('L')
            debug_images.append({
                "label": "4. グレースケール化後 (Pillow)",
                "data": pil_to_bytes(img_for_final_processing.copy()),
                "mode": img_for_final_processing.mode,
                "size": img_for_final_processing.size
            })
        current_pixels = img_for_final_processing.width * img_for_final_processing.height
        if max_pixels > 0 and current_pixels > max_pixels:
            img_before_resize = img_for_final_processing.copy()
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_width = int(img_for_final_processing.width * scale_factor)
            new_height = int(img_for_final_processing.height * scale_factor)
            if new_width >= 1 and new_height >= 1:
                img_for_final_processing = img_for_final_processing.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if img_for_final_processing.size != img_before_resize.size:
                debug_images.append({
                    "label": "5. リサイズ後 (Pillow)",
                    "data": pil_to_bytes(img_for_final_processing.copy()),
                    "mode": img_for_final_processing.mode,
                    "size": img_for_final_processing.size
                })

        # --- OCR用画像のリサイズ ---
        if img_for_ocr_input_pil and max_pixels > 0:
            ocr_current_pixels = img_for_ocr_input_pil.width * img_for_ocr_input_pil.height
            if ocr_current_pixels > max_pixels:
                scale_factor = (max_pixels / ocr_current_pixels) ** 0.5
                new_width = int(img_for_ocr_input_pil.width * scale_factor)
                new_height = int(img_for_ocr_input_pil.height * scale_factor)
                if new_width >= 1 and new_height >= 1:
                    img_for_ocr_input_pil = img_for_ocr_input_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # --- バイトデータ変換 ---
        processed_image_data_display = pil_to_bytes(img_display_vision, format_str=target_output_format)
        final_processed_mime_type = "image/jpeg" if target_output_format == "JPEG" else "image/png"

        ocr_image_data_bytes = None
        ocr_mime_type = "image/png"
        if img_for_ocr_input_pil:
            if img_for_ocr_input_pil.mode != 'L':
                img_for_ocr_input_pil = img_for_ocr_input_pil.convert('L')
            ocr_image_data_bytes = pil_to_bytes(img_for_ocr_input_pil, format_str="PNG")
            if debug_image_collector is not None and ocr_image_data_bytes:
                debug_images.append({"label": "Final OCR Input", "data": ocr_image_data_bytes, "mode": "L", "size": img_for_ocr_input_pil.size})

        if not processed_image_data_display:
            return {"error": "最終的な画像バイト変換に失敗しました。", "debug_images": debug_images}
        result_dict = {
            "processed_image": {"mime_type": final_processed_mime_type, "data": processed_image_data_display},
            "debug_images": debug_images
        }
        if ocr_image_data_bytes:
            result_dict["ocr_input_image"] = {"mime_type": ocr_mime_type, "data": ocr_image_data_bytes}
        return result_dict

    except Image.DecompressionBombError as e_bomb:
        return {"error": "画像が大きすぎるか、非対応の形式の可能性があります。", "debug_images": debug_images}
    except ValueError as e_val:
        if "Pixel count exceeds limit" in str(e_val):
            return {"error": "画像のピクセル数が大きすぎます。", "debug_images": debug_images}
        else:
            return {"error": "画像処理中にエラーが発生しました。", "debug_images": debug_images}
    except Exception as e:
        return {"error": "画像処理中に予期せぬエラーが発生しました。", "debug_images": debug_images}