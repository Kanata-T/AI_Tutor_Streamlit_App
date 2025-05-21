# utils/image_processing_parts/base_image_utils.py
from PIL import Image, ExifTags, ImageOps
from io import BytesIO
from typing import Optional, Dict, Tuple, Any, Literal
import cv2
import numpy as np
# import math # 角度計算用 (determine_90deg_rotation_by_projection_scoring では直接使っていない)
# scipy.ndimage のインポートは不要になったため削除

# ↓↓↓ 新しいOCRユーティリティファイルからインポート ↓↓↓
from .ocr_processors import PYTESSERACT_AVAILABLE, get_text_orientation_with_tesseract
# ↑↑↑ .ocr_processors の前のドット(.)は、同じディレクトリ内からの相対インポートを示します。

# --- 射影プロファイルスコアリングによる90度回転推定 ---
from .opencv_pipeline_utils import convert_pil_to_cv_gray, apply_adaptive_threshold

def get_image_orientation(img: Image.Image) -> Optional[int]:
    """
    画像のEXIF情報から向きタグを取得する。
    """
    try:
        exif = img.getexif()
        if not exif:
            return None
        for k, v in exif.items():
            if k in ExifTags.TAGS and ExifTags.TAGS[k] == 'Orientation':
                return v # type: ignore
    except Exception as e:
        print(f"[base_image_utils] Warning: Could not get EXIF data: {e}")
    return None

def correct_image_orientation(img: Image.Image) -> Image.Image:
    """
    画像のEXIF情報に基づいて向きを補正する。
    """
    orientation = get_image_orientation(img)
    if orientation == 2:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        img = img.rotate(180)
    elif orientation == 4:
        img = img.rotate(180).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        img = img.rotate(-90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        img = img.rotate(-90, expand=True)
    elif orientation == 7:
        img = img.rotate(90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        img = img.rotate(90, expand=True)
    return img

def convert_to_rgb_with_alpha_handling(img: Image.Image) -> Image.Image:
    """
    画像をRGBモードに変換する。アルファチャンネルを持つ場合は白背景で合成する。
    """
    if img.mode == 'RGB':
        return img.copy()
    if img.mode in ['RGBA', 'LA'] or (img.mode == 'P' and 'transparency' in img.info):
        try:
            background = Image.new("RGB", img.size, (255, 255, 255))
            img_to_paste = img if img.mode in ['RGBA', 'LA'] else img.convert('RGBA')
            background.paste(img_to_paste, mask=img_to_paste.split()[-1])
            return background
        except Exception as e:
            print(f"[base_image_utils] Warning: Error during alpha handling for RGB conversion: {e}. Falling back to simple convert.")
            return img.convert('RGB')
    else:
        return img.convert('RGB')

def apply_grayscale_pillow(img: Image.Image) -> Image.Image:
    """
    Pillowを使用して画像をグレースケールに変換する。
    """
    if img.mode == 'L':
        return img.copy()
    return img.convert('L')

def resize_image_pillow(img: Image.Image, max_total_pixels: Optional[int]) -> Image.Image:
    """
    Pillowを使用して、指定された総ピクセル数を超えないように画像をリサイズする。
    """
    if not max_total_pixels or max_total_pixels <= 0:
        return img.copy()
    current_pixels = img.width * img.height
    if current_pixels > max_total_pixels:
        scale_factor = (max_total_pixels / current_pixels) ** 0.5
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        if new_width >= 1 and new_height >= 1:
            try:
                return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"[base_image_utils] Warning: Error during resizing: {e}. Returning original image.")
                return img.copy()
        else:
            return img.copy()
    return img.copy()

def image_to_bytes(
    img_pil: Image.Image,
    target_format: Literal["JPEG", "PNG"] = "PNG",
    jpeg_quality: int = 85
) -> Optional[bytes]:
    """
    PIL Imageオブジェクトを指定されたフォーマットのバイトデータに変換する。
    """
    if not isinstance(img_pil, Image.Image):
        print(f"[base_image_utils] Error: Expected PIL Image, got {type(img_pil)}.")
        return None
    try:
        bio = BytesIO()
        img_to_save = img_pil.copy()
        if target_format == "JPEG":
            if img_to_save.mode not in ['RGB', 'L']:
                img_to_save = convert_to_rgb_with_alpha_handling(img_to_save)
            img_to_save.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
        elif target_format == "PNG":
            img_to_save.save(bio, format="PNG", optimize=True)
        else:
            print(f"[base_image_utils] Error: Unsupported target_format '{target_format}'. Use 'JPEG' or 'PNG'.")
            return None
        return bio.getvalue()
    except Exception as e:
        print(f"[base_image_utils] Error in image_to_bytes: {e}")
        return None

def cv_image_to_bytes(
    img_cv: np.ndarray,
    target_format: Literal["JPEG", "PNG"] = "PNG",
    jpeg_quality: int = 85
) -> Optional[bytes]:
    """
    OpenCV (numpy) 画像を指定されたフォーマットのバイトデータに変換する。
    """
    if not isinstance(img_cv, np.ndarray):
        print(f"[base_image_utils] Error: Expected numpy ndarray, got {type(img_cv)}.")
        return None
    try:
        img_pil: Optional[Image.Image] = None
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        elif len(img_cv.shape) == 2:
            img_pil = Image.fromarray(img_cv, mode='L')
        else:
            print(f"[base_image_utils] Error: cv_image_to_bytes - unsupported image shape {img_cv.shape}")
            return None
        if img_pil:
            return image_to_bytes(img_pil, target_format, jpeg_quality)
        return None
    except Exception as e:
        print(f"[base_image_utils] Error in cv_image_to_bytes: {e}")
        return None

def calculate_projection_profile_score(binary_image: np.ndarray) -> float:
    """
    二値化画像の水平射影プロファイルのスコアを計算する。
    スコアが高いほど「行らしい」とされる（ピークが鋭いなど）。
    ここでは単純に射影の分散をスコアとする例。
    Args:
        binary_image (np.ndarray): 2値画像（テキストが1=白, 背景が0=黒を期待）
    Returns:
        float: スコア値（分散）
    """
    if binary_image.size == 0:
        print("[base_image_utils] Warning: Empty binary image passed to calculate_projection_profile_score.")
        return 0.0
    projection = np.sum(binary_image, axis=1)
    if projection.size == 0:
        print("[base_image_utils] Warning: Projection array is empty in calculate_projection_profile_score.")
        return 0.0
    score = np.var(projection)
    return float(score)

def determine_90deg_rotation_by_projection_scoring(
    img_pil: Image.Image,
    thresh_block_size: int = 11, 
    thresh_c_value: int = 7
) -> int:
    """
    画像を0度, 90度, 180度, 270度回転させ、それぞれの角度での
    水平射影プロファイルのスコアを比較し、最もスコアが高い角度を
    最適な90度単位の向きとして推定する。横書きテキストを前提とする。
    """
    print("[base_image_utils] Estimating 90-degree rotation using projection profile scoring...")
    best_angle = 0
    max_score = -float('inf')

    for angle_option in [0, 90, 180, 270]:
        try:
            rotated_img_pil = img_pil.rotate(angle_option, expand=True, fillcolor='white') if angle_option != 0 else img_pil.copy()
            img_cv_gray = convert_pil_to_cv_gray(rotated_img_pil)
            if img_cv_gray is None:
                print(f"[base_image_utils] Angle {angle_option}°: Failed to convert to CV Gray for scoring.")
                continue
            
            img_cv_thresh = apply_adaptive_threshold(
                img_cv_gray, thresh_block_size, thresh_c_value, 
                threshold_type=cv2.THRESH_BINARY_INV # テキストを白にする
            )
            if img_cv_thresh is None:
                print(f"[base_image_utils] Angle {angle_option}°: Failed to apply adaptive threshold for scoring.")
                continue
            
            binary_image_for_score = img_cv_thresh / 255.0
            score = calculate_projection_profile_score(binary_image_for_score)
            print(f"[base_image_utils] Angle {angle_option}°: Projection score = {score:.4f}")

            if score > max_score:
                max_score = score
                best_angle = angle_option
        except Exception as e:
            print(f"[base_image_utils] Error scoring angle {angle_option}° for projection-based 90deg rotation: {e}")
            continue
            
    print(f"[base_image_utils] Projection-based 90deg rotation: Best angle = {best_angle}° with score {max_score:.4f}")
    return best_angle

def orient_image_comprehensively(img_pil_input: Image.Image) -> Image.Image:
    """
    画像の向きを総合的に自動補正する (90度単位のみ)。
    1. EXIFベースの補正。
    2. 射影プロファイルスコアリングによる90度回転推定。
    3. OCRベースの90度回転推定 (Tesseract OSD)。
    4. 上記2と3の結果を比較し、優先順位に従って90度回転を適用。
    """
    print(f"[base_image_utils] Starting comprehensive 90-degree orientation for image size: {img_pil_input.size}")
    current_img = img_pil_input.copy()

    # 1. EXIFベースの補正
    try:
        img_after_exif = correct_image_orientation(current_img)
        if img_after_exif.size != current_img.size:
            print(f"[base_image_utils] Applied EXIF orientation. New size: {img_after_exif.size}")
        current_img = img_after_exif
    except Exception as e_exif:
        print(f"[base_image_utils] Error during EXIF correction: {e_exif}")

    # 2. 射影プロファイルスコアリングによる90度回転角度を推定
    projection_rotation_angle = 0
    try:
        # TODO: determine_90deg_rotation_by_projection_scoring に渡す閾値パラメータを調整可能にする
        projection_rotation_angle = determine_90deg_rotation_by_projection_scoring(current_img.copy())
    except Exception as e_proj_rot:
        print(f"[base_image_utils] Error during projection-based 90deg rotation scoring: {e_proj_rot}")
        projection_rotation_angle = 0

    # 3. OCRベースの90度回転角度を推定
    ocr_rotation_angle = 0
    if PYTESSERACT_AVAILABLE:
        try:
            ocr_rotation_angle = get_text_orientation_with_tesseract(current_img.copy())
        except Exception as e_ocr_rot:
            print(f"[base_image_utils] Error during OCR-based 90-degree rotation estimation: {e_ocr_rot}")
            ocr_rotation_angle = 0
    else:
        print("[base_image_utils] Pytesseract not available for 90deg rotation estimation.")

    # 4. 射影ベースとOCRベースの結果を比較し、最終的な90度回転角度を決定
    final_90_degree_rotation = 0
    proj_suggests_rotation = projection_rotation_angle != 0
    ocr_suggests_rotation = ocr_rotation_angle != 0

    print(f"[base_image_utils] 90deg candidates: Projection={projection_rotation_angle}, OCR={ocr_rotation_angle}")

    if proj_suggests_rotation and ocr_suggests_rotation:
        if projection_rotation_angle != ocr_rotation_angle:
            final_90_degree_rotation = projection_rotation_angle
            print(f"[base_image_utils] Decision: Projection ({projection_rotation_angle}°) over OCR ({ocr_rotation_angle}°).")
        else:
            final_90_degree_rotation = projection_rotation_angle
            print(f"[base_image_utils] Decision: Projection and OCR agree ({projection_rotation_angle}°).")
    elif proj_suggests_rotation:
        final_90_degree_rotation = projection_rotation_angle
        print(f"[base_image_utils] Decision: Projection only ({projection_rotation_angle}°).")
    elif ocr_suggests_rotation:
        final_90_degree_rotation = ocr_rotation_angle
        print(f"[base_image_utils] Decision: OCR only ({ocr_rotation_angle}°).")
    else:
        final_90_degree_rotation = 0
        print("[base_image_utils] Decision: No 90-degree rotation needed from projection or OCR.")

    if final_90_degree_rotation != 0:
        try:
            print(f"[base_image_utils] Applying final 90-degree rotation: {final_90_degree_rotation}°.")
            current_img = current_img.rotate(final_90_degree_rotation, expand=True, fillcolor='white')
        except Exception as e_final_rot:
            print(f"[base_image_utils] Error applying final 90-degree rotation: {e_final_rot}")
    
    print(f"[base_image_utils] Comprehensive 90-degree orientation finished. Final image size: {current_img.size}")
    return current_img

def auto_orient_image_opencv(img_pil_input: Image.Image) -> Image.Image:
    """
    画像の向きを自動補正するためのエントリーポイント (90度単位のみ)。
    内部で包括的な90度単位の向き補正ロジックを呼び出す。
    """
    print(f"[base_image_utils] Starting auto_orient_image_opencv (90-deg only) for image size: {img_pil_input.size}")
    return orient_image_comprehensively(img_pil_input.copy())