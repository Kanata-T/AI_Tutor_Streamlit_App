# utils/image_processing_parts/base_image_utils.py
from PIL import Image, ExifTags, ImageOps
from io import BytesIO
from typing import Optional, Dict, Tuple, Any, Literal # Literalを追加
import cv2 # cv_to_pil_bytes で使用
import numpy as np # cv_to_pil_bytes で使用
import math # 角度計算用
import scipy.ndimage # 射影ヒストグラムdeskew用

# ↓↓↓ 新しいOCRユーティリティファイルからインポート ↓↓↓
from .ocr_processors import PYTESSERACT_AVAILABLE, get_text_orientation_with_tesseract
# ↑↑↑ .ocr_processors の前のドット(.)は、同じディレクトリ内からの相対インポートを示します。

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
    (元の correct_image_orientation_from_exif から名前を少し一般化)
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
        return img.copy() # 既にRGBならコピーを返す

    if img.mode in ['RGBA', 'LA'] or (img.mode == 'P' and 'transparency' in img.info):
        try:
            # アルファチャンネルを考慮して白背景に合成
            background = Image.new("RGB", img.size, (255, 255, 255))
            # RGBAやLAの場合はそのままペースト、Pモードで透明度がある場合はRGBAに変換してからペースト
            img_to_paste = img if img.mode in ['RGBA', 'LA'] else img.convert('RGBA')
            background.paste(img_to_paste, mask=img_to_paste.split()[-1])
            return background
        except Exception as e:
            print(f"[base_image_utils] Warning: Error during alpha handling for RGB conversion: {e}. Falling back to simple convert.")
            return img.convert('RGB') # フォールバック
    else:
        return img.convert('RGB') # その他のモードは単純に変換

def apply_grayscale_pillow(img: Image.Image) -> Image.Image:
    """
    Pillowを使用して画像をグレースケールに変換する。
    """
    if img.mode == 'L':
        return img.copy() # 既にグレースケールならコピーを返す
    return img.convert('L')

def resize_image_pillow(img: Image.Image, max_total_pixels: Optional[int]) -> Image.Image:
    """
    Pillowを使用して、指定された総ピクセル数を超えないように画像をリサイズする。
    max_total_pixelsがNoneまたは0以下の場合はリサイズしない。
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
                return img.copy() # エラー時は元画像を返す
        else: # 計算後のサイズが小さすぎる場合
            return img.copy()
    return img.copy() # リサイズ不要ならコピーを返す

def image_to_bytes(
    img_pil: Image.Image,
    target_format: Literal["JPEG", "PNG"] = "PNG",
    jpeg_quality: int = 85 # configから読むデフォルト値を想定
) -> Optional[bytes]:
    """
    PIL Imageオブジェクトを指定されたフォーマットのバイトデータに変換する。
    (元のpil_to_bytesから改名し、引数名を明確化)
    """
    if not isinstance(img_pil, Image.Image):
        print(f"[base_image_utils] Error: Expected PIL Image, got {type(img_pil)}.")
        return None
    try:
        bio = BytesIO()
        # JPEGの場合、RGBまたはLモードである必要がある
        img_to_save = img_pil.copy() # 元画像に影響を与えないようにコピー
        if target_format == "JPEG":
            if img_to_save.mode not in ['RGB', 'L']:
                img_to_save = convert_to_rgb_with_alpha_handling(img_to_save) # アルファ処理含むRGB変換
            img_to_save.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
        elif target_format == "PNG":
            # PNGは様々なモードをサポート。必要に応じて 'L' や 'RGB' に変換しても良いが、
            # ここではPillowのsaveに任せる。
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
    jpeg_quality: int = 85 # configから読むデフォルト値を想定
) -> Optional[bytes]:
    """
    OpenCV (numpy) 画像を指定されたフォーマットのバイトデータに変換する。
    (元のcv_to_pil_bytesから改名し、PIL Imageを介さずに直接変換も考慮したが、
     Pillowの保存機能が強力なので、一旦Pillow経由を維持)
    """
    if not isinstance(img_cv, np.ndarray):
        print(f"[base_image_utils] Error: Expected numpy ndarray, got {type(img_cv)}.")
        return None
    try:
        img_pil: Optional[Image.Image] = None
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3: # BGRカラーと仮定
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        elif len(img_cv.shape) == 2: # グレースケール
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
        binary_image (np.ndarray): 2値画像（テキストが白=1, 背景が黒=0）
    Returns:
        float: スコア値（分散）
    """
    if binary_image.size == 0:
        return 0.0
    projection = np.sum(binary_image, axis=1)
    score = np.var(projection)
    return float(score)

def deskew_image_projection_profile(img_pil: Image.Image, angle_range: float = 5.0, angle_step: float = 0.25) -> Image.Image:
    """
    射影プロファイルを用いて画像の傾き（スキュー）を補正する。
    横書きテキストを前提とする。
    Args:
        img_pil (Image.Image): 入力画像
        angle_range (float): 探索する角度範囲（±angle_range度）
        angle_step (float): 角度ステップ（度）
    Returns:
        Image.Image: スキュー補正後の画像
    """
    try:
        img_cv_gray = np.array(img_pil.convert('L'))
        thresh = cv2.adaptiveThreshold(img_cv_gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 10)
        best_angle = 0.0
        max_score = -1.0
        print(f"[base_image_utils] Deskewing with projection profile (range: ±{angle_range}°, step: {angle_step}°)...")
        for angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
            rotated_thresh = scipy.ndimage.rotate(thresh, angle, reshape=False, order=0, cval=0)
            score = calculate_projection_profile_score(rotated_thresh / 255.0)
            if score > max_score:
                max_score = score
                best_angle = angle
        if abs(best_angle) < 0.1:
            print(f"[base_image_utils] Projection profile: Negligible skew ({best_angle:.2f}°). No correction.")
            return img_pil
        print(f"[base_image_utils] Projection profile: Best skew angle: {best_angle:.2f}° with score {max_score:.2f}.")
        final_rotated_pil = img_pil.rotate(-best_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor='white')
        return final_rotated_pil
    except Exception as e:
        print(f"[base_image_utils] Error during projection profile deskew: {e}")
        import traceback
        traceback.print_exc()
        return img_pil

def auto_orient_image_opencv(img_pil_input: Image.Image) -> Image.Image:
    """
    画像の向きを自動補正する（横書きテキスト前提）。
    1. EXIFベースの補正
    2. Tesseract OCRによる90度単位の向き推定と回転 (オプション)
    3. 射影プロファイルによるスキュー（細かい傾き）補正
    """
    print(f"[base_image_utils] Starting auto_orient_image_opencv for image size: {img_pil_input.size}")
    current_img_pil = img_pil_input.copy()
    # 1. EXIFベースの補正
    try:
        current_img_pil = correct_image_orientation(current_img_pil)
        print(f"[base_image_utils] After EXIF correction, size: {current_img_pil.size}")
    except Exception as e_exif:
        print(f"[base_image_utils] Error during EXIF correction: {e_exif}")
    
    # 2. Tesseract OCRによる90度単位の向き推定と回転
    if PYTESSERACT_AVAILABLE: # インポートしたフラグを使用
        try:
            best_90_degree_rotation = get_text_orientation_with_tesseract(current_img_pil) # インポートした関数を使用
            if best_90_degree_rotation != 0:
                print(f"[base_image_utils] Applying {best_90_degree_rotation}-degree rotation based on OCR.")
                current_img_pil = current_img_pil.rotate(best_90_degree_rotation, expand=True, fillcolor='white')
                print(f"[base_image_utils] After {best_90_degree_rotation}-degree rotation, size: {current_img_pil.size}")
        except Exception as e_ocr_rot:
            # エラーメッセージに呼び出し元が ocr_processors であることを示唆する情報を追加しても良い
            print(f"[base_image_utils] Error during OCR-based 90-degree rotation (call to ocr_processors): {e_ocr_rot}")
    else:
        print("[base_image_utils] Pytesseract (from ocr_processors) not available, skipping OCR-based 90-degree orientation. Manual 90-degree rotation might be needed if image is sideways.")
    
    # 3. スキュー（細かい傾き）補正
    try:
        print("[base_image_utils] Attempting deskew with projection profile method...")
        # deskew_image_projection_profile は base_image_utils.py に残っているので、そのまま呼び出し
        current_img_pil = deskew_image_projection_profile(current_img_pil.copy(), angle_range=5.0, angle_step=0.25)
        print(f"[base_image_utils] After final deskew, size: {current_img_pil.size}")
    except Exception as e_deskew:
        print(f"[base_image_utils] Error during final deskew: {e_deskew}")
    
    return current_img_pil