# utils/image_processing_parts/opencv_pipeline_utils.py
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Literal

def convert_pil_to_cv_gray(img_pil: Image.Image) -> Optional[np.ndarray]:
    """Pillow画像をOpenCVのグレースケール画像 (numpy.ndarray) に変換する。"""
    if not isinstance(img_pil, Image.Image):
        print(f"[opencv_pipeline_utils] Error: Expected PIL Image, got {type(img_pil)}.")
        return None
    try:
        # Pillow画像がRGBモードでない場合でも、一度RGBに変換してからグレースケール化
        if img_pil.mode == 'L': # 既にグレースケールの場合
            return np.array(img_pil)
        elif img_pil.mode == 'RGBA': # アルファチャンネルがある場合
            # OpenCVはBGRAを扱うが、ここでは単純にRGBに変換してグレースケール化
            img_rgb_pil = Image.new("RGB", img_pil.size, (255, 255, 255))
            img_rgb_pil.paste(img_pil, mask=img_pil.split()[-1]) # type: ignore
            return cv2.cvtColor(np.array(img_rgb_pil), cv2.COLOR_RGB2GRAY)
        else: # その他のモード (Pなど)
            return cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"[opencv_pipeline_utils] Error converting PIL to CV Gray: {e}")
        return None

def apply_gaussian_blur(img_cv_gray: np.ndarray, kernel_size_tuple: Tuple[int, int]) -> Optional[np.ndarray]:
    """OpenCVのグレースケール画像にガウシアンブラーを適用する。"""
    if not isinstance(img_cv_gray, np.ndarray) or img_cv_gray.ndim != 2:
        print(f"[opencv_pipeline_utils] Error: Blur input must be a 2D numpy array (grayscale).")
        return None
    if not (isinstance(kernel_size_tuple, tuple) and len(kernel_size_tuple) == 2 and
            all(isinstance(k, int) for k in kernel_size_tuple)):
        print(f"[opencv_pipeline_utils] Error: kernel_size_tuple must be a tuple of two integers.")
        return None

    k_width, k_height = kernel_size_tuple
    if k_width > 0 and k_height > 0 and k_width % 2 != 0 and k_height % 2 != 0:
        try:
            return cv2.GaussianBlur(img_cv_gray, (k_width, k_height), 0)
        except Exception as e:
            print(f"[opencv_pipeline_utils] Error applying Gaussian Blur: {e}")
            return None
    elif k_width == 0 or k_height == 0: # ブラーを適用しない場合
        return img_cv_gray.copy() # 元の画像をコピーして返す
    else:
        print(f"[opencv_pipeline_utils] Warning: Invalid kernel size for Gaussian Blur ({k_width}, {k_height}). Must be positive odd numbers, or 0 to skip. Returning original.")
        return img_cv_gray.copy()


def apply_adaptive_threshold(
    img_cv_blurred_or_gray: np.ndarray, 
    block_size: int, 
    c_value: int,
    threshold_type: int = cv2.THRESH_BINARY_INV, # テキストを白(255)にする
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
) -> Optional[np.ndarray]:
    """OpenCVの画像（ブラー適用後またはグレースケール）に適応的閾値処理を適用する。"""
    if not isinstance(img_cv_blurred_or_gray, np.ndarray) or img_cv_blurred_or_gray.ndim != 2:
        print(f"[opencv_pipeline_utils] Error: Adaptive threshold input must be a 2D numpy array.")
        return None
    
    # block_size は3以上の奇数である必要がある
    current_block_size = block_size
    if current_block_size < 3:
        current_block_size = 3
    if current_block_size % 2 == 0: # 偶数なら奇数に
        current_block_size += 1
        
    try:
        return cv2.adaptiveThreshold(
            img_cv_blurred_or_gray, 255, adaptive_method, threshold_type,
            current_block_size, c_value
        )
    except Exception as e:
        print(f"[opencv_pipeline_utils] Error applying adaptive threshold: {e}")
        return None

def apply_morphological_operation(
    img_cv_binary: np.ndarray,
    operation: Literal["open", "close"],
    kernel_size: int,
    iterations: int = 1
) -> Optional[np.ndarray]:
    """OpenCVの二値化画像にモルフォロジー演算（オープニングまたはクロージング）を適用する。"""
    if not isinstance(img_cv_binary, np.ndarray) or img_cv_binary.ndim != 2:
        print(f"[opencv_pipeline_utils] Error: Morphological operation input must be a 2D numpy array (binary).")
        return None
    if not (isinstance(kernel_size, int) and kernel_size > 0):
        print(f"[opencv_pipeline_utils] Error: kernel_size for morphology must be a positive integer.")
        return None
    if not (isinstance(iterations, int) and iterations >= 1):
        print(f"[opencv_pipeline_utils] Error: iterations for morphology must be a positive integer.")
        return None

    # カーネルサイズは奇数である必要がある
    k_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    kernel = np.ones((k_size, k_size), np.uint8)
    
    op_type = None
    if operation == "open":
        op_type = cv2.MORPH_OPEN
    elif operation == "close":
        op_type = cv2.MORPH_CLOSE
    else:
        print(f"[opencv_pipeline_utils] Error: Unsupported morphological operation '{operation}'. Choose 'open' or 'close'.")
        return None
        
    try:
        return cv2.morphologyEx(img_cv_binary, op_type, kernel, iterations=iterations)
    except Exception as e:
        print(f"[opencv_pipeline_utils] Error applying morphological {operation}: {e}")
        return None