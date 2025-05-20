# utils/image_processing_parts/region_analysis_utils.py
import numpy as np
from scipy.signal import argrelmax # type: ignore # scipy.signal.argrelmax の型ヒントがない場合がある
from typing import Optional, Tuple

# === Haar-Like 行検出関数 ===
def calculate_haar_like_vertical_response(img_gray: np.ndarray, rect_height: int) -> Optional[np.ndarray]:
    """
    グレースケール画像に対して垂直方向のHaar-Like特徴量応答を計算する。
    (元の calc_haarlike_vertical から名前を少し変更)
    """
    if not isinstance(img_gray, np.ndarray) or img_gray.ndim != 2:
        print(f"[region_analysis_utils] Error: Haar-Like input must be a 2D numpy array (grayscale).")
        return None
    if not isinstance(rect_height, int) or rect_height <= 0 or rect_height % 2 != 0:
        print(f"[region_analysis_utils] Error: rect_height for Haar-Like must be a positive even integer, got {rect_height}.")
        return None

    pattern_half_height = rect_height // 2
    image_height = img_gray.shape[0]

    if image_height <= rect_height:
        print(f"[region_analysis_utils] Warning: Image height ({image_height}) is less than or equal to rect_height ({rect_height}). Skipping Haar-Like calculation.")
        return np.array([]) # 空の配列を返す

    # 応答を格納する配列
    response = np.zeros(image_height - rect_height)
    
    try:
        for i in range(image_height - rect_height):
            # 上部矩形 (S1) と下部矩形 (S2) のY座標範囲
            s1_start, s1_end = i, i + pattern_half_height
            s2_start, s2_end = i + pattern_half_height, i + rect_height
            
            # 画像境界チェックは不要 (ループ範囲で保証)
            mean_s1 = np.mean(img_gray[s1_start:s1_end, :])
            mean_s2 = np.mean(img_gray[s2_start:s2_end, :])
            response[i] = mean_s1 - mean_s2
    except Exception as e:
        print(f"[region_analysis_utils] Error during Haar-Like response calculation: {e}")
        return None
    return response

def detect_vertical_peaks(data: np.ndarray, peak_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    データ配列から指定された閾値を超える正のピーク（開始点候補）と
    絶対値が閾値を超える負のピーク（終了点候補）を検出する。
    (元の peak_detection_vertical から名前を少し変更)
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1 or len(data) == 0:
        return np.array([]), np.array([]) # 不正な入力は空のタプルを返す
    if not isinstance(peak_threshold, (int, float)):
         return np.array([]), np.array([])

    try:
        # 正のピーク (開始点候補)
        positive_peaks_indices = argrelmax(data)[0]
        valid_start_peaks = positive_peaks_indices[np.where(data[positive_peaks_indices] > peak_threshold)[0]]

        # 負のピーク (終了点候補) - データ符号を反転して正のピークとして検出
        negative_peaks_indices = argrelmax(-data)[0]
        # 閾値は絶対値で比較するため、ここでは peak_threshold をそのまま使う
        # (元のコードでは np.abs(data[peaks_end]) > threshold だったが、-data のピークは data の谷なので、
        # -data[negative_peaks_indices] が正の値になる。これが peak_threshold より大きいかを見る)
        # または、元の data[negative_peaks_indices] が -peak_threshold より小さいかを見る
        valid_end_peaks = negative_peaks_indices[np.where(data[negative_peaks_indices] < -peak_threshold)[0]]
        # 元のコードの挙動に合わせるなら:
        # valid_end_peaks = negative_peaks_indices[np.where(np.abs(data[negative_peaks_indices]) > peak_threshold)[0]]


    except Exception as e:
        print(f"[region_analysis_utils] Error during peak detection: {e}")
        return np.array([]), np.array([])
        
    return valid_start_peaks, valid_end_peaks

# === 水平射影プロファイル関数 ===
def get_horizontal_projection_bounds(
    binary_image: np.ndarray, 
    threshold_ratio: float = 0.01
) -> Optional[Tuple[int, int]]:
    """
    二値化画像の水平射影プロファイルから、テキスト領域の左右の境界を推定する。
    (元の get_horizontal_projection から名前を少し変更)
    """
    if not isinstance(binary_image, np.ndarray) or binary_image.ndim != 2:
        print(f"[region_analysis_utils] Error: Horizontal projection input must be a 2D numpy array (binary).")
        return None
    if not (0 < threshold_ratio < 1):
        print(f"[region_analysis_utils] Error: threshold_ratio must be between 0 and 1, got {threshold_ratio}.")
        return None # 不正な閾値

    try:
        # 画像が白黒反転している場合（テキストが0、背景が255）を考慮
        # 一般的にはテキストが255（白）、背景が0（黒）を期待する
        # ここでは入力が「テキストが前景ピクセル値」であることを前提とする
        # (例: OpenCVのadaptiveThresholdでTHRESH_BINARY_INVを使った結果など)
        projection = np.sum(binary_image, axis=0) # 列ごとのピクセル和 (垂直方向の射影)
        
        if projection.max() == 0: # 画像が真っ黒、またはテキストがない
            return None

        threshold = projection.max() * threshold_ratio
        text_columns_indices = np.where(projection > threshold)[0]

        if len(text_columns_indices) == 0: # 閾値を超える列がない
            return None
            
        min_col = text_columns_indices.min()
        max_col = text_columns_indices.max()
        return int(min_col), int(max_col)

    except Exception as e:
        print(f"[region_analysis_utils] Error in get_horizontal_projection_bounds: {e}")
        return None