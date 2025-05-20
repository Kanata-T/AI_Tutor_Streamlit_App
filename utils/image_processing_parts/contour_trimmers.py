# utils/image_processing_parts/contour_trimmers.py
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple, List, Any

# 分割したモジュールからインポート
from .base_image_utils import cv_image_to_bytes, image_to_bytes # デバッグ画像保存用
from .opencv_pipeline_utils import (
    convert_pil_to_cv_gray,
    apply_gaussian_blur,
    apply_adaptive_threshold,
    apply_morphological_operation,
)
from .region_analysis_utils import (
    calculate_haar_like_vertical_response,
    detect_vertical_peaks,
    get_horizontal_projection_bounds,
)

DEFAULT_JPEG_QUALITY_FOR_DEBUG = 85 # デバッグ画像保存時のデフォルトJPEG品質

def trim_image_by_contours(
    img_pil_input: Image.Image,
    trim_params: Dict[str, Any],
    jpeg_quality_for_debug_bytes: int = DEFAULT_JPEG_QUALITY_FOR_DEBUG,
    debug_image_collector: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    OpenCVを使用して輪郭検出し、画像をトリミングする。
    Haar-Like特徴量と水平射影による補助も利用可能。
    戻り値: (トリミングされたPillow画像, トリミングされた二値化Pillow画像 (OCR用))
    """
    if not isinstance(img_pil_input, Image.Image):
        print(f"[contour_trimmers] Error: Input must be a PIL Image.")
        return None, None # 早期リターン
    if not isinstance(trim_params, dict):
        print(f"[contour_trimmers] Error: trim_params must be a dictionary.")
        return None, None # 早期リターン

    # --- パラメータの取得と検証 ---
    # 必須ではないが、存在を期待するキーのデフォルト値を設定
    padding = int(trim_params.get("padding", 0))
    adaptive_thresh_block_size = int(trim_params.get("adaptive_thresh_block_size", 11))
    adaptive_thresh_c = int(trim_params.get("adaptive_thresh_c", 7))
    min_contour_area_ratio = float(trim_params.get("min_contour_area_ratio", 0.00005))
    
    gb_kernel_conf = trim_params.get("gaussian_blur_kernel", (0,0))
    if isinstance(gb_kernel_conf, list) and len(gb_kernel_conf) == 2:
        gaussian_blur_kernel = tuple(map(int, gb_kernel_conf))
    elif isinstance(gb_kernel_conf, tuple) and len(gb_kernel_conf) == 2 and all(isinstance(x, int) for x in gb_kernel_conf):
        gaussian_blur_kernel = gb_kernel_conf
    else: # 不正な形式や値の場合はブラーなしにする
        print(f"[contour_trimmers] Warning: Invalid gaussian_blur_kernel format: {gb_kernel_conf}. Defaulting to no blur (0,0).")
        gaussian_blur_kernel = (0,0)

    # haar_apply = bool(trim_params.get("haar_apply", False))
    # haar_rect_h = int(trim_params.get("haar_rect_h", 20))
    # haar_peak_threshold = float(trim_params.get("haar_peak_threshold", 10.0))
    
    # h_proj_apply = bool(trim_params.get("h_proj_apply", False))
    # h_proj_threshold_ratio = float(trim_params.get("h_proj_threshold_ratio", 0.01))
    
    morph_open_apply = bool(trim_params.get("morph_open_apply", False))
    morph_open_kernel_size = int(trim_params.get("morph_open_kernel_size", 3))
    morph_open_iterations = int(trim_params.get("morph_open_iterations", 1))
    
    morph_close_apply = bool(trim_params.get("morph_close_apply", False))
    morph_close_kernel_size = int(trim_params.get("morph_close_kernel_size", 3))
    morph_close_iterations = int(trim_params.get("morph_close_iterations", 1))

    img_pil_original_for_crop = img_pil_input.copy() # トリミング失敗時のフォールバック用
    final_cropped_pil_image: Optional[Image.Image] = None
    final_ocr_binary_image: Optional[Image.Image] = None

    try:
        # 1. PIL画像をOpenCVグレースケールに変換
        img_cv_gray = convert_pil_to_cv_gray(img_pil_input)
        if img_cv_gray is None:
            print("[contour_trimmers] Error: Failed to convert PIL to CV Gray.")
            return img_pil_original_for_crop, None
        original_height, original_width = img_cv_gray.shape[:2]
        
        if debug_image_collector is not None:
            debug_data = cv_image_to_bytes(img_cv_gray.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({"label": "Trim - 0. Gray Input", "data": debug_data, "mode": "L", "size": (original_width, original_height)})

        # --- ▼ Haar-Like特徴量によるY軸範囲推定のロジックをコメントアウト ▼ ---
        # y_min_haar_final, y_max_haar_final = 0, original_height # デフォルトは画像全体
        # if haar_apply:
        #     haar_response = calculate_haar_like_vertical_response(img_cv_gray, haar_rect_h)
        #     if haar_response is not None and len(haar_response) > 0:
        #         p_start, p_end = detect_vertical_peaks(haar_response, haar_peak_threshold)
        #         if len(p_start) > 0 and len(p_end) > 0:
        #             temp_y_min = np.min(p_start)
        #             valid_ends_after_min_start = p_end[p_end > temp_y_min]
        #             if len(valid_ends_after_min_start) > 0:
        #                 y_min_haar_final = max(0, temp_y_min) 
        #                 y_max_haar_final = min(original_height, np.max(valid_ends_after_min_start) + haar_rect_h)
        #                 if debug_image_collector is not None:
        #                     viz_h = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
        #                     cv2.rectangle(viz_h, (0, y_min_haar_final), (original_width - 1, y_max_haar_final), (0, 255, 255), 2) # Cyan
        #                     debug_data = cv_image_to_bytes(viz_h, jpeg_quality=jpeg_quality_for_debug_bytes)
        #                     if debug_data:
        #                         debug_image_collector.append({"label": "Trim - A1. Haar Y-Range", "data": debug_data, "mode": "RGB", "size": (original_width, original_height)})

        # --- 2. ガウシアンブラー ---
        img_blurred = apply_gaussian_blur(img_cv_gray, gaussian_blur_kernel)
        if img_blurred is None: img_blurred = img_cv_gray.copy() # 失敗時は元画像
        if debug_image_collector is not None and gaussian_blur_kernel != (0,0): # 適用時のみ
            debug_data = cv_image_to_bytes(img_blurred.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({"label": f"Trim - 1. Blur(K{gaussian_blur_kernel})", "data": debug_data, "mode": "L", "size": (original_width, original_height)})

        # --- 3. 適応的閾値処理 (二値化) ---
        img_thresh = apply_adaptive_threshold(img_blurred, adaptive_thresh_block_size, adaptive_thresh_c)
        if img_thresh is None:
            print("[contour_trimmers] Error: Adaptive thresholding failed.")
            return img_pil_original_for_crop, None
        if debug_image_collector is not None:
            debug_data = cv_image_to_bytes(img_thresh.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({"label": f"Trim - 2. Threshold(B{adaptive_thresh_block_size},C{adaptive_thresh_c})", "data": debug_data, "mode": "L", "size": (original_width, original_height)})
        
        # --- ▼ 水平射影によるX軸範囲推定のロジックをコメントアウト ▼ ---
        # x_min_hproj_final, x_max_hproj_final = 0, original_width # デフォルトは画像全体
        # if h_proj_apply:
        #     proj_res = get_horizontal_projection_bounds(img_thresh, h_proj_threshold_ratio)
        #     if proj_res:
        #         x_min_hproj_final, x_max_hproj_final = proj_res
        #         if debug_image_collector is not None:
        #             viz_hp = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR) # グレースケールからBGRに
        #             cv2.rectangle(viz_hp, (x_min_hproj_final, 0), (x_max_hproj_final, original_height - 1), (255, 255, 0), 2) # Yellow
        #             debug_data = cv_image_to_bytes(viz_hp, jpeg_quality=jpeg_quality_for_debug_bytes)
        #             if debug_data:
        #                 debug_image_collector.append({"label": "Trim - A2. H-Proj X-Range", "data": debug_data, "mode": "RGB", "size": (original_width, original_height)})

        # --- 4. モルフォロジー演算 ---
        img_morphed = img_thresh.copy() # 二値化画像をベースに
        if morph_open_apply:
            opened_img = apply_morphological_operation(img_morphed, "open", morph_open_kernel_size, morph_open_iterations)
            if opened_img is not None: 
                img_morphed = opened_img
                if debug_image_collector is not None:
                    debug_data = cv_image_to_bytes(img_morphed.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
                    if debug_data:
                        debug_image_collector.append({"label": f"Trim - MorphOPEN(K{morph_open_kernel_size},I{morph_open_iterations})", "data": debug_data, "mode": "L", "size": (original_width, original_height)})
        if morph_close_apply:
            closed_img = apply_morphological_operation(img_morphed, "close", morph_close_kernel_size, morph_close_iterations)
            if closed_img is not None:
                img_morphed = closed_img
                if debug_image_collector is not None:
                    debug_data = cv_image_to_bytes(img_morphed.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
                    if debug_data:
                        debug_image_collector.append({"label": f"Trim - MorphCLOSE(K{morph_close_kernel_size},I{morph_close_iterations})", "data": debug_data, "mode": "L", "size": (original_width, original_height)})
        
        contours_input_image = img_morphed # 輪郭検出の入力はこのモルフォロジー処理後の画像

        # --- 5. 輪郭検出 ---
        contours, _ = cv2.findContours(contours_input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[contour_trimmers] No contours found.")
            return img_pil_original_for_crop, None
        if debug_image_collector is not None:
            img_all_contours_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_all_contours_drawn, contours, -1, (0,0,255), 1) # Red
            debug_data = cv_image_to_bytes(img_all_contours_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({"label": "Trim - 3a. All Contours", "data": debug_data, "mode": "RGB", "size": (original_width, original_height)})

        # --- 6. 有効な輪郭の選択とフィルタリング ---
        min_absolute_area = original_width * original_height * min_contour_area_ratio
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_absolute_area:
                # --- ▼ 空間フィルタのロジックをコメントアウト ▼ ---
                # if haar_apply or h_proj_apply:
                #     x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                #     contour_center_x, contour_center_y = x_c + w_c // 2, y_c + h_c // 2
                #     in_haar_y_range = not haar_apply or (y_min_haar_final <= contour_center_y <= y_max_haar_final)
                #     in_hproj_x_range = not h_proj_apply or (x_min_hproj_final <= contour_center_x <= x_max_hproj_final)
                #     if in_haar_y_range and in_hproj_x_range:
                #         valid_contours.append(contour)
                # else: # 空間フィルタなし
                #     valid_contours.append(contour)
                # --- ▲ ここまで ▲ ---
                valid_contours.append(contour) # 常に有効な輪郭として追加 (面積フィルタのみ)
        
        if not valid_contours:
            print("[contour_trimmers] No valid contours after filtering.")
            return img_pil_original_for_crop, None
        if debug_image_collector is not None:
            img_valid_contours_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_valid_contours_drawn, valid_contours, -1, (0,255,0), 1) # Green
            debug_data = cv_image_to_bytes(img_valid_contours_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({"label": "Trim - 3b. Valid Contours", "data": debug_data, "mode": "RGB", "size": (original_width, original_height)})

        # --- 7. 有効な輪郭全体を包含するBoundingBoxを計算 ---
        # (元のコードの merged_x_min, ... のロジックをboundingRect(np.concatenate(...))で代替)
        # all_points = np.concatenate([c for c in valid_contours]) # 全ての輪郭の点を結合
        # x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(all_points)
        # より安全なのは、各輪郭のBBoxを計算し、それらの最小/最大を取る方法（元のコードに近い）
        x_coords = []
        y_coords = []
        for contour in valid_contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            x_coords.extend([x_c, x_c + w_c])
            y_coords.extend([y_c, y_c + h_c])
        
        if not x_coords or not y_coords: # 念のため
             return img_pil_original_for_crop, None

        x_bbox = min(x_coords)
        y_bbox = min(y_coords)
        w_bbox = max(x_coords) - x_bbox
        h_bbox = max(y_coords) - y_bbox

        if debug_image_collector is not None:
            img_bbox_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img_bbox_drawn, (x_bbox,y_bbox), (x_bbox+w_bbox,y_bbox+h_bbox), (255,0,0), 2) # Blue
            debug_data = cv_image_to_bytes(img_bbox_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({"label": "Trim - 3c. Initial BBox", "data": debug_data, "mode": "RGB", "size": (original_width, original_height)})

        # --- ▼ BoundingBoxの補正 (Haar-Likeと水平射影) のロジックをコメントアウト ▼ ---
        # final_x, final_y, final_w, final_h = x_bbox, y_bbox, w_bbox, h_bbox
        # corrected_by_features = False
        # if haar_apply and y_max_haar_final > y_min_haar_final: # Haarの結果が有効なら
        #     new_y_min = min(final_y, y_min_haar_final)
        #     new_y_max = max(final_y + final_h, y_max_haar_final)
        #     final_y = new_y_min
        #     final_h = new_y_max - final_y
        #     corrected_by_features = True
        # if h_proj_apply and x_max_hproj_final > x_min_hproj_final: # 水平射影の結果が有効なら
        #     new_x_min = min(final_x, x_min_hproj_final)
        #     new_x_max = max(final_x + final_w, x_max_hproj_final)
        #     final_x = new_x_min
        #     final_w = new_x_max - final_x
        #     corrected_by_features = True
        # if debug_image_collector is not None and corrected_by_features:
        #     img_corrected_bbox_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
        #     cv2.rectangle(img_corrected_bbox_drawn, (x_bbox,y_bbox), (x_bbox+w_bbox,y_bbox+h_bbox), (255,0,0), 3) # 元のBBox (Blue)
        #     cv2.rectangle(img_corrected_bbox_drawn, (final_x,final_y), (final_x+final_w,final_y+final_h), (0,165,255), 2) # 補正後BBox (Orange)
        #     debug_data = cv_image_to_bytes(img_corrected_bbox_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
        #     if debug_data:
        #         debug_image_collector.append({"label": "Trim - 3d. Corrected BBox", "data": debug_data, "mode":"RGB", "size":(original_width,original_height)})
        # --- ▲ ここまで ▲ ---
        # ↓↓↓ 補正を行わないので、元のBBoxをそのまま使う ↓↓↓
        final_x, final_y, final_w, final_h = x_bbox, y_bbox, w_bbox, h_bbox

        # --- 8. パディングと最終的な切り出し ---
        crop_x1 = max(0, final_x - padding)
        crop_y1 = max(0, final_y - padding)
        crop_x2 = min(original_width, final_x + final_w + padding)
        crop_y2 = min(original_height, final_y + final_h + padding)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            print(f"[contour_trimmers] Warning: Invalid crop region after padding/correction ({crop_x1},{crop_y1},{crop_x2},{crop_y2}). Skipping trim.")
            return img_pil_original_for_crop, None

        # 元のPillow画像 (向き補正済み) から切り出す
        final_cropped_pil_image = img_pil_original_for_crop.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        if debug_image_collector is not None and final_cropped_pil_image:
            # ここで final_cropped_pil_image.mode を見て適切なフォーマットで保存する
            # 例: JPEGかPNGか、あるいはそのままのモードで
            fmt_for_debug = "PNG" # 安全のためPNG
            if final_cropped_pil_image.mode == "RGB": fmt_for_debug = "JPEG"
            
            debug_data = image_to_bytes(final_cropped_pil_image.copy(), target_format=fmt_for_debug, jpeg_quality=jpeg_quality_for_debug_bytes) # type: ignore
            if debug_data:
                debug_image_collector.append({"label": "Trim - 4a. Cropped (Original)", "data": debug_data, "mode": final_cropped_pil_image.mode, "size": final_cropped_pil_image.size})
        
        # OCR用の二値化画像を、輪郭検出に使った二値化画像 (contours_input_image) から切り出す
        cropped_binary_cv = contours_input_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped_binary_cv.size > 0:
            final_ocr_binary_image = Image.fromarray(cropped_binary_cv, mode='L')
            if debug_image_collector is not None and final_ocr_binary_image:
                debug_data = image_to_bytes(final_ocr_binary_image.copy(), target_format="PNG") # OCR用はPNG
                if debug_data:
                    debug_image_collector.append({"label": "Trim - 4b. Cropped (Binary OCR)", "data": debug_data, "mode": "L", "size": final_ocr_binary_image.size})
        else:
            print("[contour_trimmers] Warning: Cropped binary for OCR is empty. Falling back to grayscale of cropped original.")
            if final_cropped_pil_image:
                final_ocr_binary_image = final_cropped_pil_image.convert('L') # フォールバック

        return final_cropped_pil_image, final_ocr_binary_image

    except Exception as e:
        print(f"[contour_trimmers] Critical error during contour trimming process: {e}")
        import traceback
        traceback.print_exc()
        # エラー時は、トリミングされていない元のPillow画像とNoneを返す
        return img_pil_original_for_crop, None