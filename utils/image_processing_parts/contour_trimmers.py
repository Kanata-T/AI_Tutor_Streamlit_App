import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple, List, Any

from .base_image_utils import cv_image_to_bytes, image_to_bytes
from .opencv_pipeline_utils import (
    convert_pil_to_cv_gray,
    apply_gaussian_blur,
    apply_adaptive_threshold,
    apply_morphological_operation,
)

DEFAULT_JPEG_QUALITY_FOR_DEBUG = 85  # デバッグ画像保存時のデフォルトJPEG品質

def trim_image_by_contours(
    img_pil_input: Image.Image,
    trim_params: Dict[str, Any],
    jpeg_quality_for_debug_bytes: int = DEFAULT_JPEG_QUALITY_FOR_DEBUG,
    debug_image_collector: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """OpenCVを使用して輪郭検出し、画像をトリミングします。

    主な処理フロー:
    1.  入力Pillow画像をOpenCVグレースケール画像に変換します。
    2.  ガウシアンブラーを適用します（オプション）。
    3.  適応的閾値処理を適用し、二値化画像を得ます。
    4.  モルフォロジー演算（オープニング、クロージング）を適用します（オプション）。
    5.  処理後の二値化画像から輪郭を検出します。
    6.  検出された輪郭を面積に基づいてフィルタリングします。
    7.  有効な輪郭群を包含する最小のバウンディングボックスを計算します。
    8.  計算されたバウンディングボックスにパディングを適用し、元のPillow画像と
        二値化画像を切り出します。

    Args:
        img_pil_input: トリミング対象のPillow画像。
        trim_params: トリミング処理のパラメータを含む辞書。
            以下のキーを期待します:
            - padding (int): 切り出し領域の周囲に追加するパディング（ピクセル単位）。
            - adaptive_thresh_block_size (int): 適応的閾値処理のブロックサイズ。
            - adaptive_thresh_c (int): 適応的閾値処理のC値。
            - min_contour_area_ratio (float): 画像総面積に対する最小輪郭面積の割合。
            - gaussian_blur_kernel (tuple[int, int]): ガウシアンブラーのカーネルサイズ。(0,0)で無効。
            - morph_open_apply (bool): オープニング処理を適用するか。
            - morph_open_kernel_size (int): オープニングのカーネルサイズ。
            - morph_open_iterations (int): オープニングの繰り返し回数。
            - morph_close_apply (bool): クロージング処理を適用するか。
            - morph_close_kernel_size (int): クロージングのカーネルサイズ。
            - morph_close_iterations (int): クロージングの繰り返し回数。
        jpeg_quality_for_debug_bytes: デバッグ画像（JPEG形式）の品質。
        debug_image_collector: デバッグ画像と情報を収集するためのリスト。

    Returns:
        トリミングされたメインのPillow画像と、OCR用のトリミングされた二値化Pillow画像のタプル。
        処理に失敗した場合は (None, None) または (元の画像, None) を返すことがあります。
    """
    if not isinstance(img_pil_input, Image.Image):
        print(f"[contour_trimmers] Error: 入力はPillow画像である必要があります。")
        return None, None
    if not isinstance(trim_params, dict):
        print(f"[contour_trimmers] Error: trim_paramsは辞書である必要があります。")
        return None, None

    # パラメータの取得と検証
    padding = int(trim_params.get("padding", 0))
    adaptive_thresh_block_size = int(trim_params.get("adaptive_thresh_block_size", 11))
    adaptive_thresh_c = int(trim_params.get("adaptive_thresh_c", 7))
    min_contour_area_ratio = float(trim_params.get("min_contour_area_ratio", 0.00005))
    
    gb_kernel_conf = trim_params.get("gaussian_blur_kernel", (0,0))
    if isinstance(gb_kernel_conf, list) and len(gb_kernel_conf) == 2:
        gaussian_blur_kernel = tuple(map(int, gb_kernel_conf))
    elif isinstance(gb_kernel_conf, tuple) and len(gb_kernel_conf) == 2 and all(isinstance(x, int) for x in gb_kernel_conf):
        gaussian_blur_kernel = gb_kernel_conf
    else:
        print(f"[contour_trimmers] Warning: 不正なgaussian_blur_kernel形式: {gb_kernel_conf}。ブラーなし(0,0)にフォールバックします。")
        gaussian_blur_kernel = (0,0)

    morph_open_apply = bool(trim_params.get("morph_open_apply", False))
    morph_open_kernel_size = int(trim_params.get("morph_open_kernel_size", 3))
    morph_open_iterations = int(trim_params.get("morph_open_iterations", 1))
    
    morph_close_apply = bool(trim_params.get("morph_close_apply", False))
    morph_close_kernel_size = int(trim_params.get("morph_close_kernel_size", 3))
    morph_close_iterations = int(trim_params.get("morph_close_iterations", 1))

    img_pil_original_for_crop = img_pil_input.copy()  # トリミング失敗時のフォールバック用
    final_cropped_pil_image: Optional[Image.Image] = None
    final_ocr_binary_image: Optional[Image.Image] = None

    try:
        # 1. PIL画像をOpenCVグレースケールに変換
        img_cv_gray = convert_pil_to_cv_gray(img_pil_input)
        if img_cv_gray is None:
            print("[contour_trimmers] Error: PIL画像からOpenCVグレースケール画像への変換に失敗しました。")
            return img_pil_original_for_crop, None # 元の画像を返し、OCR用はNone
        original_height, original_width = img_cv_gray.shape[:2]
        
        if debug_image_collector is not None:
            debug_data = cv_image_to_bytes(img_cv_gray.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({
                    "label": "Trim - 0. グレースケール入力",
                    "data": debug_data,
                    "mode": "L",
                    "size": (original_width, original_height)
                })

        # 2. ガウシアンブラー
        img_blurred = apply_gaussian_blur(img_cv_gray, gaussian_blur_kernel)
        if img_blurred is None:  # ブラー処理失敗時は元画像を使用
            img_blurred = img_cv_gray.copy()
        if debug_image_collector is not None and gaussian_blur_kernel != (0,0): # ブラー適用時のみデバッグ画像追加
            debug_data = cv_image_to_bytes(img_blurred.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({
                    "label": f"Trim - 1. ブラー適用 (カーネル: {gaussian_blur_kernel})",
                    "data": debug_data,
                    "mode": "L",
                    "size": (original_width, original_height)
                })

        # 3. 適応的閾値処理 (二値化)
        img_thresh = apply_adaptive_threshold(img_blurred, adaptive_thresh_block_size, adaptive_thresh_c)
        if img_thresh is None:
            print("[contour_trimmers] Error: 適応的閾値処理に失敗しました。")
            return img_pil_original_for_crop, None
        if debug_image_collector is not None:
            debug_data = cv_image_to_bytes(img_thresh.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({
                    "label": f"Trim - 2. 適応的閾値処理 (ブロック: {adaptive_thresh_block_size}, C: {adaptive_thresh_c})",
                    "data": debug_data,
                    "mode": "L",
                    "size": (original_width, original_height)
                })
        
        # 4. モルフォロジー演算
        img_morphed = img_thresh.copy()  # 二値化画像をベースに処理
        if morph_open_apply:
            opened_img = apply_morphological_operation(img_morphed, "open", morph_open_kernel_size, morph_open_iterations)
            if opened_img is not None: 
                img_morphed = opened_img
                if debug_image_collector is not None:
                    debug_data = cv_image_to_bytes(img_morphed.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
                    if debug_data:
                        debug_image_collector.append({
                            "label": f"Trim - モルフォロジー: オープニング (カーネル: {morph_open_kernel_size}, 回数: {morph_open_iterations})",
                            "data": debug_data, "mode": "L", "size": (original_width, original_height)
                        })
        if morph_close_apply:
            closed_img = apply_morphological_operation(img_morphed, "close", morph_close_kernel_size, morph_close_iterations)
            if closed_img is not None:
                img_morphed = closed_img
                if debug_image_collector is not None:
                    debug_data = cv_image_to_bytes(img_morphed.copy(), jpeg_quality=jpeg_quality_for_debug_bytes)
                    if debug_data:
                        debug_image_collector.append({
                            "label": f"Trim - モルフォロジー: クロージング (カーネル: {morph_close_kernel_size}, 回数: {morph_close_iterations})",
                            "data": debug_data, "mode": "L", "size": (original_width, original_height)
                        })
        
        contours_input_image = img_morphed  # 輪郭検出の入力はこのモルフォロジー処理後の画像

        # 5. 輪郭検出
        contours, _ = cv2.findContours(contours_input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[contour_trimmers] 輪郭が検出されませんでした。")
            return img_pil_original_for_crop, None
        if debug_image_collector is not None:
            img_all_contours_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_all_contours_drawn, contours, -1, (0,0,255), 1) # 赤色で全輪郭を描画
            debug_data = cv_image_to_bytes(img_all_contours_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({
                    "label": "Trim - 3a. 検出された全ての輪郭",
                    "data": debug_data, "mode": "RGB", "size": (original_width, original_height)
                })

        # 6. 有効な輪郭の選択とフィルタリング (面積ベースのみ)
        min_absolute_area = original_width * original_height * min_contour_area_ratio
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_absolute_area]
        
        if not valid_contours:
            print("[contour_trimmers] 面積フィルタリング後、有効な輪郭がありませんでした。")
            return img_pil_original_for_crop, None
        if debug_image_collector is not None:
            img_valid_contours_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_valid_contours_drawn, valid_contours, -1, (0,255,0), 1) # 緑色で有効な輪郭を描画
            debug_data = cv_image_to_bytes(img_valid_contours_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({
                    "label": "Trim - 3b. 面積フィルタリング後の有効な輪郭",
                    "data": debug_data, "mode": "RGB", "size": (original_width, original_height)
                })

        # 7. 有効な輪郭全体を包含するBoundingBoxを計算
        x_coords = []
        y_coords = []
        for contour in valid_contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            x_coords.extend([x_c, x_c + w_c])
            y_coords.extend([y_c, y_c + h_c])
        
        if not x_coords or not y_coords: # 念のためのチェック
             print("[contour_trimmers] 有効な輪郭からバウンディングボックス座標が取得できませんでした。")
             return img_pil_original_for_crop, None

        x_bbox = min(x_coords)
        y_bbox = min(y_coords)
        w_bbox = max(x_coords) - x_bbox
        h_bbox = max(y_coords) - y_bbox

        if debug_image_collector is not None:
            img_bbox_drawn = cv2.cvtColor(img_cv_gray.copy(), cv2.COLOR_GRAY2BGR)
            # 青色で計算されたバウンディングボックスを描画
            cv2.rectangle(img_bbox_drawn, (x_bbox,y_bbox), (x_bbox+w_bbox,y_bbox+h_bbox), (255,0,0), 2)
            debug_data = cv_image_to_bytes(img_bbox_drawn, jpeg_quality=jpeg_quality_for_debug_bytes)
            if debug_data:
                debug_image_collector.append({
                    "label": "Trim - 3c. 有効輪郭群の結合バウンディングボックス",
                    "data": debug_data, "mode": "RGB", "size": (original_width, original_height)
                })

        final_x, final_y, final_w, final_h = x_bbox, y_bbox, w_bbox, h_bbox

        # 8. パディングと最終的な切り出し
        crop_x1 = max(0, final_x - padding)
        crop_y1 = max(0, final_y - padding)
        crop_x2 = min(original_width, final_x + final_w + padding)
        crop_y2 = min(original_height, final_y + final_h + padding)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1: # 切り出し領域が無効な場合
            print(f"[contour_trimmers] Warning: パディング適用後の切り出し領域が無効です ({crop_x1},{crop_y1},{crop_x2},{crop_y2})。トリミングをスキップします。")
            return img_pil_original_for_crop, None

        # 元のPillow画像 (向き補正済み、カラースケールも元のまま) から切り出す
        final_cropped_pil_image = img_pil_original_for_crop.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        if debug_image_collector is not None and final_cropped_pil_image:
            fmt_for_debug = "PNG" if final_cropped_pil_image.mode != "RGB" else "JPEG"
            debug_data = image_to_bytes(final_cropped_pil_image.copy(), target_format=fmt_for_debug, jpeg_quality=jpeg_quality_for_debug_bytes) # type: ignore
            if debug_data:
                debug_image_collector.append({
                    "label": "Trim - 4a. 切り出し後 (メイン画像)",
                    "data": debug_data,
                    "mode": final_cropped_pil_image.mode,
                    "size": final_cropped_pil_image.size
                })
        
        # OCR用の二値化画像を、輪郭検出に使用した二値化画像 (contours_input_image) から切り出す
        cropped_binary_cv = contours_input_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped_binary_cv.size > 0: # 切り出した二値化画像が空でないことを確認
            final_ocr_binary_image = Image.fromarray(cropped_binary_cv, mode='L')
            if debug_image_collector is not None and final_ocr_binary_image:
                debug_data = image_to_bytes(final_ocr_binary_image.copy(), target_format="PNG") # OCR用はPNG推奨
                if debug_data:
                    debug_image_collector.append({
                        "label": "Trim - 4b. 切り出し後 (OCR用二値化画像)",
                        "data": debug_data,
                        "mode": "L",
                        "size": final_ocr_binary_image.size
                    })
        else:
            print("[contour_trimmers] Warning: OCR用の切り出し二値化画像が空になりました。メインの切り出し画像をグレースケール化してフォールバックします。")
            if final_cropped_pil_image: # メインの切り出し画像が存在すれば、それをグレースケール化
                final_ocr_binary_image = final_cropped_pil_image.convert('L')

        return final_cropped_pil_image, final_ocr_binary_image

    except Exception as e:
        print(f"[contour_trimmers] 輪郭ベースのトリミング処理中に致命的なエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        # エラー時は、トリミングされていない元のPillow画像とNoneを返す
        return img_pil_original_for_crop, None