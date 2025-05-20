# utils/image_processor.py
from PIL import Image # Image.MAX_IMAGE_PIXELS の設定のため
from io import BytesIO
from typing import Optional, Dict, Literal, Tuple, List, Any

# config_loader と分割した画像処理部品をインポート
from . import config_loader
from .image_processing_parts.base_image_utils import (
    correct_image_orientation,
    convert_to_rgb_with_alpha_handling,
    apply_grayscale_pillow,
    resize_image_pillow,
    image_to_bytes,
    cv_image_to_bytes  # ★ 追加 (ocr_processors から返されたデバッグCV画像をバイト化するため)
)
from .image_processing_parts.contour_trimmers import trim_image_by_contours
from .image_processing_parts.opencv_pipeline_utils import (
    convert_pil_to_cv_gray,
    apply_adaptive_threshold,
)

# --- 追加: OCRベーストリミング用ユーティリティのインポート ---
from .image_processing_parts.ocr_processors import (
    trim_image_by_ocr_text_bounds,
    PYTESSERACT_AVAILABLE as OCR_PROCESSORS_PYTESSERACT_AVAILABLE  # 別名でインポートして衝突を避ける
)

# --- 設定値読み込み ---
_full_config = config_loader.get_config()
IMG_PROC_CONFIG = _full_config.get("image_processing", {})

# Pillowの最大ピクセル数設定
pillow_max_pixels_cfg = IMG_PROC_CONFIG.get("pillow_max_image_pixels", 225000000) # デフォルト2.25億ピクセル
if pillow_max_pixels_cfg is not None and pillow_max_pixels_cfg > 0: # 0以下は無効とみなす
    try:
        Image.MAX_IMAGE_PIXELS = int(pillow_max_pixels_cfg)
    except (ValueError, TypeError):
        print(f"[image_processor] Warning: Invalid 'pillow_max_image_pixels' in config: {pillow_max_pixels_cfg}. Using Pillow default.")
        Image.MAX_IMAGE_PIXELS = None # Pillowの内部デフォルトに任せる
else: # Noneまたは0以下の場合
    Image.MAX_IMAGE_PIXELS = None # Pillowの内部デフォルトに任せる

# MIMEタイプ関連
SUPPORTED_MIME_TYPES = IMG_PROC_CONFIG.get("supported_mime_types", ["image/png", "image/jpeg", "image/webp"])
CONVERTABLE_MIME_TYPES = IMG_PROC_CONFIG.get("convertable_mime_types", {"image/gif": "image/png", "image/bmp": "image/png"})

# preprocess_uploaded_image 関数のデフォルト引数値
DEFAULT_MAX_PIXELS_RESIZE = IMG_PROC_CONFIG.get("default_max_pixels_for_resizing", 4000000) # 例: 2000x2000
DEFAULT_OUTPUT_FORMAT: Literal["JPEG", "PNG"] = IMG_PROC_CONFIG.get("default_output_format", "JPEG").upper() # type: ignore
DEFAULT_JPEG_QUALITY = IMG_PROC_CONFIG.get("default_jpeg_quality", 85)
DEFAULT_APPLY_GRAYSCALE = IMG_PROC_CONFIG.get("apply_grayscale", True)

# OpenCVトリミングのデフォルトパラメータ (config.yaml からロード)
# これは contour_trimmers.trim_image_by_contours に渡す trim_params の基礎となる
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG = IMG_PROC_CONFIG.get("opencv_trimming", {}).copy() # 必ずコピーして使う

# configにキーが存在しない場合のフォールバック値を設定 (setdefaultは辞書を直接変更する)
# contour_trimmers.py 側でも .get(key, default) でフォールバックするので、ここでは必須ではないが、
# config構造の明確化や、このモジュールレベルでのデフォルト値保証には役立つ。
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("apply", True)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("padding", 0)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("adaptive_thresh_block_size", 11)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("adaptive_thresh_c", 7)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("min_contour_area_ratio", 0.00005)

# gaussian_blur_kernel: contour_trimmers.py は (width, height) のタプルを期待。
# config.yaml が "gaussian_blur_kernel": [w,h] 形式か、
# "gaussian_blur_kernel_width" と "gaussian_blur_kernel_height" の個別キーを持つ場合に対応。
gb_kernel_cfg = DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("gaussian_blur_kernel")
if isinstance(gb_kernel_cfg, list) and len(gb_kernel_cfg) == 2:
    DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG["gaussian_blur_kernel"] = tuple(map(int, gb_kernel_cfg))
elif "gaussian_blur_kernel_width" in DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG and \
     "gaussian_blur_kernel_height" in DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG:
    DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG["gaussian_blur_kernel"] = (
        int(DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("gaussian_blur_kernel_width", 0)),
        int(DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("gaussian_blur_kernel_height", 0))
    )
else: # どちらの形式もない場合のフォールバック
    DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("gaussian_blur_kernel", (0, 0))


DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_open_apply", False)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_open_kernel_size", 3)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_open_iterations", 1)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_close_apply", False)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_close_kernel_size", 3)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_close_iterations", 1)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("haar_apply", True) # configの値を優先
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("haar_rect_h", 22)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("haar_peak_threshold", 7.0)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("h_proj_apply", True)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("h_proj_threshold_ratio", 0.15)


# === メインの前処理関数 (preprocess_uploaded_image) ===
def preprocess_uploaded_image(
    uploaded_file_data: bytes,
    mime_type: str,
    max_pixels: Optional[int] = None,
    output_format: Optional[Literal["JPEG", "PNG"]] = None,
    jpeg_quality: Optional[int] = None,
    grayscale: Optional[bool] = None,
    apply_trimming_opencv_override: Optional[bool] = None,
    trim_params_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    アップロードされた画像データを前処理するメイン関数。
    向き補正、トリミング（オプション）、リサイズ、フォーマット変換などを行う。
    戻り値: 処理結果を含む辞書。エラー時は "error" キーを含む。
    """
    debug_images: List[Dict[str, Any]] = []
    if not uploaded_file_data:
        return {"error": "画像データが提供されていません。", "debug_images": debug_images}

    # --- 引数のデフォルト値解決 ---
    effective_max_pixels = max_pixels if max_pixels is not None else DEFAULT_MAX_PIXELS_RESIZE
    effective_output_format = (output_format if output_format is not None else DEFAULT_OUTPUT_FORMAT).upper()
    effective_jpeg_quality = jpeg_quality if jpeg_quality is not None else DEFAULT_JPEG_QUALITY
    effective_grayscale = grayscale if grayscale is not None else DEFAULT_APPLY_GRAYSCALE

    # --- MIMEタイプと保存フォーマットの決定 ---
    original_mime_type = mime_type.lower()
    target_format_for_final_bytes = effective_output_format
    mime_type_for_llm = ""
    if original_mime_type in CONVERTABLE_MIME_TYPES:
        mime_type_for_llm = CONVERTABLE_MIME_TYPES[original_mime_type]
        if mime_type_for_llm == "image/png": target_format_for_final_bytes = "PNG"
        elif mime_type_for_llm == "image/jpeg": target_format_for_final_bytes = "JPEG"
        else:
            target_format_for_final_bytes = "PNG"
            mime_type_for_llm = "image/png"
        print(f"[image_processor] Converting {original_mime_type} to {mime_type_for_llm} (final bytes as {target_format_for_final_bytes}).")
    elif original_mime_type in SUPPORTED_MIME_TYPES:
        mime_type_for_llm = original_mime_type
    else:
        return {"error": f"サポートされていない入力画像形式です: {original_mime_type}。", "debug_images": debug_images}
    if mime_type_for_llm not in ["image/jpeg", "image/png"]:
        print(f"[image_processor] Warning: LLM MIME type '{mime_type_for_llm}' not directly supported by Gemini. Defaulting to image/png for LLM.")
        mime_type_for_llm = "image/png"
        if target_format_for_final_bytes != "PNG":
            print(f"[image_processor] Also changing final bytes format to PNG due to LLM MIME type fallback.")
            target_format_for_final_bytes = "PNG"

    # --- OpenCVトリミングのパラメータ解決 ---
    current_cv_trim_params = DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.copy()
    if trim_params_override:
        current_cv_trim_params.update(trim_params_override)
    if apply_trimming_opencv_override is not None:
        should_apply_opencv_trim = apply_trimming_opencv_override
    else:
        should_apply_opencv_trim = bool(current_cv_trim_params.get("apply", True))
    current_cv_trim_params["apply"] = should_apply_opencv_trim

    if should_apply_opencv_trim:
        print(f"[image_processor] OpenCV Trimming will be applied with params: {current_cv_trim_params}")
    else:
        print(f"[image_processor] OpenCV Trimming will be skipped.")

    try:
        img_pil = Image.open(BytesIO(uploaded_file_data))
        initial_bytes_png = image_to_bytes(img_pil.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
        if initial_bytes_png:
            debug_images.append({"label": "0. Initial Load (as PNG)", "data": initial_bytes_png, "mode": img_pil.mode, "size": img_pil.size})

        img_oriented = correct_image_orientation(img_pil)
        if img_oriented.size != img_pil.size or \
           (hasattr(img_pil, '_getexif') and img_pil._getexif() and img_pil._getexif().get(0x0112, 1) != 1):
            oriented_bytes_png = image_to_bytes(img_oriented.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
            if oriented_bytes_png:
                debug_images.append({"label": "1. Orientation Corrected (as PNG)", "data": oriented_bytes_png, "mode": img_oriented.mode, "size": img_oriented.size})

        # --- 結果を格納する辞書を準備 ---
        processing_results = {
            "contour_trim": {"main_image": None, "ocr_binary": None, "debug_label_prefix": "ContourTrim"},
            "ocr_trim": {"main_image": None, "ocr_binary": None, "debug_label_prefix": "OCRTrim"}
        }

        # === A. 輪郭ベースのトリミング実行 ===
        if bool(current_cv_trim_params.get("apply", True)):
            print("[image_processor] Executing Contour-based trimming...")
            contour_debug_collector: List[Dict[str, Any]] = []
            temp_main_contour, temp_ocr_contour = trim_image_by_contours(
                img_oriented.copy(),
                trim_params=current_cv_trim_params,
                jpeg_quality_for_debug_bytes=effective_jpeg_quality,
                debug_image_collector=contour_debug_collector
            )
            processing_results["contour_trim"]["main_image"] = temp_main_contour
            processing_results["contour_trim"]["ocr_binary"] = temp_ocr_contour
            for item in contour_debug_collector:
                item["label"] = f"ContourTrim - {item.get('label', '')}"
                debug_images.append(item)
            if not temp_main_contour:
                 print("[image_processor] Contour-based trimming did not yield a main image.")
        else:
            print("[image_processor] Contour-based trimming is disabled by 'apply' flag.")
            processing_results["contour_trim"]["main_image"] = img_oriented.copy()

        # === B. OCRベースのトリミング実行 ===
        if OCR_PROCESSORS_PYTESSERACT_AVAILABLE:
            print("[image_processor] Executing OCR-based trimming...")
            ocr_trim_padding = int(current_cv_trim_params.get("ocr_trim_padding", 0))
            ocr_trim_lang = str(current_cv_trim_params.get("ocr_trim_lang", "eng+jpn"))
            ocr_trim_min_conf = int(current_cv_trim_params.get("ocr_trim_min_conf", 25))
            ocr_min_h = int(current_cv_trim_params.get("ocr_trim_min_box_height", 5))
            ocr_max_h_r = float(current_cv_trim_params.get("ocr_trim_max_box_height_ratio", 0.3))
            ocr_min_w = int(current_cv_trim_params.get("ocr_trim_min_box_width", 5))
            ocr_max_w_r = float(current_cv_trim_params.get("ocr_trim_max_box_width_ratio", 0.8))
            ocr_min_ar = float(current_cv_trim_params.get("ocr_trim_min_aspect_ratio", 0.05))
            ocr_max_ar = float(current_cv_trim_params.get("ocr_trim_max_aspect_ratio", 20.0))
            ocr_tess_cfg = str(current_cv_trim_params.get("ocr_tesseract_config", "--psm 6"))
            
            temp_main_ocr, debug_cv_ocr_trim = trim_image_by_ocr_text_bounds(
                img_oriented.copy(),
                padding=ocr_trim_padding,
                lang=ocr_trim_lang,
                min_confidence=ocr_trim_min_conf,
                min_box_height=ocr_min_h,
                max_box_height_ratio=ocr_max_h_r,
                min_box_width=ocr_min_w,
                max_box_width_ratio=ocr_max_w_r,
                min_aspect_ratio=ocr_min_ar,
                max_aspect_ratio=ocr_max_ar,
                tesseract_config=ocr_tess_cfg,
                return_debug_cv_image=True
            )
            processing_results["ocr_trim"]["main_image"] = temp_main_ocr
            processing_results["ocr_trim"]["ocr_binary"] = None
            if debug_cv_ocr_trim is not None:
                debug_bytes_ocr = cv_image_to_bytes(debug_cv_ocr_trim, target_format="JPEG", jpeg_quality=effective_jpeg_quality)
                if debug_bytes_ocr:
                    debug_images.append({
                        "label": "OCRTrim - Text Detection Detail", 
                        "data": debug_bytes_ocr, "mode": "RGB",
                        "size": (debug_cv_ocr_trim.shape[1], debug_cv_ocr_trim.shape[0])
                    })
            if temp_main_ocr and debug_images is not None:
                 ocr_trimmed_final_bytes = image_to_bytes(temp_main_ocr.copy(), target_format="PNG")
                 if ocr_trimmed_final_bytes:
                    debug_images.append({
                        "label": "OCRTrim - Final Cropped", "data": ocr_trimmed_final_bytes,
                        "mode": temp_main_ocr.mode, "size": temp_main_ocr.size
                    })
            if not temp_main_ocr:
                print("[image_processor] OCR-based trimming did not yield a main image.")
        else:
            print("[image_processor] OCR-based trimming skipped (Pytesseract not available).")
            processing_results["ocr_trim"]["main_image"] = img_oriented.copy()

        # --- 最終的な画像選択 (ここでは輪郭ベースを優先) ---
        img_after_main_processing_step = processing_results["contour_trim"]["main_image"]
        if img_after_main_processing_step is None:
            img_after_main_processing_step = img_oriented.copy()
        ocr_binary_intermediate = processing_results["contour_trim"]["ocr_binary"]

        # --- 表示/Vision API用画像の最終処理 (Pillowベース) ---
        img_display_vision = img_after_main_processing_step.copy()
        img_display_vision = convert_to_rgb_with_alpha_handling(img_display_vision)
        if debug_images is not None and img_display_vision.mode != img_after_main_processing_step.mode:
            rgb_bytes = image_to_bytes(img_display_vision.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
            if rgb_bytes:
                debug_images.append({"label": "2. Display/Vision - RGB Converted (as PNG)", "data": rgb_bytes, "mode": img_display_vision.mode, "size": img_display_vision.size})
        if effective_grayscale:
            img_display_vision_prev_mode = img_display_vision.mode
            img_display_vision = apply_grayscale_pillow(img_display_vision)
            if debug_images is not None and img_display_vision.mode != img_display_vision_prev_mode:
                gray_bytes = image_to_bytes(img_display_vision.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
                if gray_bytes:
                    debug_images.append({"label": "3. Display/Vision - Grayscale (as PNG)", "data": gray_bytes, "mode": img_display_vision.mode, "size": img_display_vision.size})
        img_display_vision_prev_size = img_display_vision.size
        img_display_vision = resize_image_pillow(img_display_vision, effective_max_pixels)
        if debug_images is not None and img_display_vision.size != img_display_vision_prev_size:
            resized_bytes = image_to_bytes(img_display_vision.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
            if resized_bytes:
                debug_images.append({"label": "4. Display/Vision - Resized (as PNG)", "data": resized_bytes, "mode": img_display_vision.mode, "size": img_display_vision.size})
        final_display_vision_bytes = image_to_bytes(
            img_display_vision, 
            target_format=target_format_for_final_bytes,
            jpeg_quality=effective_jpeg_quality
        )
        if not final_display_vision_bytes:
            return {"error": "表示/Vision用画像のバイト変換に失敗しました。", "debug_images": debug_images}
        if debug_images is not None:
             debug_images.append({"label": f"5. Final Display/Vision ({target_format_for_final_bytes})", "data": final_display_vision_bytes, "mode": img_display_vision.mode, "size": img_display_vision.size})

        # --- OCR入力用画像の準備 ---
        final_ocr_input_bytes: Optional[bytes] = None
        ocr_mime_type_llm = "image/png"
        img_for_ocr: Optional[Image.Image] = None
        ocr_source_description = "Unknown"
        if ocr_binary_intermediate:
            img_for_ocr = ocr_binary_intermediate.copy()
            ocr_source_description = "From contour_trimmers (trimmed & binarized)"
            print(f"[image_processor] OCR image: Using pre-binarized image from contour_trimmers.")
        else:
            source_image_for_binarization: Optional[Image.Image] = None
            source_desc_key_for_binarization = ""
            if should_apply_opencv_trim and img_after_main_processing_step:
                source_image_for_binarization = img_after_main_processing_step.copy()
                source_desc_key_for_binarization = "TrimmedMain"
                print(f"[image_processor] OCR image: Attempting to binarize the '{source_desc_key_for_binarization}' image.")
            elif not should_apply_opencv_trim and img_oriented:
                source_image_for_binarization = img_oriented.copy()
                source_desc_key_for_binarization = "OrientedOriginal_NoTrim"
                print(f"[image_processor] OCR image: Attempting to binarize the '{source_desc_key_for_binarization}' image (trimming was skipped).")
            elif img_oriented:
                source_image_for_binarization = img_oriented.copy()
                source_desc_key_for_binarization = "OrientedOriginal_Fallback"
                print(f"[image_processor] OCR image: Fallback to binarizing '{source_desc_key_for_binarization}' image (trim failure or missing intermediate).")
            else:
                print(f"[image_processor] OCR image: Critical - No suitable source image found for binarization.")
            if source_image_for_binarization:
                cv_gray_for_ocr = convert_pil_to_cv_gray(source_image_for_binarization)
                if cv_gray_for_ocr is not None:
                    block_size = current_cv_trim_params.get("adaptive_thresh_block_size", DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("adaptive_thresh_block_size", 11))
                    c_val = current_cv_trim_params.get("adaptive_thresh_c", DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("adaptive_thresh_c", 7))
                    binary_cv_for_ocr = apply_adaptive_threshold(cv_gray_for_ocr, block_size, c_val)
                    if binary_cv_for_ocr is not None:
                        img_for_ocr = Image.fromarray(binary_cv_for_ocr, mode='L')
                        ocr_source_description = f"Binarized from {source_desc_key_for_binarization} (block:{block_size}, c:{c_val})"
                        print(f"[image_processor] OCR image: Successfully binarized from '{source_desc_key_for_binarization}'.")
                        if debug_images is not None:
                            debug_ocr_bytes = image_to_bytes(img_for_ocr.copy(), target_format="PNG")
                            if debug_ocr_bytes:
                                debug_images.append({
                                    "label": f"OCR - Binarized ({source_desc_key_for_binarization})",
                                    "data": debug_ocr_bytes, "mode": img_for_ocr.mode, "size": img_for_ocr.size
                                })
                    else:
                        print(f"[image_processor] OCR image: Binarization failed for '{source_desc_key_for_binarization}'. Falling back to Pillow grayscale.")
                        img_for_ocr = apply_grayscale_pillow(source_image_for_binarization.copy())
                        ocr_source_description = f"Pillow Grayscale from {source_desc_key_for_binarization} (binarization failed)"
                else:
                    print(f"[image_processor] OCR image: CV Gray conversion failed for '{source_desc_key_for_binarization}'. Falling back to Pillow grayscale.")
                    img_for_ocr = apply_grayscale_pillow(source_image_for_binarization.copy())
                    ocr_source_description = f"Pillow Grayscale from {source_desc_key_for_binarization} (CV gray failed)"
            else:
                 print(f"[image_processor] OCR image: No source image available for OCR processing step. OCR will likely fail.")
        if img_for_ocr:
            img_for_ocr_prev_size = img_for_ocr.size
            img_for_ocr_resized = resize_image_pillow(img_for_ocr, effective_max_pixels)
            if debug_images is not None and img_for_ocr_resized.size != img_for_ocr_prev_size:
                ocr_resized_bytes = image_to_bytes(img_for_ocr_resized.copy(), target_format="PNG")
                if ocr_resized_bytes:
                    debug_images.append({"label": f"OCR - Resized ({ocr_source_description})", "data": ocr_resized_bytes, "mode": img_for_ocr_resized.mode, "size": img_for_ocr_resized.size})
            img_for_ocr = img_for_ocr_resized
            if img_for_ocr.mode != 'L':
                original_mode_for_log = img_for_ocr.mode
                img_for_ocr = img_for_ocr.convert('L')
                print(f"[image_processor] OCR image: Converted mode '{original_mode_for_log}' to 'L'.")
            final_ocr_input_bytes = image_to_bytes(img_for_ocr, target_format="PNG")
            if final_ocr_input_bytes and debug_images is not None:
                    final_ocr_label = f"Final OCR Input (PNG) - Source: {ocr_source_description}"
                    if len(final_ocr_label) > 100: final_ocr_label = final_ocr_label[:97] + "..."
                    debug_images.append({
                        "label": final_ocr_label,
                        "data": final_ocr_input_bytes, 
                        "mode": img_for_ocr.mode, 
                        "size": img_for_ocr.size
                    })
        else:
            print("[image_processor] Warning: img_for_ocr is None after processing attempts. No OCR image will be generated.")
            ocr_source_description = "Generation Failed"

        # --- 結果の組み立て ---
        result: Dict[str, Any] = {
            "processed_image": { "mime_type": mime_type_for_llm, "data": final_display_vision_bytes },
            "debug_images": debug_images,
            "ocr_source_description": ocr_source_description,
            "contour_trimmed_image_data": None,
            "ocr_trimmed_image_data": None,
        }
        if processing_results["contour_trim"]["main_image"]:
            result["contour_trimmed_image_data"] = image_to_bytes(processing_results["contour_trim"]["main_image"].copy(), target_format=effective_output_format, jpeg_quality=effective_jpeg_quality)
        if processing_results["ocr_trim"]["main_image"]:
             result["ocr_trimmed_image_data"] = image_to_bytes(processing_results["ocr_trim"]["main_image"].copy(), target_format=effective_output_format, jpeg_quality=effective_jpeg_quality)
        if final_ocr_input_bytes:
            result["ocr_input_image"] = { "mime_type": ocr_mime_type_llm, "data": final_ocr_input_bytes }
        return result
    except Image.DecompressionBombError as e_bomb:
        return {"error": f"画像が大きすぎるか、非対応の形式の可能性があります (DecompressionBomb): {e_bomb}", "debug_images": debug_images}
    except ValueError as e_val:
        if "Pixel count exceeds limit" in str(e_val):
            return {"error": f"画像のピクセル数が大きすぎます (Pillow limit): {e_val}", "debug_images": debug_images}
        print(f"[image_processor] ValueError during preprocessing: {e_val}")
        import traceback; traceback.print_exc()
        return {"error": f"画像処理中に予期せぬValueErrorが発生しました: {e_val}", "debug_images": debug_images}
    except Exception as e:
        import traceback
        print(f"[image_processor] Unexpected critical error in preprocess_uploaded_image: {e}")
        traceback.print_exc()
        return {"error": f"画像処理中に予期せぬエラーが発生しました: {e}", "debug_images": debug_images}