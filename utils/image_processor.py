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
    cv_image_to_bytes
)
from .image_processing_parts.contour_trimmers import trim_image_by_contours
from .image_processing_parts.opencv_pipeline_utils import (
    convert_pil_to_cv_gray,
    apply_adaptive_threshold,
)
from .image_processing_parts.ocr_processors import (
    trim_image_by_ocr_text_bounds,
    PYTESSERACT_AVAILABLE as OCR_PROCESSORS_PYTESSERACT_AVAILABLE
)

# --- 設定値読み込み ---
_full_config = config_loader.get_config()
IMG_PROC_CONFIG = _full_config.get("image_processing", {})

# Pillowの最大許容ピクセル数設定 (DecompressionBomb対策)
pillow_max_pixels_cfg = IMG_PROC_CONFIG.get("pillow_max_image_pixels", 225000000) # デフォルト2.25億ピクセル
if pillow_max_pixels_cfg is not None and isinstance(pillow_max_pixels_cfg, (int, float)) and pillow_max_pixels_cfg > 0: # 数値型かつ正の値かチェック
    try:
        Image.MAX_IMAGE_PIXELS = int(pillow_max_pixels_cfg)
    except (ValueError, TypeError): # int変換失敗はほぼないはずだが念のため
        print(f"[image_processor] Warning: config.yaml内の 'pillow_max_image_pixels' ({pillow_max_pixels_cfg}) の設定に問題がある可能性があります。Pillowのデフォルト値を使用します。")
        Image.MAX_IMAGE_PIXELS = None 
else:
    if pillow_max_pixels_cfg is not None: # Noneでなく、0以下や不正な型だった場合
        print(f"[image_processor] Warning: config.yaml内の 'pillow_max_image_pixels' ({pillow_max_pixels_cfg}) は無効な値です。Pillowのデフォルト値を使用します。")
    Image.MAX_IMAGE_PIXELS = None # Noneまたは0以下、不正な型の場合はPillowデフォルト

# サポートMIMEタイプと変換ルール
SUPPORTED_MIME_TYPES = IMG_PROC_CONFIG.get("supported_mime_types", ["image/png", "image/jpeg", "image/webp"])
CONVERTABLE_MIME_TYPES = IMG_PROC_CONFIG.get("convertable_mime_types", {"image/gif": "image/png", "image/bmp": "image/png"})

# preprocess_uploaded_image 関数のデフォルト引数値 (config.yaml からロード)
DEFAULT_MAX_PIXELS_RESIZE = IMG_PROC_CONFIG.get("default_max_pixels_for_resizing", 4000000)
DEFAULT_OUTPUT_FORMAT: Literal["JPEG", "PNG"] = IMG_PROC_CONFIG.get("default_output_format", "JPEG").upper() # type: ignore
DEFAULT_JPEG_QUALITY = IMG_PROC_CONFIG.get("default_jpeg_quality", 85)
DEFAULT_APPLY_GRAYSCALE = IMG_PROC_CONFIG.get("apply_grayscale", True)

# OpenCV輪郭ベーストリミングのデフォルトパラメータ (config.yaml からロード)
# これは contour_trimmers.trim_image_by_contours に渡す trim_params の基礎となる。
# config.yaml の opencv_trimming セクションをコピーして使用する。
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG = IMG_PROC_CONFIG.get("opencv_trimming", {}).copy()

# config.yaml にキーが存在しない場合のフォールバック値を設定。
# contour_trimmers.py 側でも .get(key, default) でフォールバックするため必須ではないが、
# config構造の明確化や、このモジュールレベルでのデフォルト値保証に役立つ。
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("apply", True)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("padding", 0)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("adaptive_thresh_block_size", 11)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("adaptive_thresh_c", 7)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("min_contour_area_ratio", 0.00005)

# ガウシアンブラーカーネルの設定: config.yaml がリスト形式か個別キー形式かに対応
gb_kernel_cfg = DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("gaussian_blur_kernel")
if isinstance(gb_kernel_cfg, list) and len(gb_kernel_cfg) == 2: # [幅, 高さ] 形式
    DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG["gaussian_blur_kernel"] = tuple(map(int, gb_kernel_cfg))
elif "gaussian_blur_kernel_width" in DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG and \
     "gaussian_blur_kernel_height" in DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG: # 個別キー形式
    DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG["gaussian_blur_kernel"] = (
        int(DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("gaussian_blur_kernel_width", 0)),
        int(DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("gaussian_blur_kernel_height", 0))
    )
else: # どちらの形式もない場合のフォールバック (ブラーなし)
    DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("gaussian_blur_kernel", (0, 0))

# モルフォロジー演算関連のパラメータ
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_open_apply", False)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_open_kernel_size", 3)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_open_iterations", 1)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_close_apply", False)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_close_kernel_size", 3)
DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.setdefault("morph_close_iterations", 1)

# Haar-Like および 水平射影関連のパラメータ設定は、機能廃止に伴い削除済み

# ★変更: デフォルトトリミング戦略をconfigから読み込み
DEFAULT_TRIMMING_STRATEGY = IMG_PROC_CONFIG.get("default_trimming_strategy", "ocr_then_contour")

def preprocess_uploaded_image(
    uploaded_file_data: bytes,
    mime_type: str,
    max_pixels: Optional[int] = None,
    output_format: Optional[Literal["JPEG", "PNG"]] = None,
    jpeg_quality: Optional[int] = None,
    grayscale: Optional[bool] = None,
    apply_trimming_opencv_override: Optional[bool] = None,
    trim_params_override: Optional[Dict[str, Any]] = None,
    trimming_strategy_override: Optional[str] = None
) -> Dict[str, Any]:
    """アップロードされた画像データを前処理するメイン関数。

    処理フロー:
    1.  入力バリデーションとパラメータのデフォルト値解決。
    2.  MIMEタイプに基づいて、LLM連携用と最終出力用の画像フォーマットを決定。
    3.  Pillowで画像を開き、初期のデバッグ画像（PNG形式）を保存。
    4.  EXIF情報と射影プロファイルに基づいて画像の向きを補正 (base_image_utils.correct_image_orientation)。
        向きが変更された場合、デバッグ画像を保存。
    5.  輪郭ベーストリミング (contour_trimmers.trim_image_by_contours) を実行。
        -   `apply_trimming_opencv_override` と `trim_params_override` (およびconfigの`opencv_trimming`) に基づいて実行。
        -   結果（メイン画像、OCR用二値化画像）とデバッグ画像を収集。
    6.  OCRベーストリミング (ocr_processors.trim_image_by_ocr_text_bounds) を実行。
        -   Pytesseractが利用可能な場合のみ。
        -   `trim_params_override` (およびconfigの`opencv_trimming.ocr_trim_*`) に基づいて実行。
        -   結果（メイン画像）とデバッグ画像（テキスト検出状況、最終クロップ）を収集。
    7.  最終的なメイン画像を選択。現状は輪郭ベースの結果を優先。失敗時は向き補正済み画像。
    8.  メイン画像に対して、表示/Vision API用の最終処理（RGB変換、グレースケール化、リサイズ）をPillowで行う。
        各ステップでデバッグ画像を保存。最終的なバイトデータとMIMEタイプを生成。
    9.  OCR入力用の画像を準備。
        -   輪郭ベーストリミングから二値化画像が得られていればそれを使用。
        -   なければ、選択されたメイン画像（または向き補正済み画像）をOpenCVで二値化。
        -   二値化失敗時はPillowでグレースケール化。
        -   リサイズ、'L'モード変換後、最終的なPNGバイトデータを生成。
        -   各ステップでデバッグ画像を保存。
    10. 全ての処理結果（表示/Vision用画像、OCR入力用画像、各トリミング手法の結果画像、デバッグ画像リストなど）
        を含む辞書を返す。エラー発生時は "error" キーを含む辞書を返す。

    Args:
        uploaded_file_data: アップロードされた画像のバイトデータ。
        mime_type: 画像のMIMEタイプ。
        max_pixels: リサイズ処理時の最大総ピクセル数。Noneの場合はconfigのデフォルト値。
        output_format: 最終出力画像のフォーマット ("JPEG" または "PNG")。Noneの場合はconfigのデフォルト値。
        jpeg_quality: JPEG出力時の品質。Noneの場合はconfigのデフォルト値。
        grayscale: 表示/Vision API用画像にグレースケールを適用するか。Noneの場合はconfigのデフォルト値。
        apply_trimming_opencv_override: 輪郭ベーストリミングの適用/非適用を強制的に上書きするか。
                                       Noneの場合はconfigまたはtrim_params_overrideの設定に従う。
        trim_params_override: 画像処理（特にトリミング）のパラメータを上書きするための辞書。
                              configの値よりも優先される。
        trimming_strategy_override: トリミング戦略を上書きするための文字列。

    Returns:
        処理結果を含む辞書。主要なキーは以下の通り:
        - "processed_image": {"mime_type": str, "data": bytes} (表示/Vision API用)
        - "ocr_input_image": {"mime_type": str, "data": bytes} (OCR処理用、PNG形式)
        - "debug_images": List[Dict[str, Any]] (各処理ステップのデバッグ画像情報)
        - "ocr_source_description": str (OCR用画像がどの処理から生成されたかの説明)
        - "contour_trimmed_image_data": Optional[bytes] (輪郭ベーストリミング結果の画像データ)
        - "ocr_trimmed_image_data": Optional[bytes] (OCRベーストリミング結果の画像データ)
        - "error": str (エラー発生時のみ)
    """
    debug_images: List[Dict[str, Any]] = [] # デバッグ画像情報を格納するリスト
    if not uploaded_file_data:
        return {"error": "画像データが提供されていません。", "debug_images": debug_images}

    # --- 引数のデフォルト値解決 ---
    effective_max_pixels = max_pixels if max_pixels is not None else DEFAULT_MAX_PIXELS_RESIZE
    effective_output_format = (output_format if output_format is not None else DEFAULT_OUTPUT_FORMAT).upper()
    effective_jpeg_quality = jpeg_quality if jpeg_quality is not None else DEFAULT_JPEG_QUALITY
    effective_grayscale = grayscale if grayscale is not None else DEFAULT_APPLY_GRAYSCALE

    # --- MIMEタイプと保存フォーマットの決定 ---
    original_mime_type = mime_type.lower()
    target_format_for_final_bytes = effective_output_format # 表示/Vision用画像の最終フォーマット
    mime_type_for_llm = "" # LLMに渡す際のMIMEタイプ

    if original_mime_type in CONVERTABLE_MIME_TYPES: # GIFやBMPなど、変換が必要な形式の場合
        mime_type_for_llm = CONVERTABLE_MIME_TYPES[original_mime_type]
        # 変換先のMIMEタイプに応じて最終出力フォーマットも調整
        if mime_type_for_llm == "image/png": target_format_for_final_bytes = "PNG"
        elif mime_type_for_llm == "image/jpeg": target_format_for_final_bytes = "JPEG"
        else: # 不明な変換先の場合はPNGにフォールバック
            target_format_for_final_bytes = "PNG"
            mime_type_for_llm = "image/png"
        print(f"[image_processor] MIMEタイプ変換: {original_mime_type} -> {mime_type_for_llm} (最終出力形式: {target_format_for_final_bytes})")
    elif original_mime_type in SUPPORTED_MIME_TYPES: # 直接サポートされている形式の場合
        mime_type_for_llm = original_mime_type
    else: # サポート外の形式
        return {"error": f"サポートされていない入力画像形式です: {original_mime_type}。", "debug_images": debug_images}
    
    # LLM (Gemini) がサポートするMIMEタイプか確認し、そうでなければPNGにフォールバック
    if mime_type_for_llm not in ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]: # TODO: Geminiの最新サポート状況確認
        print(f"[image_processor] Warning: LLM用MIMEタイプ '{mime_type_for_llm}' はGeminiで直接サポートされていません。LLM用には image/png にフォールバックします。")
        mime_type_for_llm = "image/png"
        if target_format_for_final_bytes != "PNG": # LLM用がPNGになった場合、最終出力もPNGに合わせる
            print(f"[image_processor] LLM用MIMEタイプのフォールバックに伴い、最終出力形式もPNGに変更します。")
            target_format_for_final_bytes = "PNG"

    # --- 輪郭ベーストリミングのパラメータ解決 ---
    # configのデフォルト値をコピーし、引数 `trim_params_override` で上書き
    current_cv_trim_params = DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.copy()
    if trim_params_override:
        current_cv_trim_params.update(trim_params_override)
    
    # 輪郭ベーストリミングを適用するかどうかの決定
    if apply_trimming_opencv_override is not None: # 引数で強制指定されている場合
        should_apply_contour_trim = apply_trimming_opencv_override
    else: # そうでなければ、解決済みのパラメータ辞書内の 'apply' キーの値に従う
        should_apply_contour_trim = bool(current_cv_trim_params.get("apply", True))
    current_cv_trim_params["apply"] = should_apply_contour_trim # 最終的な適用フラグをパラメータ辞書にも反映

    if should_apply_contour_trim:
        print(f"[image_processor] 輪郭ベーストリミングが適用されます。パラメータ: {current_cv_trim_params}")
    else:
        print(f"[image_processor] 輪郭ベーストリミングはスキップされます。")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 修正点: should_apply_ocr_trim_tuning の定義を追加 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    should_apply_ocr_trim_tuning = bool(
        current_cv_trim_params.get(
            "ocr_trim_tuning_apply", # UIからのキー
            DEFAULT_OPENCV_TRIM_PARAMS_FROM_CONFIG.get("ocr_trim_tuning_apply", True) # configのデフォルト
        )
    )
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    # ★変更: デフォルトトリミング戦略をconfigから読み込み
    effective_trimming_strategy = trimming_strategy_override if trimming_strategy_override is not None else DEFAULT_TRIMMING_STRATEGY
    print(f"[image_processor] 使用するトリミング戦略: {effective_trimming_strategy}")

    try:
        # --- 画像読み込みと初期向き補正 ---
        img_pil = Image.open(BytesIO(uploaded_file_data))
        # 初期ロード状態をデバッグ用に保存 (PNG形式)
        initial_bytes_png = image_to_bytes(img_pil.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
        if initial_bytes_png:
            debug_images.append({"label": "0. 初期ロード時 (PNG変換後)", "data": initial_bytes_png, "mode": img_pil.mode, "size": img_pil.size})

        # EXIFベースと射影プロファイルベースの向き補正
        img_oriented = correct_image_orientation(img_pil)
        # 向きが変更されたか、またはEXIFに回転情報があった場合のみデバッグ画像追加
        if img_oriented.size != img_pil.size or \
           (hasattr(img_pil, '_getexif') and img_pil._getexif() and img_pil._getexif().get(0x0112, 1) != 1):
            oriented_bytes_png = image_to_bytes(img_oriented.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
            if oriented_bytes_png:
                debug_images.append({"label": "1. EXIF/射影プロファイル向き補正後 (PNG)", "data": oriented_bytes_png, "mode": img_oriented.mode, "size": img_oriented.size})

        # --- 各トリミング手法の結果を格納する辞書を準備 ---
        processing_results = {
            "contour_trim": {"main_image": None, "ocr_binary": None, "debug_label_prefix": "ContourTrim"},
            "ocr_trim": {"main_image": None, "ocr_binary": None, "debug_label_prefix": "OCRTrim"} # ocr_binaryは現在未使用
        }

        # === A. 輪郭ベースのトリミング実行 ===
        if should_apply_contour_trim:
            print("[image_processor] 輪郭ベーストリミングを実行します...")
            contour_debug_collector: List[Dict[str, Any]] = [] # このトリミング手法固有のデバッグ画像を収集
            contour_result_main, contour_result_ocr_binary = trim_image_by_contours(
                img_oriented.copy(), # 向き補正済みの画像をコピーして使用
                trim_params=current_cv_trim_params, # 解決済みのパラメータを渡す
                jpeg_quality_for_debug_bytes=effective_jpeg_quality,
                debug_image_collector=contour_debug_collector
            )
            processing_results["contour_trim"]["main_image"] = contour_result_main
            processing_results["contour_trim"]["ocr_binary"] = contour_result_ocr_binary # OCR用の二値化画像も受け取る
            # 集めたデバッグ画像にプレフィックスを付けて全体リストに追加
            for item in contour_debug_collector:
                item["label"] = f"輪郭Trim - {item.get('label', '')}"
                debug_images.append(item)
            if not contour_result_main:
                 print("[image_processor] 輪郭ベーストリミングでは有効なメイン画像が得られませんでした。")
        else: # 輪郭ベーストリミングをスキップする場合
            print("[image_processor] 輪郭ベーストリミングは 'apply' フラグによりスキップされました。")
            # スキップ時は、向き補正済み画像をそのまま次のステップの入力候補とする
            processing_results["contour_trim"]["main_image"] = img_oriented.copy()

        # --- OCRベーストリミングの実行条件を決定 ---
        execute_ocr_trim = False
        if OCR_PROCESSORS_PYTESSERACT_AVAILABLE:
            if trimming_strategy_override is None:
                # チューニングモード: UIのチェックボックスに従う
                execute_ocr_trim = should_apply_ocr_trim_tuning
                if execute_ocr_trim:
                    print("[image_processor] OCRベーストリミングを実行します (Pytesseract利用可 & チューニングUI設定ON)。")
                else:
                    print("[image_processor] OCRベーストリミングはスキップされました (チューニングUI設定で無効化)。")
            else:
                # AIチューターモード: 戦略に 'ocr' が含まれていれば必ず実行
                if "ocr" in effective_trimming_strategy:
                    execute_ocr_trim = True
                    print(f"[image_processor] OCRベーストリミングを実行します (戦略: {effective_trimming_strategy})。")
                else:
                    print(f"[image_processor] OCRベーストリミングはスキップされました (戦略: {effective_trimming_strategy} にOCR含まず)。")
        else:
            print("[image_processor] OCRベーストリミングはスキップされました (Pytesseract利用不可)。")

        # === B. OCRベースのトリミング実行 ===
        ocr_result_main = None
        debug_cv_ocr_trim_detail = None
        if execute_ocr_trim:
            # パラメータ取得 (current_cv_trim_params から。configの値は既にマージされている)
            ocr_trim_padding = int(current_cv_trim_params.get("ocr_trim_padding", 0))
            ocr_trim_lang = str(current_cv_trim_params.get("ocr_trim_lang", "eng+jpn"))
            ocr_trim_min_conf = int(current_cv_trim_params.get("ocr_trim_min_conf", 30))
            ocr_min_h = int(current_cv_trim_params.get("ocr_trim_min_box_height", 5))
            ocr_max_h_r = float(current_cv_trim_params.get("ocr_trim_max_box_height_ratio", 0.3))
            ocr_min_w = int(current_cv_trim_params.get("ocr_trim_min_box_width", 5))
            ocr_max_w_r = float(current_cv_trim_params.get("ocr_trim_max_box_width_ratio", 0.8))
            ocr_min_ar = float(current_cv_trim_params.get("ocr_trim_min_aspect_ratio", 0.1))
            ocr_max_ar = float(current_cv_trim_params.get("ocr_trim_max_aspect_ratio", 10.0))
            ocr_tess_cfg = str(current_cv_trim_params.get("ocr_tesseract_config", "--psm 6"))

            ocr_result_main, debug_cv_ocr_trim_detail = trim_image_by_ocr_text_bounds(
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
            processing_results["ocr_trim"]["main_image"] = ocr_result_main
            processing_results["ocr_trim"]["ocr_binary"] = None 
            if debug_cv_ocr_trim_detail is not None:
                debug_bytes_ocr_detail = cv_image_to_bytes(debug_cv_ocr_trim_detail, target_format="JPEG", jpeg_quality=effective_jpeg_quality)
                if debug_bytes_ocr_detail:
                    debug_images.append({
                        "label": "OCRTrim - テキストボックス検出詳細 (フィルタ前後)", 
                        "data": debug_bytes_ocr_detail, "mode": "RGB",
                        "size": (debug_cv_ocr_trim_detail.shape[1], debug_cv_ocr_trim_detail.shape[0])
                    })
            if ocr_result_main:
                ocr_trimmed_final_bytes_png = image_to_bytes(ocr_result_main.copy(), target_format="PNG")
                if ocr_trimmed_final_bytes_png:
                    debug_images.append({
                        "label": "OCRTrim - 最終クロップ結果 (PNG)", "data": ocr_trimmed_final_bytes_png,
                        "mode": ocr_result_main.mode, "size": ocr_result_main.size
                    })
            if not ocr_result_main:
                print("[image_processor] OCRベーストリミングでは有効なメイン画像が得られませんでした。")
        else:
            # スキップ時は ocr_result_main = None のまま
            processing_results["ocr_trim"]["main_image"] = None
            processing_results["ocr_trim"]["ocr_binary"] = None

        # --- ★変更: トリミング戦略に基づいてメイン画像とOCR用二値化画像を選択 ---
        img_after_main_processing_step: Optional[Image.Image] = None
        ocr_binary_intermediate: Optional[Image.Image] = None
        selected_trim_method_log = "未定"

        contour_main = processing_results["contour_trim"]["main_image"]
        contour_binary = processing_results["contour_trim"]["ocr_binary"]
        ocr_main = processing_results["ocr_trim"]["main_image"]
        # ocr_binary は OCRTrim からは直接提供されないので、後段で生成

        if effective_trimming_strategy == "ocr_then_contour":
            if ocr_main:
                img_after_main_processing_step = ocr_main
                selected_trim_method_log = "OCRベーストリミング"
            elif contour_main:
                img_after_main_processing_step = contour_main
                ocr_binary_intermediate = contour_binary
                selected_trim_method_log = "輪郭ベーストリミング (OCR失敗/無効フォールバック)"
            else:
                img_after_main_processing_step = img_oriented.copy()
                selected_trim_method_log = "トリミングなし (両方失敗/無効フォールバック)"

        elif effective_trimming_strategy == "contour_then_ocr":
            if contour_main:
                img_after_main_processing_step = contour_main
                ocr_binary_intermediate = contour_binary
                selected_trim_method_log = "輪郭ベーストリミング"
            elif ocr_main:
                img_after_main_processing_step = ocr_main
                selected_trim_method_log = "OCRベーストリミング (輪郭失敗/無効フォールバック)"
            else:
                img_after_main_processing_step = img_oriented.copy()
                selected_trim_method_log = "トリミングなし (両方失敗/無効フォールバック)"

        elif effective_trimming_strategy == "ocr_only":
            if ocr_main:
                img_after_main_processing_step = ocr_main
                selected_trim_method_log = "OCRベーストリミング (OCR Only)"
            else:
                img_after_main_processing_step = img_oriented.copy()
                selected_trim_method_log = "トリミングなし (OCR Only失敗/無効フォールバック)"

        elif effective_trimming_strategy == "contour_only":
            if contour_main:
                img_after_main_processing_step = contour_main
                ocr_binary_intermediate = contour_binary
                selected_trim_method_log = "輪郭ベーストリミング (Contour Only)"
            else:
                img_after_main_processing_step = img_oriented.copy()
                selected_trim_method_log = "トリミングなし (Contour Only失敗/無効フォールバック)"

        elif effective_trimming_strategy == "none":
            img_after_main_processing_step = img_oriented.copy()
            selected_trim_method_log = "トリミングなし (戦略: none)"
            
        else: # 未知の戦略またはデフォルトのフォールバック (ocr_then_contour と同様)
            print(f"[image_processor] Warning: 未知のトリミング戦略 '{effective_trimming_strategy}'。ocr_then_contour にフォールバックします。")
            if ocr_main:
                img_after_main_processing_step = ocr_main
                selected_trim_method_log = "OCRベーストリミング (未知戦略フォールバック)"
            elif contour_main:
                img_after_main_processing_step = contour_main
                ocr_binary_intermediate = contour_binary
                selected_trim_method_log = "輪郭ベーストリミング (未知戦略・OCR失敗フォールバック)"
            else:
                img_after_main_processing_step = img_oriented.copy()
                selected_trim_method_log = "トリミングなし (未知戦略・両方失敗フォールバック)"

        if img_after_main_processing_step is None: # このフォールバックは重要
            print("[image_processor] Critical: メイン処理ステップ後の画像がNoneです。向き補正済み画像にフォールバックします。")
            img_after_main_processing_step = img_oriented.copy()
            ocr_binary_intermediate = None # メインが None なら二値化もリセット
            selected_trim_method_log += " (Critical Fallback to Oriented)"
        print(f"[image_processor] 最終的なメイン画像の選択元: {selected_trim_method_log}")

        # --- 表示/Vision API用画像の最終処理 (Pillowベース) ---
        # 選択されたメイン画像をコピーして処理
        img_display_vision = img_after_main_processing_step.copy()
        
        # RGBA/Pモードの場合はRGBに変換 (アルファチャンネル除去または背景色付与)
        img_display_vision_prev_mode_for_debug = img_display_vision.mode
        img_display_vision = convert_to_rgb_with_alpha_handling(img_display_vision)
        if img_display_vision.mode != img_display_vision_prev_mode_for_debug: # モード変更があった場合
            rgb_converted_bytes_png = image_to_bytes(img_display_vision.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
            if rgb_converted_bytes_png:
                debug_images.append({"label": "2. 表示/Vision用 - RGB変換後 (PNG)", "data": rgb_converted_bytes_png, "mode": img_display_vision.mode, "size": img_display_vision.size})
        
        # グレースケール化 (オプション)
        if effective_grayscale:
            img_display_vision_prev_mode_for_debug = img_display_vision.mode
            img_display_vision = apply_grayscale_pillow(img_display_vision)
            if img_display_vision.mode != img_display_vision_prev_mode_for_debug or img_display_vision_prev_mode_for_debug != 'L': # モード変更または元々Lでなかった場合
                gray_bytes_png = image_to_bytes(img_display_vision.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
                if gray_bytes_png:
                    debug_images.append({"label": "3. 表示/Vision用 - グレースケール化後 (PNG)", "data": gray_bytes_png, "mode": img_display_vision.mode, "size": img_display_vision.size})
        
        # リサイズ (オプション)
        img_display_vision_prev_size_for_debug = img_display_vision.size
        img_display_vision = resize_image_pillow(img_display_vision, effective_max_pixels)
        if img_display_vision.size != img_display_vision_prev_size_for_debug: # サイズ変更があった場合
            resized_bytes_png = image_to_bytes(img_display_vision.copy(), target_format="PNG", jpeg_quality=effective_jpeg_quality)
            if resized_bytes_png:
                debug_images.append({"label": "4. 表示/Vision用 - リサイズ後 (PNG)", "data": resized_bytes_png, "mode": img_display_vision.mode, "size": img_display_vision.size})
        
        # 最終的なバイトデータに変換
        final_display_vision_bytes = image_to_bytes(
            img_display_vision, 
            target_format=target_format_for_final_bytes, # configまたは引数で決定されたフォーマット
            jpeg_quality=effective_jpeg_quality
        )
        if not final_display_vision_bytes: # 変換失敗
            return {"error": "表示/Vision用画像のバイトデータ変換に失敗しました。", "debug_images": debug_images}
        # 最終結果をデバッグ画像リストに追加
        debug_images.append({"label": f"5. 最終 表示/Vision用画像 ({target_format_for_final_bytes})", "data": final_display_vision_bytes, "mode": img_display_vision.mode, "size": img_display_vision.size})

        # --- OCR入力用画像の準備 ---
        final_ocr_input_bytes: Optional[bytes] = None
        ocr_mime_type_llm = "image/png" # OCR用はPNG形式を推奨
        img_for_ocr_processing: Optional[Image.Image] = None # OCR処理対象のPillow画像
        ocr_source_description = "不明" # OCR用画像がどこから来たかの説明

        if ocr_binary_intermediate is not None: # 輪郭トリミングから二値化画像が得られている場合
            img_for_ocr_processing = ocr_binary_intermediate.copy()
            ocr_source_description = f"戦略({selected_trim_method_log})で選択された二値化画像"
            print(f"[image_processor] OCR用画像: {ocr_source_description} を使用します。")
        else:
            source_image_for_ocr_binarization: Optional[Image.Image] = img_after_main_processing_step.copy()
            source_desc_key = f"戦略({selected_trim_method_log})で選択されたメイン画像"
            if source_image_for_ocr_binarization:
                print(f"[image_processor] OCR用画像: '{source_desc_key}' を二値化します。")
                cv_gray_for_ocr = convert_pil_to_cv_gray(source_image_for_ocr_binarization)
                if cv_gray_for_ocr is not None:
                    block_size = current_cv_trim_params.get("adaptive_thresh_block_size", 11)
                    c_val = current_cv_trim_params.get("adaptive_thresh_c", 7)
                    binary_cv_for_ocr = apply_adaptive_threshold(cv_gray_for_ocr, block_size, c_val)
                    if binary_cv_for_ocr is not None:
                        img_for_ocr_processing = Image.fromarray(binary_cv_for_ocr, mode='L')
                        ocr_source_description = f"{source_desc_key}からOpenCV二値化(B{block_size},C{c_val})"
                        debug_ocr_bin_bytes_png = image_to_bytes(img_for_ocr_processing.copy(), "PNG")
                        if debug_ocr_bin_bytes_png:
                            debug_images.append({"label": f"OCR用 - {source_desc_key}から二値化", "data": debug_ocr_bin_bytes_png, "mode": 'L', "size": img_for_ocr_processing.size})
                    else:
                        img_for_ocr_processing = apply_grayscale_pillow(source_image_for_ocr_binarization.copy())
                        ocr_source_description = f"{source_desc_key}からPillowグレースケール(OpenCV二値化失敗)"
                else:
                    img_for_ocr_processing = apply_grayscale_pillow(source_image_for_ocr_binarization.copy())
                    ocr_source_description = f"{source_desc_key}からPillowグレースケール(CVグレースケール失敗)"
            else:
                 ocr_source_description = "OCR用ソース画像なし(致命的)"
        
        # OCR処理用画像が存在すれば、最終処理（リサイズ、モード変換、バイト変換）
        if img_for_ocr_processing:
            img_for_ocr_prev_size_for_debug = img_for_ocr_processing.size
            # OCR用画像もリサイズ (表示/Vision用と同じ最大ピクセル数設定を使用)
            img_for_ocr_resized = resize_image_pillow(img_for_ocr_processing, effective_max_pixels)
            if img_for_ocr_resized.size != img_for_ocr_prev_size_for_debug: # サイズ変更があった場合
                ocr_resized_bytes_png = image_to_bytes(img_for_ocr_resized.copy(), target_format="PNG")
                if ocr_resized_bytes_png:
                    label_resized_ocr = f"OCR用 - リサイズ後 (元: {ocr_source_description})"[:100] # ラベル長制限
                    debug_images.append({"label": label_resized_ocr, "data": ocr_resized_bytes_png, "mode": img_for_ocr_resized.mode, "size": img_for_ocr_resized.size})
            img_for_ocr_processing = img_for_ocr_resized # リサイズ後の画像で更新
            
            # モードが 'L' (グレースケール) でない場合は変換
            if img_for_ocr_processing.mode != 'L':
                original_mode_for_log = img_for_ocr_processing.mode
                img_for_ocr_processing = img_for_ocr_processing.convert('L')
                print(f"[image_processor] OCR用画像: モード '{original_mode_for_log}' を 'L' に変換しました。")
            
            # 最終的なOCR入力用バイトデータ (PNG形式)
            final_ocr_input_bytes = image_to_bytes(img_for_ocr_processing, target_format="PNG")
            if final_ocr_input_bytes:
                    final_ocr_label = f"最終 OCR入力用画像 (PNG) - 元: {ocr_source_description}"[:100] # ラベル長制限
                    debug_images.append({
                        "label": final_ocr_label,
                        "data": final_ocr_input_bytes, 
                        "mode": img_for_ocr_processing.mode, 
                        "size": img_for_ocr_processing.size
                    })
            else:
                print("[image_processor] Warning: OCR用画像の最終バイトデータ変換に失敗しました。")
                ocr_source_description += " (最終バイト変換失敗)"
        else: # OCR処理用画像が生成できなかった場合
            print("[image_processor] Warning: OCR処理ステップ後、img_for_ocr_processing が None です。OCR用画像は生成されません。")
            ocr_source_description = "生成失敗"

        # --- 結果の組み立て ---
        result: Dict[str, Any] = {
            "processed_image": { # 表示/Vision API用
                "mime_type": mime_type_for_llm, # LLMに渡すMIMEタイプ
                "data": final_display_vision_bytes
            },
            "debug_images": debug_images,
            "ocr_source_description": ocr_source_description,
            "contour_trimmed_image_data": None, # 輪郭ベーストリミングの結果画像 (バイト)
            "ocr_trimmed_image_data": None,     # OCRベーストリミングの結果画像 (バイト)
        }
        # 各トリミング手法の結果画像をバイトデータとして結果辞書に追加
        if processing_results["contour_trim"]["main_image"]:
            result["contour_trimmed_image_data"] = image_to_bytes(
                processing_results["contour_trim"]["main_image"].copy(),
                target_format=effective_output_format, # 表示用と同じフォーマット
                jpeg_quality=effective_jpeg_quality
            )
        if processing_results["ocr_trim"]["main_image"]:
             result["ocr_trimmed_image_data"] = image_to_bytes(
                processing_results["ocr_trim"]["main_image"].copy(),
                target_format=effective_output_format, # 表示用と同じフォーマット
                jpeg_quality=effective_jpeg_quality
            )
        # OCR入力用画像が生成されていれば、それも結果辞書に追加
        if final_ocr_input_bytes:
            result["ocr_input_image"] = {
                "mime_type": ocr_mime_type_llm, # PNG形式
                "data": final_ocr_input_bytes
            }
        return result

    except Image.DecompressionBombError as e_bomb: # Pillowの最大ピクセル数超過エラー
        return {"error": f"画像が大きすぎるか、非対応の形式の可能性があります (DecompressionBomb): {e_bomb}", "debug_images": debug_images}
    except ValueError as e_val: # その他のPillow関連エラーなど
        if "Pixel count exceeds limit" in str(e_val): # Image.MAX_IMAGE_PIXELS 超過
            return {"error": f"画像のピクセル数がPillowの許容上限を超えています: {e_val}", "debug_images": debug_images}
        print(f"[image_processor] 画像前処理中に予期せぬValueErrorが発生しました: {e_val}")
        import traceback; traceback.print_exc()
        return {"error": f"画像処理中に予期せぬValueErrorが発生しました: {e_val}", "debug_images": debug_images}
    except Exception as e: # その他の予期せぬクリティカルエラー
        import traceback
        print(f"[image_processor] 画像前処理 (preprocess_uploaded_image) 中に予期せぬクリティカルエラーが発生しました: {e}")
        traceback.print_exc()
        return {"error": f"画像処理中に予期せぬエラーが発生しました: {e}", "debug_images": debug_images}