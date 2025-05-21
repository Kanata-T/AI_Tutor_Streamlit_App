import streamlit as st
import copy
from PIL import Image
from io import BytesIO

from utils.image_processor import preprocess_uploaded_image
from utils.image_processing_parts.base_image_utils import auto_orient_image_opencv, image_to_bytes
from .display_helpers import display_debug_images_app

def render_tuning_mode():
    """画像処理チューニングモードのUIとロジックをレンダリングします。"""
    st.title("🖼️ 画像処理 パラメータチューナー") # タイトルを汎用的に

    if not st.session_state.get("tuning_initialized", False):
        st.session_state.tuning_raw_image_data = None
        st.session_state.tuning_raw_image_mime_type = None
        st.session_state.tuning_current_debug_images = []
        st.session_state.tuning_image_key_counter = st.session_state.get("tuning_image_key_counter", 0)
        st.session_state.tuning_trigger_reprocess = False
        
        # チューニングUI用の編集可能パラメータの初期化
        # config.yamlの 'opencv_trimming' セクション全体を固定値として持つ想定
        # (輪郭ベースとOCRベースのパラメータが混在している)
        if "tuning_editable_cv_params" not in st.session_state or not st.session_state.tuning_initialized:
            if "tuning_fixed_cv_params" in st.session_state: # app.pyでロード済みのはず
                st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
            else: 
                st.session_state.tuning_editable_cv_params = {} 
                st.warning("固定CVパラメータ (tuning_fixed_cv_params) がセッションに見つかりません。")

        if "tuning_editable_other_params" not in st.session_state or not st.session_state.tuning_initialized:
            if "tuning_fixed_other_params" in st.session_state: # app.pyでロード済みのはず
                st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
            else:
                st.session_state.tuning_editable_other_params = {}
                st.warning("固定その他パラメータ (tuning_fixed_other_params) がセッションに見つかりません。")

        if "selected_param_set_name_tuning" not in st.session_state or not st.session_state.tuning_initialized:
            st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
            
        st.session_state.tuning_initialized = True
        st.session_state.tutor_initialized = False
        print("[TuningModeUI] 画像チューニングモードのセッション状態が初期化されました。")

    with st.sidebar:
        st.header("⚙️ 処理パラメータ (チューニング)")
        
        st.markdown("#### 1. 画像を選択")
        uploaded_file_obj_tune = st.file_uploader(
            "調整対象の画像", type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
            key=f"tuning_file_uploader_{st.session_state.get('tuning_image_key_counter', 0)}"
        )
        if st.button("現在の画像をクリア", key="clear_image_button_tuning"):
            st.session_state.tuning_raw_image_data = None
            st.session_state.tuning_raw_image_mime_type = None
            st.session_state.tuning_current_debug_images = []
            st.session_state.tuning_image_key_counter += 1
            if "tuning_fixed_cv_params" in st.session_state: # 固定値に戻す
                st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
            if "tuning_fixed_other_params" in st.session_state: # 固定値に戻す
                st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
            st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
            st.session_state.tuning_trigger_reprocess = False
            st.session_state.last_processing_result_tuning = None
            print("[TuningModeUI] 画像とパラメータがクリアされました。")
            st.rerun()

        st.markdown("#### 2. パラメータセット提案 (輪郭ベース用)")
        # プリセットは主に輪郭ベース用。OCRパラメータは固定値から取得される。
        fixed_cv_params_for_presets = st.session_state.get("tuning_fixed_cv_params", {})
        all_parameter_presets = {
            "現在の編集値": {},
            "00: 固定値 (推奨)": fixed_cv_params_for_presets, # これにはOCRの固定値も含まれる
            "01: 基本 (ブラーなし)": {
                "apply":True, "padding":15, "adaptive_thresh_block_size":11, 
                "adaptive_thresh_c":7, "min_contour_area_ratio":0.0005, 
                "gaussian_blur_kernel_width":0, "gaussian_blur_kernel_height":0, 
                "morph_open_apply":False, "morph_close_apply":False
            },
            "02: 小さい文字向け (ブロックサイズ小, C値調整, オープニング有)": {
                "apply":True, "padding":10, "adaptive_thresh_block_size":7, 
                "adaptive_thresh_c":3, "min_contour_area_ratio":0.0001, 
                "gaussian_blur_kernel_width":0, "gaussian_blur_kernel_height":0, 
                "morph_open_apply":True, "morph_open_kernel_size":2,
                "morph_open_iterations":1, "morph_close_apply":False
            },
        }
        selected_preset_name = st.selectbox(
            "輪郭パラメータセット:", 
            list(all_parameter_presets.keys()), 
            key="param_set_selector_tuning",
            index=list(all_parameter_presets.keys()).index(
                st.session_state.get("selected_param_set_name_tuning", "00: 固定値 (推奨)")
            )
        )
        if selected_preset_name != st.session_state.get("selected_param_set_name_tuning"):
            st.session_state.selected_param_set_name_tuning = selected_preset_name
            if selected_preset_name != "現在の編集値" and selected_preset_name in all_parameter_presets:
                new_editable_params = st.session_state.get("tuning_fixed_cv_params", {}).copy()
                preset_specific_values = all_parameter_presets[selected_preset_name]
                new_editable_params.update(preset_specific_values) # プリセット値で上書き
                st.session_state.tuning_editable_cv_params = new_editable_params
                print(f"[TuningModeUI] パラメータセット '{selected_preset_name}' が適用されました。")
            st.rerun()
        
        # --- 輪郭ベーストリミング設定 ---
        st.markdown("#### 3. 輪郭ベーストリミング 詳細設定")
        if "tuning_editable_cv_params" not in st.session_state:
            st.warning("編集可能CVパラメータが初期化されていません。")
            return 
        editable_params = st.session_state.tuning_editable_cv_params # 輪郭とOCR両方のパラメータが混在
        if not isinstance(editable_params, dict):
            st.error("セッション内の 'tuning_editable_cv_params' が不正な形式です。")
            return

        editable_params["apply"] = st.checkbox( # 輪郭ベースの適用フラグ
            "輪郭ベーストリミング適用", 
            value=editable_params.get("apply", True), 
            key="contour_trim_apply_tuning"
        )
        if editable_params.get("apply"):
            editable_params["padding"] = st.number_input(
                "輪郭: パディング (px)", min_value=0, max_value=200, 
                value=editable_params.get("padding", 0), step=1, 
                key="contour_trim_padding_tuning"
            )
            block_size_input_contour = st.number_input(
                "輪郭 適応閾値: ブロック (3+奇数)", min_value=3, 
                value=editable_params.get("adaptive_thresh_block_size", 11), step=2, 
                key="contour_trim_block_size_tuning"
            )
            editable_params["adaptive_thresh_block_size"] = block_size_input_contour if block_size_input_contour % 2 != 0 else block_size_input_contour + 1
            editable_params["adaptive_thresh_c"] = st.number_input(
                "輪郭 適応閾値: C値", 
                value=editable_params.get("adaptive_thresh_c", 7), step=1, 
                key="contour_trim_c_value_tuning"
            )
            editable_params["min_contour_area_ratio"] = st.number_input(
                "輪郭: 最小輪郭面積比", min_value=0.0, max_value=0.1, 
                value=editable_params.get("min_contour_area_ratio", 0.00005), 
                step=0.00001, format="%.5f", key="contour_trim_min_area_tuning"
            )
            editable_params["gaussian_blur_kernel_width"] = st.number_input(
                "輪郭 ブラー: 幅 (0で無効,奇数)", min_value=0, 
                value=editable_params.get("gaussian_blur_kernel_width", 5), step=1, 
                key="contour_trim_blur_width_tuning"
            )
            editable_params["gaussian_blur_kernel_height"] = st.number_input(
                "輪郭 ブラー: 高さ (0で無効,奇数)", min_value=0, 
                value=editable_params.get("gaussian_blur_kernel_height", 5), step=1, 
                key="contour_trim_blur_height_tuning"
            )
            st.markdown("###### 輪郭 モルフォロジー変換")
            editable_params["morph_open_apply"] = st.checkbox(
                "輪郭 オープニング", 
                value=editable_params.get("morph_open_apply", False), 
                key="contour_trim_morph_open_apply_tuning"
            )
            if editable_params.get("morph_open_apply"):
                open_kernel_contour = st.number_input(
                    "輪郭 Openカーネル(奇数)",min_value=1,
                    value=editable_params.get("morph_open_kernel_size",3),step=2,
                    key="contour_trim_morph_open_kernel_tuning")
                editable_params["morph_open_kernel_size"]=open_kernel_contour if open_kernel_contour%2!=0 else open_kernel_contour+1
                editable_params["morph_open_iterations"]=st.number_input(
                    "輪郭 Openイテレーション",min_value=1,
                    value=editable_params.get("morph_open_iterations",1),step=1,
                    key="contour_trim_morph_open_iter_tuning")
            editable_params["morph_close_apply"] = st.checkbox(
                "輪郭 クロージング", 
                value=editable_params.get("morph_close_apply", False), 
                key="contour_trim_morph_close_apply_tuning"
            )
            if editable_params.get("morph_close_apply"):
                close_kernel_contour = st.number_input(
                    "輪郭 Closeカーネル(奇数)",min_value=1,
                    value=editable_params.get("morph_close_kernel_size",3),step=2,
                    key="contour_trim_morph_close_kernel_tuning")
                editable_params["morph_close_kernel_size"]=close_kernel_contour if close_kernel_contour%2!=0 else close_kernel_contour+1
                editable_params["morph_close_iterations"]=st.number_input(
                    "輪郭 Closeイテレーション",min_value=1,
                    value=editable_params.get("morph_close_iterations",1),step=1,
                    key="contour_trim_morph_close_iter_tuning")

        # --- OCRベーストリミング設定 (新規追加) ---
        st.markdown("#### 4. OCRベーストリミング 詳細設定")
        # editable_params は輪郭とOCRのパラメータが混在している辞書をそのまま使用
        editable_params["ocr_trim_tuning_apply"] = st.checkbox( # チューニングモードでのOCRトリミング適用フラグ
            "OCRベーストリミング適用 (チューニング用)",
            value=editable_params.get("ocr_trim_tuning_apply", True), # configのデフォルト値
            key="ocr_trim_tuning_apply_checkbox"
        )
        if editable_params.get("ocr_trim_tuning_apply"):
            editable_params["ocr_trim_padding"] = st.number_input(
                "OCR: パディング (px)", min_value=0, max_value=200,
                value=editable_params.get("ocr_trim_padding", 0), step=1,
                key="ocr_trim_padding_tuning"
            )
            editable_params["ocr_trim_lang"] = st.text_input(
                "OCR: 言語 (例: eng+jpn, jpn)",
                value=editable_params.get("ocr_trim_lang", "eng+jpn"),
                key="ocr_trim_lang_tuning"
            )
            editable_params["ocr_trim_min_conf"] = st.number_input(
                "OCR: 最小信頼度 (0-100)", min_value=0, max_value=100,
                value=editable_params.get("ocr_trim_min_conf", 30), step=1,
                key="ocr_trim_min_conf_tuning"
            )
            editable_params["ocr_tesseract_config"] = st.text_input(
                "OCR: Tesseract追加設定 (例: --psm 6)",
                value=editable_params.get("ocr_tesseract_config", "--psm 6"),
                key="ocr_tesseract_config_tuning"
            )
            st.markdown("###### OCR: テキストボックスフィルタ")
            cols_ocr_filter1 = st.columns(2)
            with cols_ocr_filter1[0]:
                editable_params["ocr_trim_min_box_height"] = st.number_input(
                    "最小高さ(px)", min_value=1,
                    value=editable_params.get("ocr_trim_min_box_height", 5),
                    key="ocr_min_h_tune"
                )
                editable_params["ocr_trim_min_box_width"] = st.number_input(
                    "最小幅(px)", min_value=1,
                    value=editable_params.get("ocr_trim_min_box_width", 5),
                    key="ocr_min_w_tune"
                )
                editable_params["ocr_trim_min_aspect_ratio"] = st.number_input(
                    "最小AR (幅/高)", min_value=0.01, max_value=100.0, step=0.01, format="%.2f",
                    value=editable_params.get("ocr_trim_min_aspect_ratio", 0.1),
                    key="ocr_min_ar_tune"
                )
            with cols_ocr_filter1[1]:
                editable_params["ocr_trim_max_box_height_ratio"] = st.number_input(
                    "最大高さ比(対画像)", min_value=0.01, max_value=1.0, step=0.01, format="%.2f",
                    value=editable_params.get("ocr_trim_max_box_height_ratio", 0.3),
                    key="ocr_max_hr_tune"
                )
                editable_params["ocr_trim_max_box_width_ratio"] = st.number_input(
                    "最大幅比(対画像)", min_value=0.01, max_value=1.0, step=0.01, format="%.2f",
                    value=editable_params.get("ocr_trim_max_box_width_ratio", 0.8),
                    key="ocr_max_wr_tune"
                )
                editable_params["ocr_trim_max_aspect_ratio"] = st.number_input(
                    "最大AR (幅/高)", min_value=0.01, max_value=100.0, step=0.01, format="%.2f",
                    value=editable_params.get("ocr_trim_max_aspect_ratio", 10.0),
                    key="ocr_max_ar_tune"
                )

        # --- その他処理設定 ---
        st.markdown("#### 5. その他処理設定") # セクション番号を更新
        if "tuning_editable_other_params" not in st.session_state:
            st.warning("編集可能その他パラメータが初期化されていません。")
            return
        editable_other_params = st.session_state.tuning_editable_other_params
        if not isinstance(editable_other_params, dict):
            st.error("セッション内の 'tuning_editable_other_params' が不正な形式です。")
            return
            
        editable_other_params["grayscale"] = st.checkbox(
            "グレースケール化 (表示/Vision用)", 
            value=editable_other_params.get("grayscale", True), 
            key="other_grayscale_tuning"
        )
        editable_other_params["max_pixels"] = st.number_input(
            "リサイズ: 最大総ピクセル数 (0で無効)", min_value=0, 
            value=editable_other_params.get("max_pixels", 4000000), step=100000, 
            key="other_max_pixels_tuning",
            help="画像の幅x高さがこの値を超えないようにリサイズ。0で無効。"
        )

        if st.button(
            "この設定で画像を再処理", 
            key="reprocess_button_tuning", 
            type="primary", 
            use_container_width=True, 
            disabled=(st.session_state.get("tuning_raw_image_data") is None)
        ):
            st.session_state.tuning_trigger_reprocess = True
            print("[TuningModeUI] '画像を再処理'ボタンが押されました。")

    # --- メインエリア: 画像表示と処理結果 ---
    # アップロードされた元画像（向き補正済み）の表示
    if st.session_state.get("tuning_raw_image_data"):
        st.markdown("#### 元画像 (チューニング対象)")
        try:
            # セッションに保存されているバイトデータからPillow画像オブジェクトを生成
            img_pil_oriented_from_session = Image.open(BytesIO(st.session_state.tuning_raw_image_data))
            
            # 表示用にリサイズ（UI表示が大きくなりすぎないように）
            display_max_width = 800 
            if img_pil_oriented_from_session.width > display_max_width:
                aspect_ratio = img_pil_oriented_from_session.height / img_pil_oriented_from_session.width
                new_height = int(display_max_width * aspect_ratio)
                img_to_display = img_pil_oriented_from_session.resize((display_max_width, new_height), Image.Resampling.LANCZOS)
            else:
                img_to_display = img_pil_oriented_from_session.copy()
            
            st.image(
                img_to_display,
                caption=f"元画像 (向き補正・表示リサイズ済) | MIME: {st.session_state.get('tuning_raw_image_mime_type', 'N/A')} | サイズ: {img_pil_oriented_from_session.width}x{img_pil_oriented_from_session.height}",
                use_container_width=False # 元のアスペクト比を保つ
            )
        except Exception as e:
            st.error(f"元画像の表示準備中にエラーが発生しました: {e}")
            # エラー時も、可能であればバイトデータを直接表示試行
            st.image(
                st.session_state.tuning_raw_image_data,
                caption=f"元画像 (表示エラー) | MIME: {st.session_state.get('tuning_raw_image_mime_type', 'N/A')}",
                use_container_width=True 
            )

    # 前回の画像処理結果（輪郭ベースとOCRベース）の表示
    if "last_processing_result_tuning" in st.session_state and \
       st.session_state.last_processing_result_tuning is not None:
        
        result_data = st.session_state.last_processing_result_tuning
        st.markdown("---") # 区切り線
        st.subheader("最新のトリミング結果")
        
        cols = st.columns(2) # 2カラムレイアウトで比較表示
        with cols[0]:
            st.markdown("##### 輪郭ベーストリミング")
            if result_data.get("contour_trimmed_image_data"):
                st.image(result_data["contour_trimmed_image_data"], use_container_width=True, caption="輪郭ベースの結果")
            else:
                st.info("輪郭ベーストリミングの結果はありませんでした（または無効）。")
        with cols[1]:
            st.markdown("##### OCRベーストリミング")
            if result_data.get("ocr_trimmed_image_data"):
                st.image(result_data["ocr_trimmed_image_data"], use_container_width=True, caption="OCRベースの結果")
            else:
                st.info("OCRベーストリミングの結果はありませんでした（またはスキップ）。")
        st.markdown("---")

    # デバッグ画像群の表示
    # 選択されているパラメータセット名をタイトルに含める
    debug_display_title = f"「{st.session_state.get('selected_param_set_name_tuning', '現在の編集値')}」設定での" \
                           if st.session_state.get('selected_param_set_name_tuning') != "現在の編集値" \
                           else "現在の設定での"
    
    display_debug_images_app(
        st.session_state.get("tuning_current_debug_images", []), 
        title_prefix=debug_display_title
    )

    # --- 画像処理ロジック (チューニングモード用) ---
    should_reprocess_now = False # この実行サイクルで再処理を行うかどうかのフラグ

    # 1. 新しい画像がアップロードされた場合の処理
    if uploaded_file_obj_tune:
        new_uploaded_bytes = uploaded_file_obj_tune.getvalue()
        # 現在セッションに保存されている画像データと比較し、変更があれば処理
        # (向き補正後のバイトデータで比較するため、一度向き補正を行う)
        try:
            img_pil_uploaded = Image.open(BytesIO(new_uploaded_bytes))
            print("[TuningModeUI] 新しい画像がアップロードされました。OpenCVによる自動向き補正を試みます...")
            img_pil_auto_oriented = auto_orient_image_opencv(img_pil_uploaded) # OpenCVベースの向き補正
            
            # 向き補正後の画像をバイトデータに変換してセッション保存用とする
            # MIMEタイプに応じて保存フォーマットを決定
            original_mime_type = uploaded_file_obj_tune.type.lower()
            session_image_format: str = "PNG" # デフォルトはPNG
            if original_mime_type == "image/jpeg":
                session_image_format = "JPEG"
            elif original_mime_type == "image/webp": # WebPがPillowでサポートされていればWebP、そうでなければPNG
                try:
                    Image.open(BytesIO(b'')).save(BytesIO(), format='WEBP') # WebP保存テスト
                    session_image_format = "WEBP"
                except Exception:
                    session_image_format = "PNG"
            
            jpeg_quality_for_session = st.session_state.get("tuning_fixed_other_params", {}).get("jpeg_quality", 85)
            oriented_bytes_for_session_storage = image_to_bytes(
                img_pil_auto_oriented, 
                target_format=session_image_format, # type: ignore
                jpeg_quality=jpeg_quality_for_session
            )
            
            if oriented_bytes_for_session_storage:
                # 向き補正後のデータが現在のデータと異なる場合、またはデータがまだない場合
                if st.session_state.get("tuning_raw_image_data") is None or \
                   oriented_bytes_for_session_storage != st.session_state.tuning_raw_image_data:
                    
                    st.session_state.tuning_raw_image_data = oriented_bytes_for_session_storage
                    st.session_state.tuning_raw_image_mime_type = uploaded_file_obj_tune.type # 元のMIMEタイプを保存
                    st.session_state.tuning_current_debug_images = [] # デバッグ画像をクリア
                    should_reprocess_now = True # 新しい画像なので再処理
                    # 新しい画像がアップロードされたら、パラメータは「固定値(推奨)」に戻す
                    st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
                    if "tuning_fixed_cv_params" in st.session_state:
                        st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
                    if "tuning_fixed_other_params" in st.session_state:
                         st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
                    print("[TuningModeUI] 新規（または変更された）画像データ（向き補正済）をセッションに保存し、再処理をスケジュールしました。パラメータは推奨値にリセット。")
            else:
                # 向き補正後のバイト変換に失敗した場合のフォールバック (ほぼ起こらないはずだが念のため)
                st.warning("画像の自動向き補正後のバイトデータ変換に失敗しました。元の未補正画像で処理を試みます。")
                if st.session_state.get("tuning_raw_image_data") is None or \
                   new_uploaded_bytes != st.session_state.tuning_raw_image_data:
                    st.session_state.tuning_raw_image_data = new_uploaded_bytes # 元のバイト列を保存
                    # (同様のセッション更新と再処理スケジューリング)
        except Exception as e_orientation:
            st.error(f"画像アップロード時の自動向き補正処理でエラーが発生しました: {e_orientation}")
            # エラー発生時は、元のバイトデータをそのまま使用して処理を試みる
            if st.session_state.get("tuning_raw_image_data") is None or \
               new_uploaded_bytes != st.session_state.tuning_raw_image_data:
                st.session_state.tuning_raw_image_data = new_uploaded_bytes
                # (同様のセッション更新と再処理スケジューリング)

    # 2. 「再処理ボタン」が押された場合の処理
    if st.session_state.get("tuning_trigger_reprocess"):
        should_reprocess_now = True
        st.session_state.tuning_trigger_reprocess = False # フラグを消費
        print("[TuningModeUI] '画像を再処理'ボタンにより再処理がトリガーされました。")

    # 3. 実際の再処理実行
    if should_reprocess_now and st.session_state.get("tuning_raw_image_data"):
        print(f"[TuningModeUI] 画像再処理開始。パラメータ:\n"
              f"  編集中のCV/OCR関連: {st.session_state.get('tuning_editable_cv_params')}\n"
              f"  編集中のその他: {st.session_state.get('tuning_editable_other_params')}")
        
        current_editable_cv_ocr_params = st.session_state.get("tuning_editable_cv_params", {})
        current_editable_other_params = st.session_state.get("tuning_editable_other_params", {})
        
        # image_processorに渡すパラメータには、UIで編集された全てのCV/OCR関連パラメータを含める。
        # 'apply' (輪郭ベース用) と 'ocr_trim_tuning_apply' (OCRベース用) もそのまま渡す。
        # image_processor側でこれらのフラグを見て処理を分岐する。
        params_for_processor = current_editable_cv_ocr_params.copy()

        with st.spinner("チューニングモードで画像処理を実行中..."):
            processing_result = preprocess_uploaded_image(
                uploaded_file_data=st.session_state.tuning_raw_image_data, 
                mime_type=st.session_state.tuning_raw_image_mime_type,
                max_pixels=current_editable_other_params.get("max_pixels", 0),
                grayscale=current_editable_other_params.get("grayscale", True),
                output_format=st.session_state.get("tuning_fixed_other_params", {}).get("output_format", "JPEG"),
                jpeg_quality=st.session_state.get("tuning_fixed_other_params", {}).get("jpeg_quality", 85),
                # apply_trimming_opencv_override は輪郭ベーストリミングの適用フラグとして利用
                apply_trimming_opencv_override=params_for_processor.get("apply"), 
                # trim_params_override にはOCR関連のフラグやパラメータも全て含めて渡す
                trim_params_override=params_for_processor 
            )
        
        st.session_state.last_processing_result_tuning = processing_result
        if processing_result and "error" not in processing_result:
            st.session_state.tuning_current_debug_images = processing_result.get("debug_images", [])
            print("[TuningModeUI] 画像処理完了。")
        elif processing_result and "error" in processing_result:
            st.error(f"画像処理エラー (チューニング): {processing_result['error']}")
            st.session_state.tuning_current_debug_images = processing_result.get("debug_images", [])
            print(f"[TuningModeUI] 画像処理エラー: {processing_result['error']}")
        else:
            st.error("画像処理で予期せぬエラー (チューニング)。結果返されず。")
            st.session_state.tuning_current_debug_images = []
            print("[TuningModeUI] 画像処理で予期せぬエラー（結果がNone）。")
        st.rerun()