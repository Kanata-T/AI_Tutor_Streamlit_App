# ui/tuning_mode_ui.py
import streamlit as st
import copy # パラメータコピーのため
from PIL import Image  # 画像リサイズ用に追加
from io import BytesIO  # バイトデータから画像を開くため追加

# utilsモジュールのインポート
from utils.image_processor import preprocess_uploaded_image
from utils.image_processing_parts.base_image_utils import correct_image_orientation, image_to_bytes, auto_orient_image_opencv  # auto_orient_image_opencv を追加

# uiヘルパーのインポート (もしチューニングモードでも共通表示部品を使うなら)
from .display_helpers import display_debug_images_app # チューニングモードではデバッグ画像表示に使う

def render_tuning_mode():
    """画像処理チューニングモードのUIとロジックをレンダリングします。"""
    st.title("🖼️ OpenCV 画像トリミング パラメータチューナー")

    # 画像チューニングモード専用のセッション状態初期化 (app.pyから移動)
    # この初期化は app.py のモード切り替え時に tuning_initialized = False とすることで再実行される
    if not st.session_state.get("tuning_initialized", False):
        st.session_state.tuning_raw_image_data = None
        st.session_state.tuning_raw_image_mime_type = None
        st.session_state.tuning_current_debug_images = []
        st.session_state.tuning_image_key_counter = st.session_state.get("tuning_image_key_counter", 0) # 既存値を維持
        st.session_state.tuning_trigger_reprocess = False
        
        # チューニングUI用の編集可能パラメータ (固定値をコピーして開始)
        # tuning_fixed_cv_params, tuning_fixed_other_params は app.py の共通初期化で設定済みのはず
        if "tuning_editable_cv_params" not in st.session_state or not st.session_state.tuning_initialized:
             # 固定値が存在することを確認してからコピー
            if "tuning_fixed_cv_params" in st.session_state:
                st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
            else: # フォールバック (通常は発生しないはず)
                st.session_state.tuning_editable_cv_params = {} 
                st.warning("tuning_fixed_cv_params がセッションにありません。デフォルト値をロードできませんでした。")

        if "tuning_editable_other_params" not in st.session_state or not st.session_state.tuning_initialized:
            if "tuning_fixed_other_params" in st.session_state:
                st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
            else:
                st.session_state.tuning_editable_other_params = {}
                st.warning("tuning_fixed_other_params がセッションにありません。デフォルト値をロードできませんでした。")

        if "selected_param_set_name_tuning" not in st.session_state or not st.session_state.tuning_initialized:
            st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
            
        st.session_state.tuning_initialized = True # 初期化完了フラグ
        st.session_state.tutor_initialized = False # 他のモードの初期化フラグをリセット
        print("[TuningModeUI] Image Tuning mode specific state initialized via render_tuning_mode.")
        # ここで st.rerun() は不要。UI要素の表示に必要な状態が整う。

    # --- 画像処理チューニングモードのUIとロジック ---
    with st.sidebar:
        st.header("⚙️ 処理パラメータ (チューニング)")
        st.markdown("#### 1. 画像を選択")
        uploaded_file_obj_tune_complete_v2 = st.file_uploader(
            "調整対象の画像", type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
            key=f"tuning_file_uploader_{st.session_state.get('tuning_image_key_counter', 0)}_ui_module" # キー変更
        )
        if st.button("現在の画像をクリア", key="clear_image_button_tune_ui_module"): # キー変更
            st.session_state.tuning_raw_image_data = None
            st.session_state.tuning_raw_image_mime_type = None
            st.session_state.tuning_current_debug_images = []
            st.session_state.tuning_image_key_counter = st.session_state.get('tuning_image_key_counter', 0) + 1
            # パラメータも固定値に戻す
            if "tuning_fixed_cv_params" in st.session_state:
                st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
            if "tuning_fixed_other_params" in st.session_state:
                st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
            st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
            st.session_state.tuning_trigger_reprocess = False
            st.rerun()

        st.markdown("#### 2. パラメータセット提案")
        # プリセット定義 (st.session_state.tuning_fixed_cv_params が初期化されている前提)
        fixed_cv_params_for_presets = st.session_state.get("tuning_fixed_cv_params", {}) # 安全に取得
        all_presets_tune_complete_v2 = {
            "現在の編集値": {}, # このキーは特別扱い
            "00: 固定値 (推奨)": fixed_cv_params_for_presets,
            "01: 基本 (ブラーなし)": {"apply":True, "padding":15, "adaptive_thresh_block_size":11, "adaptive_thresh_c":7, "min_contour_area_ratio":0.0005, "gaussian_blur_kernel_width":0, "gaussian_blur_kernel_height":0, "morph_open_apply":False, "morph_close_apply":False, "haar_apply":False, "h_proj_apply":False},
            "02: 小さい文字 (Block小,C調整,Open有)": {"apply":True, "padding":10, "adaptive_thresh_block_size":7, "adaptive_thresh_c":3, "min_contour_area_ratio":0.0001, "gaussian_blur_kernel_width":0, "gaussian_blur_kernel_height":0, "morph_open_apply":True, "morph_open_kernel_size":2, "morph_open_iterations":1, "morph_close_apply":False, "haar_apply":True, "h_proj_apply":True},
        }

        selected_set_name_tune_ui_v2 = st.selectbox(
            "パラメータセット:", 
            list(all_presets_tune_complete_v2.keys()), 
            key="param_set_tune_select_ui_module", # キー変更
            index=list(all_presets_tune_complete_v2.keys()).index(st.session_state.get("selected_param_set_name_tuning", "00: 固定値 (推奨)"))
        )
        if selected_set_name_tune_ui_v2 != st.session_state.get("selected_param_set_name_tuning"):
            st.session_state.selected_param_set_name_tuning = selected_set_name_tune_ui_v2
            if selected_set_name_tune_ui_v2 != "現在の編集値" and selected_set_name_tune_ui_v2 in all_presets_tune_complete_v2:
                # 固定値をベースに、選択されたプリセットの値をマージする
                new_editable_params_v2 = st.session_state.get("tuning_fixed_cv_params", {}).copy()
                preset_values = all_presets_tune_complete_v2[selected_set_name_tune_ui_v2]
                new_editable_params_v2.update(preset_values) # プリセットの値で上書き
                st.session_state.tuning_editable_cv_params = new_editable_params_v2
            st.rerun() # パラメータセット変更時は再描画してUIに反映
        
        st.markdown("#### 3. OpenCV トリミング詳細設定")
        # editable_cv_tune_v2 はセッション状態から直接参照・更新
        if "tuning_editable_cv_params" not in st.session_state: # 安全対策
            st.warning("編集可能CVパラメータが初期化されていません。")
            return # これ以上進めない

        editable_cv_tune_v2 = st.session_state.tuning_editable_cv_params
        
        # editable_cv_tune_v2 が辞書であることを確認
        if not isinstance(editable_cv_tune_v2, dict):
            st.error("tuning_editable_cv_params が不正な形式です。")
            return

        editable_cv_tune_v2["apply"] = st.checkbox("OpenCVトリミング適用", value=editable_cv_tune_v2.get("apply", True), key="cv_apply_tune_ui_module") # キー変更
        if editable_cv_tune_v2.get("apply"):
            editable_cv_tune_v2["padding"] = st.number_input("パディング", 0, 200, editable_cv_tune_v2.get("padding", 0), 1, key="cv_pad_num_ui_module")
            
            bs_val_v2 = st.number_input("適応閾値ブロック(3以上奇数)", 3, value=editable_cv_tune_v2.get("adaptive_thresh_block_size", 11), step=2, key="cv_block_num_ui_module")
            editable_cv_tune_v2["adaptive_thresh_block_size"] = bs_val_v2 if bs_val_v2 % 2 != 0 else bs_val_v2 + 1
            
            editable_cv_tune_v2["adaptive_thresh_c"] = st.number_input("適応閾値C", value=editable_cv_tune_v2.get("adaptive_thresh_c", 7), step=1, key="cv_c_num_ui_module")
            editable_cv_tune_v2["min_contour_area_ratio"] = st.number_input("最小輪郭面積比", 0.0, 0.1, editable_cv_tune_v2.get("min_contour_area_ratio", 0.00005), 0.00001, "%.5f", key="cv_area_num_ui_module")
            
            editable_cv_tune_v2["gaussian_blur_kernel_width"] = st.number_input("ブラー幅(0で無効,奇数)", 0, value=editable_cv_tune_v2.get("gaussian_blur_kernel_width", 5), step=1, key="cv_blurw_num_ui_module")
            editable_cv_tune_v2["gaussian_blur_kernel_height"] = st.number_input("ブラー高さ(0で無効,奇数)", 0, value=editable_cv_tune_v2.get("gaussian_blur_kernel_height", 5), step=1, key="cv_blurh_num_ui_module")
            
            st.markdown("###### モルフォロジー変換")
            editable_cv_tune_v2["morph_open_apply"] = st.checkbox("オープニング", value=editable_cv_tune_v2.get("morph_open_apply", False), key="cv_mopen_cb_ui_module")
            if editable_cv_tune_v2.get("morph_open_apply"):
                k_op_v2=st.number_input("Openカーネル(奇数)",1,value=editable_cv_tune_v2.get("morph_open_kernel_size",3),step=2,key="cv_mopenk_num_ui_module")
                editable_cv_tune_v2["morph_open_kernel_size"]=k_op_v2 if k_op_v2%2!=0 else k_op_v2+1
                editable_cv_tune_v2["morph_open_iterations"]=st.number_input("Openイテレーション",1,value=editable_cv_tune_v2.get("morph_open_iterations",1),step=1,key="cv_mopeni_num_ui_module")

            editable_cv_tune_v2["morph_close_apply"] = st.checkbox("クロージング", value=editable_cv_tune_v2.get("morph_close_apply", False), key="cv_mclose_cb_ui_module")
            if editable_cv_tune_v2.get("morph_close_apply"):
                k_cl_v2=st.number_input("Closeカーネル(奇数)",1,value=editable_cv_tune_v2.get("morph_close_kernel_size",3),step=2,key="cv_mclosek_num_ui_module")
                editable_cv_tune_v2["morph_close_kernel_size"]=k_cl_v2 if k_cl_v2%2!=0 else k_cl_v2+1
                editable_cv_tune_v2["morph_close_iterations"]=st.number_input("Closeイテレーション",1,value=editable_cv_tune_v2.get("morph_close_iterations",1),step=1,key="cv_mclosei_num_ui_module")

            with st.expander("補助的トリミング (実験的)", expanded=False):
                editable_cv_tune_v2["haar_apply"]=st.checkbox("Haar-Like",value=editable_cv_tune_v2.get("haar_apply", True),key="cv_haar_cb_ui_module")
                if editable_cv_tune_v2.get("haar_apply"):
                    hrh_v2=st.number_input("Haarマスク高(偶数)",4,100,editable_cv_tune_v2.get("haar_rect_h",22),2,key="cv_haarrh_num_ui_module")
                    editable_cv_tune_v2["haar_rect_h"]=hrh_v2 if hrh_v2%2==0 else hrh_v2+1
                    editable_cv_tune_v2["haar_peak_threshold"]=st.number_input("Haarピーク閾値",0.0,100.0,editable_cv_tune_v2.get("haar_peak_threshold",7.0),0.5,"%.1f",key="cv_haarpt_num_ui_module")
                
                editable_cv_tune_v2["h_proj_apply"]=st.checkbox("水平射影",value=editable_cv_tune_v2.get("h_proj_apply", True),key="cv_hproj_cb_ui_module")
                if editable_cv_tune_v2.get("h_proj_apply"):
                    editable_cv_tune_v2["h_proj_threshold_ratio"]=st.number_input("水平射影閾値比",0.001,0.5,editable_cv_tune_v2.get("h_proj_threshold_ratio",0.15),0.001,"%.3f",key="cv_hprojtr_num_ui_module")

        st.markdown("#### 4. その他処理設定")
        if "tuning_editable_other_params" not in st.session_state: # 安全対策
            st.warning("編集可能その他パラメータが初期化されていません。")
            return

        editable_other_tune_v2 = st.session_state.tuning_editable_other_params
        if not isinstance(editable_other_tune_v2, dict):
            st.error("tuning_editable_other_params が不正な形式です。")
            return
            
        editable_other_tune_v2["grayscale"] = st.checkbox("グレースケール化", value=editable_other_tune_v2.get("grayscale", True), key="grayscale_tune_ui_module") # キー変更
        editable_other_tune_v2["max_pixels"] = st.number_input("リサイズ最大ピクセル(0で無効)", 0, value=editable_other_tune_v2.get("max_pixels", 4000000), step=100000, key="maxpix_tune_ui_module") # キー変更

        if st.button("この設定で画像を再処理", key="reprocess_tune_btn_ui_module", type="primary", use_container_width=True, disabled=(st.session_state.get("tuning_raw_image_data") is None)): # キー変更
            st.session_state.tuning_trigger_reprocess = True
            # UIからのパラメータ変更は即座にセッション状態に反映されているので、ここで st.rerun() は不要。
            # tuning_trigger_reprocess フラグを立てるだけで、次の実行サイクルで再処理ロジックが動く。
    
    # メインエリア表示 (チューニングモード)
    if st.session_state.get("tuning_raw_image_data"):  # このデータは既に向き補正済みのはず
        st.markdown("#### 元画像 (チューニング対象):")
        try:
            img_pil_oriented_from_session = Image.open(BytesIO(st.session_state.tuning_raw_image_data))
            display_max_width = 800
            if img_pil_oriented_from_session.width > display_max_width:
                aspect_ratio = img_pil_oriented_from_session.height / img_pil_oriented_from_session.width
                new_height = int(display_max_width * aspect_ratio)
                img_display_resized = img_pil_oriented_from_session.resize((display_max_width, new_height), Image.Resampling.LANCZOS)
            else:
                img_display_resized = img_pil_oriented_from_session.copy()
            st.image(
                img_display_resized,
                caption=f"元画像 (向き補正・表示リサイズ済) - MIME: {st.session_state.get('tuning_raw_image_mime_type', 'N/A')}, Oriented Size: {img_pil_oriented_from_session.width}x{img_pil_oriented_from_session.height}",
                use_container_width=False
            )
        except Exception as e:
            st.error(f"元画像の表示準備中にエラーが発生しました: {e}")
            st.image(
                st.session_state.tuning_raw_image_data,
                caption=f"元画像 (表示エラー) - MIME: {st.session_state.get('tuning_raw_image_mime_type', 'N/A')}",
                use_container_width=True
            )

    # --- ▼ 両方のトリミング結果を表示 ▼ ---
    if "last_processing_result_tuning" in st.session_state:
        result_data = st.session_state.last_processing_result_tuning
        cols = st.columns(2)
        with cols[0]:
            st.markdown("##### 輪郭ベーストリミング結果:")
            if result_data.get("contour_trimmed_image_data"):
                st.image(result_data["contour_trimmed_image_data"], use_container_width=True, caption="輪郭ベース")
            else:
                st.info("輪郭ベーストリミング結果なし")
        with cols[1]:
            st.markdown("##### OCRベーストリミング結果:")
            if result_data.get("ocr_trimmed_image_data"):
                st.image(result_data["ocr_trimmed_image_data"], use_container_width=True, caption="OCRベース")
            else:
                st.info("OCRベーストリミング結果なし (またはスキップ)")
    # --- ▲ ここまで ▲ ---
    
    display_title_tune_complete_v2 = f"「{st.session_state.get('selected_param_set_name_tuning', '現在の編集値')}」設定での " \
                                     if st.session_state.get('selected_param_set_name_tuning') != "現在の編集値" else "現在の設定での "
    
    # display_debug_images_app は ui.display_helpers からインポートして使用
    display_debug_images_app(st.session_state.get("tuning_current_debug_images", []), title_prefix=display_title_tune_complete_v2)

    # チューニング用のロジック (画像アップロード時と再処理ボタン押下時の処理)
    should_reprocess_tuning_final_v2 = False
    # 新しい画像がアップロードされたか、または既存の画像と異なる場合
    if uploaded_file_obj_tune_complete_v2:
        new_image_bytes_raw = uploaded_file_obj_tune_complete_v2.getvalue()
        try:
            img_pil_temp = Image.open(BytesIO(new_image_bytes_raw))
            # OpenCVベースの自動向き補正を適用
            print("[TuningModeUI] Attempting OpenCV auto orientation...")
            img_pil_auto_oriented = auto_orient_image_opencv(img_pil_temp)
            original_mime = uploaded_file_obj_tune_complete_v2.type.lower()
            output_fmt_for_session: str = "PNG"  # デフォルト
            if original_mime == "image/jpeg":
                output_fmt_for_session = "JPEG"
            elif original_mime == "image/webp":
                try:
                    Image.open(BytesIO(b'')).save(BytesIO(), format='WEBP')
                    output_fmt_for_session = "WEBP"
                except Exception:
                    output_fmt_for_session = "PNG"
            jpeg_q = st.session_state.get("tuning_fixed_other_params", {}).get("jpeg_quality", 85)
            oriented_bytes_for_session = image_to_bytes(img_pil_auto_oriented, target_format=output_fmt_for_session, jpeg_quality=jpeg_q)  # type: ignore
            if oriented_bytes_for_session:
                if st.session_state.get("tuning_raw_image_data") is None or oriented_bytes_for_session != st.session_state.tuning_raw_image_data:
                    st.session_state.tuning_raw_image_data = oriented_bytes_for_session
                    st.session_state.tuning_raw_image_mime_type = uploaded_file_obj_tune_complete_v2.type
                    st.session_state.tuning_current_debug_images = []
                    should_reprocess_tuning_final_v2 = True
                    st.session_state.selected_param_set_name_tuning = "現在の編集値"
                    if "tuning_fixed_cv_params" in st.session_state:
                        st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
                    if "tuning_fixed_other_params" in st.session_state:
                        st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
                    print("[TuningModeUI] New (or different auto-oriented) image data stored, set to reprocess.")
            else:
                st.warning("画像の自動向き補正後のバイトデータ変換に失敗しました。")
                if st.session_state.get("tuning_raw_image_data") is None or new_image_bytes_raw != st.session_state.tuning_raw_image_data:
                    st.session_state.tuning_raw_image_data = new_image_bytes_raw
        except Exception as e_orient:
            st.error(f"画像アップロード時の自動向き補正処理でエラー: {e_orient}")
            if st.session_state.get("tuning_raw_image_data") is None or new_image_bytes_raw != st.session_state.tuning_raw_image_data:
                st.session_state.tuning_raw_image_data = new_image_bytes_raw

    if st.session_state.get("tuning_trigger_reprocess"):
        should_reprocess_tuning_final_v2 = True
        st.session_state.tuning_trigger_reprocess = False # フラグを消費
        print("[TuningModeUI] Reprocess triggered by button.")

    if should_reprocess_tuning_final_v2 and st.session_state.get("tuning_raw_image_data"):
        print(f"[TuningModeUI] Reprocessing with: CV={st.session_state.get('tuning_editable_cv_params')}, Other={st.session_state.get('tuning_editable_other_params')}")
        
        editable_cv_p_final_v2 = st.session_state.get("tuning_editable_cv_params", {})
        editable_o_p_final_v2 = st.session_state.get("tuning_editable_other_params", {})
        
        # preprocess_uploaded_image に渡す trim_params は 'apply' を含まない想定が元コードにあったか確認
        # 元コード: params_to_pass_tune_final = {k:v for k,v in editable_cv_p_final_v2.items() if k != "apply"}
        # image_processor.py の preprocess_uploaded_image が apply_trimming_opencv_override と trim_params_override をどう使うかによる
        # ここでは、元のロジックを尊重し、'apply'を除外する
        params_to_pass_tune_final = {k: v for k, v in editable_cv_p_final_v2.items() if k != "apply"}

        with st.spinner("チューニング画像処理中..."):
            res_tune_final_v2 = preprocess_uploaded_image(
                uploaded_file_data=st.session_state.tuning_raw_image_data, 
                mime_type=st.session_state.tuning_raw_image_mime_type,
                max_pixels=editable_o_p_final_v2.get("max_pixels", 0), # .getで安全に
                grayscale=editable_o_p_final_v2.get("grayscale", True),
                # output_format, jpeg_quality はチューニングUIにはないので固定値を使用するか、
                # st.session_state.tuning_fixed_other_params から持ってくる
                output_format=st.session_state.get("tuning_fixed_other_params", {}).get("output_format", "JPEG"),
                jpeg_quality=st.session_state.get("tuning_fixed_other_params", {}).get("jpeg_quality", 85),
                apply_trimming_opencv_override=editable_cv_p_final_v2.get("apply", True), # .getで安全に
                trim_params_override=params_to_pass_tune_final 
            )
        if res_tune_final_v2 and "error" not in res_tune_final_v2:
            st.session_state.tuning_current_debug_images = res_tune_final_v2.get("debug_images", [])
        elif res_tune_final_v2 and "error" in res_tune_final_v2:
            st.error(f"画像処理エラー(Tune): {res_tune_final_v2['error']}")
            st.session_state.tuning_current_debug_images = res_tune_final_v2.get("debug_images", []) # エラー時もデバッグ画像があれば表示
        else:
            st.error("画像処理で予期せぬエラー(Tune)が発生しました。")
            st.session_state.tuning_current_debug_images = []
        st.rerun() # 処理結果をUIに反映させるために再実行