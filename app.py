# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import copy # deepcopyのため

# coreモジュールのインポート (state_managerはデバッグ表示で一部使用)
from core import state_manager

# utilsモジュールのインポート
from utils.config_loader import get_config # get_image_processing_config から汎用的な名前に変更を想定

# uiモジュールのインポート
from ui.tutor_mode_ui import render_tutor_mode
from ui.tuning_mode_ui import render_tuning_mode
from ui.display_helpers import display_debug_images_app

# --- 初期設定 ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini APIキーが設定されていません。.envファイルを確認してください。")
    st.stop()
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Gemini APIの設定に失敗しました: {e}")
    st.stop()

# --- Streamlit UI設定 ---
st.set_page_config(page_title="AI学習チューター プロトタイプ", layout="wide")

# --- セッション状態の基本初期化 ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "AIチューター" # デフォルトモード

if "tutor_initialized" not in st.session_state:
    st.session_state.tutor_initialized = False
if "tuning_initialized" not in st.session_state:
    st.session_state.tuning_initialized = False

# --- アプリケーション全体で共通の初期化 (初回のみ実行) ---
if "common_params_initialized" not in st.session_state:
    app_config = get_config() # config.yaml 全体をロード
    
    # 画像処理関連の設定をロード
    img_proc_config = app_config.get("image_processing", {})
    cv_trim_config = img_proc_config.get("opencv_trimming", {})
    
    # --- ここで app_config をセッションに保存 ---
    st.session_state.app_config = app_config
    
    # AIチューターモードおよびチューニングモードの「固定/デフォルト」パラメータをconfigからロード
    # OpenCVトリミングパラメータ
    st.session_state.tuning_fixed_cv_params = {
        "apply": cv_trim_config.get("apply", True),
        "padding": cv_trim_config.get("padding", 0),
        "adaptive_thresh_block_size": cv_trim_config.get("adaptive_thresh_block_size", 11),
        "adaptive_thresh_c": cv_trim_config.get("adaptive_thresh_c", 7),
        "min_contour_area_ratio": cv_trim_config.get("min_contour_area_ratio", 0.00005),
        "gaussian_blur_kernel_width": cv_trim_config.get("gaussian_blur_kernel_width", 5),
        "gaussian_blur_kernel_height": cv_trim_config.get("gaussian_blur_kernel_height", 5),
        "morph_open_apply": cv_trim_config.get("morph_open_apply", False),
        "morph_open_kernel_size": cv_trim_config.get("morph_open_kernel_size", 3),
        "morph_open_iterations": cv_trim_config.get("morph_open_iterations", 1),
        "morph_close_apply": cv_trim_config.get("morph_close_apply", False),
        "morph_close_kernel_size": cv_trim_config.get("morph_close_kernel_size", 3),
        "morph_close_iterations": cv_trim_config.get("morph_close_iterations", 1),
        "haar_apply": cv_trim_config.get("haar_apply", False), # config.yaml で True に更新推奨
        "haar_rect_h": cv_trim_config.get("haar_rect_h", 22), # config.yaml で 22 に更新推奨
        "haar_peak_threshold": cv_trim_config.get("haar_peak_threshold", 7.0), # config.yaml で 7.0 に更新推奨
        "h_proj_apply": cv_trim_config.get("h_proj_apply", False), # config.yaml で True に更新推奨
        "h_proj_threshold_ratio": cv_trim_config.get("h_proj_threshold_ratio", 0.15), # config.yaml で 0.15 に更新推奨
        # --- ▼ OCRベーストリミング関連のパラメータを config から読み込み ▼ ---
        # "ocr_trim_apply_as_fallback": cv_trim_config.get("ocr_trim_apply_as_fallback", True),
        "ocr_trim_padding": cv_trim_config.get("ocr_trim_padding", 0),
        "ocr_trim_lang": cv_trim_config.get("ocr_trim_lang", "eng+jpn"),
        "ocr_trim_min_conf": cv_trim_config.get("ocr_trim_min_conf", 25),
        # --- ▼ OCRトリミングのフィルタリング用パラメータ読み込み ▼ ---
        "ocr_trim_min_box_height": cv_trim_config.get("ocr_trim_min_box_height", 5),
        "ocr_trim_max_box_height_ratio": cv_trim_config.get("ocr_trim_max_box_height_ratio", 0.3),
        "ocr_trim_min_box_width": cv_trim_config.get("ocr_trim_min_box_width", 5),
        "ocr_trim_max_box_width_ratio": cv_trim_config.get("ocr_trim_max_box_width_ratio", 0.8),
        "ocr_trim_min_aspect_ratio": cv_trim_config.get("ocr_trim_min_aspect_ratio", 0.05),
        "ocr_trim_max_aspect_ratio": cv_trim_config.get("ocr_trim_max_aspect_ratio", 20.0),
        "ocr_tesseract_config": cv_trim_config.get("ocr_tesseract_config", "--psm 6"),
        # --- ▲ ここまで追加 ▲ ---
    }
    
    # その他の画像処理パラメータ
    st.session_state.tuning_fixed_other_params = {
        "grayscale": img_proc_config.get("apply_grayscale", True),
        "output_format": img_proc_config.get("default_output_format", "JPEG"),
        "jpeg_quality": img_proc_config.get("default_jpeg_quality", 85),
        "max_pixels": img_proc_config.get("default_max_pixels_for_resizing", 4000000),
    }
    
    st.session_state.common_params_initialized = True
    # デバッグ用に初期化されたパラメータを表示
    print("[App] Common parameters (including image processing fixed params from config) initialized.")
    print(f"[App] Initial tuning_fixed_cv_params: {st.session_state.tuning_fixed_cv_params}")
    print(f"[App] Initial tuning_fixed_other_params: {st.session_state.tuning_fixed_other_params}")

# --- モードごとの初期化フラグ管理 (各render関数内で実際の初期化処理) ---
# モードが変更された場合、または初めてそのモードが選択された場合に、
# 対応するrender関数内で初期化が行われるように、ここではフラグのみを管理。
# 実際の初期化処理 (例: state_manager.initialize_session_state()) は各render関数冒頭で行う。

if st.session_state.app_mode == "AIチューター" and not st.session_state.tutor_initialized:
    print(f"[App] AI Tutor mode selected. Tutor needs initialization (current flag: {st.session_state.tutor_initialized}).")
    # render_tutor_mode()内で st.session_state.tutor_initialized = True となる

elif st.session_state.app_mode == "画像処理チューニング" and not st.session_state.tuning_initialized:
    print(f"[App] Image Tuning mode selected. Tuning needs initialization (current flag: {st.session_state.tuning_initialized}).")
    # render_tuning_mode()内で st.session_state.tuning_initialized = True となる

# --- モード選択UI (サイドバー) ---
current_app_mode = st.session_state.app_mode
selected_mode_in_sidebar = st.sidebar.radio(
    "アプリケーションモードを選択:",
    ("AIチューター", "画像処理チューニング"),
    key="app_mode_selector_radio", # シンプルなキーに変更
    index=0 if current_app_mode == "AIチューター" else 1
)

if selected_mode_in_sidebar != current_app_mode:
    st.session_state.app_mode = selected_mode_in_sidebar
    # モード変更時に、新しいモードの初期化フラグをFalseにリセットして再初期化を促す
    if st.session_state.app_mode == "AIチューター":
        st.session_state.tutor_initialized = False
        print("[App] Switched to AI Tutor mode. Reset tutor_initialized flag.")
    elif st.session_state.app_mode == "画像処理チューニング":
        st.session_state.tuning_initialized = False
        print("[App] Switched to Image Tuning mode. Reset tuning_initialized flag.")
    st.rerun() # モード変更をUIに即時反映させる

# --- メインコンテンツエリアのレンダリング ---
if st.session_state.app_mode == "AIチューター":
    render_tutor_mode()
elif st.session_state.app_mode == "画像処理チューニング":
    render_tuning_mode()

# --- サイドバー下部の共通デバッグ情報 ---
with st.sidebar:
    st.markdown("---")
    if st.session_state.app_mode == "AIチューター":
        st.header("デバッグ情報 (AIチューター)")
        st.write(f"現在のステップ: `{state_manager.get_current_step()}`") # state_manager は tutor_mode_ui で主に管理
        st.write(f"処理中: `{st.session_state.get('processing', False)}`")
        st.write(f"曖昧フラグ: `{st.session_state.get('is_request_ambiguous', False)}`")
        st.write(f"明確化試行: `{st.session_state.get('clarification_attempts', 0)}`")
        
        # AIチューターモードでの画像処理デバッグ表示チェックボックス
        show_debug_tutor = st.checkbox(
            "画像処理デバッグ表示 (AIチューター実行時)", 
            value=st.session_state.get("show_img_debug_in_tutor_mode", False), # キー名を短縮・統一
            key="show_img_debug_tutor_cb"
        )
        st.session_state.show_img_debug_in_tutor_mode = show_debug_tutor

        if st.session_state.show_img_debug_in_tutor_mode and \
           st.session_state.get("last_debug_images_tutor_run_final_v3"): # このキー名は tutor_mode_ui.py で設定されるもの
            with st.expander("前回の画像処理詳細 (AIチューター)", expanded=False):
                display_debug_images_app(
                    st.session_state.last_debug_images_tutor_run_final_v3, 
                    title_prefix="固定パラメータでの"
                )
        
        with st.expander("セッションステート抜粋 (AIチューター)", expanded=False):
            # 表示するキーをフィルタリング (チューニング関連と巨大データを除外)
            tutor_ss_display = {
                k: v for k, v in st.session_state.items() 
                if not k.startswith("tuning_") and \
                   not k.startswith("editable_") and \
                   k not in ["common_params_initialized", "app_mode_selector_radio", 
                              "show_img_debug_tutor_cb", "last_debug_images_tutor_run_final_v3",
                              "tuning_fixed_cv_params", "tuning_fixed_other_params"] # これらは別で表示/管理
            }
            if "uploaded_file_data" in tutor_ss_display and tutor_ss_display["uploaded_file_data"] is not None:
                if isinstance(tutor_ss_display["uploaded_file_data"], dict):
                     tutor_ss_display["uploaded_file_data"] = {
                        k_i: (f"<bytes {len(v_i)}>" if k_i == "data" and isinstance(v_i, bytes) else v_i)
                        for k_i, v_i in tutor_ss_display["uploaded_file_data"].items()
                    }
                else:
                    tutor_ss_display["uploaded_file_data"] = "<Invalid format>"

            if "messages" in tutor_ss_display and isinstance(tutor_ss_display["messages"], list):
                tutor_ss_display["messages"] = f"<List of {len(tutor_ss_display['messages'])} messages>"
            
            # current_explanation と initial_analysis_result も巨大になる可能性があるので要約
            if "current_explanation" in tutor_ss_display and isinstance(tutor_ss_display["current_explanation"], str):
                tutor_ss_display["current_explanation"] = tutor_ss_display["current_explanation"][:200] + "..." if len(tutor_ss_display["current_explanation"]) > 200 else tutor_ss_display["current_explanation"]
            if "initial_analysis_result" in tutor_ss_display and isinstance(tutor_ss_display["initial_analysis_result"], dict):
                 tutor_ss_display["initial_analysis_result"] = f"<Dict with keys: {list(tutor_ss_display['initial_analysis_result'].keys())}>"


            st.json(tutor_ss_display, expanded=False)

    elif st.session_state.app_mode == "画像処理チューニング":
        st.header("デバッグ情報 (チューニング)")
        with st.expander("編集中のCVパラメータ (チューニング)", expanded=False):
            st.json(st.session_state.get("tuning_editable_cv_params", {}))
        with st.expander("編集中のその他パラメータ (チューニング)", expanded=False):
            st.json(st.session_state.get("tuning_editable_other_params", {}))
        with st.expander("固定CVパラメータ (Config由来)", expanded=False):
            st.json(st.session_state.get("tuning_fixed_cv_params", {}))
        with st.expander("固定その他パラメータ (Config由来)", expanded=False):
            st.json(st.session_state.get("tuning_fixed_other_params", {}))