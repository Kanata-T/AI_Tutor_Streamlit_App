# app_deepnote.py
# Deepnote ネイティブ Streamlit サポート用最適化版

import streamlit as st
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# OpenCVのインポートエラー対策
try:
    # OpenCVの環境変数設定
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ OpenCVのインポートに失敗しました: {e}")
    st.info("💡 初期化スクリプトを実行してください: `python init_deepnote.py`")
    OPENCV_AVAILABLE = False

# Deepnote統合サポート
try:
    import deepnote_toolkit
    deepnote_toolkit.set_integration_env()
    DEEPNOTE_ENV = True
except ImportError:
    DEEPNOTE_ENV = False

# 依存関係の確認とインポート
def safe_import_modules():
    """安全なモジュールインポート"""
    missing_modules = []
    
    try:
        from core import state_manager
    except ImportError as e:
        missing_modules.append(f"core.state_manager: {e}")
        return False, missing_modules
    
    try:
        from utils.config_loader import get_config
    except ImportError as e:
        missing_modules.append(f"utils.config_loader: {e}")
        return False, missing_modules
    
    try:
        from ui.tutor_mode_ui import render_tutor_mode
        from ui.tuning_mode_ui import render_tuning_mode
        from ui.display_helpers import display_debug_images_app
    except ImportError as e:
        missing_modules.append(f"ui modules: {e}")
        return False, missing_modules
    
    return True, missing_modules

# モジュールのインポート試行
import_success, import_errors = safe_import_modules()

if not import_success:
    st.error("❌ 必要なモジュールのインポートに失敗しました")
    for error in import_errors:
        st.error(f"   {error}")
    st.info("💡 解決方法:")
    st.code("python init_deepnote.py")
    st.stop()

# 成功した場合のインポート
from core import state_manager
from utils.config_loader import get_config
from ui.tutor_mode_ui import render_tutor_mode
from ui.tuning_mode_ui import render_tuning_mode
from ui.display_helpers import display_debug_images_app

# --- 環境変数の読み込み ---
load_dotenv()

# --- Streamlit UI設定 ---
st.set_page_config(
    page_title="🚀 AI学習チューター", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API設定 ---
def setup_gemini_api():
    """Gemini APIの設定"""
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not API_KEY or API_KEY == "your_gemini_api_key_here":
        st.error("⚠️ GEMINI_API_KEY が設定されていません")
        if DEEPNOTE_ENV:
            st.info("💡 Deepnoteでは .env ファイルにAPIキーを設定してください")
            st.code('GEMINI_API_KEY="your_actual_api_key"')
        else:
            st.info("💡 .env ファイルまたは環境変数でAPIキーを設定してください")
        st.stop()
    
    try:
        genai.configure(api_key=API_KEY)
        return True
    except Exception as e:
        st.error(f"❌ Gemini APIの設定に失敗: {e}")
        st.stop()

# --- 環境チェック ---
def check_environment():
    """環境の確認"""
    issues = []
    
    if not OPENCV_AVAILABLE:
        issues.append("OpenCV が利用できません")
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        issues.append("Tesseract OCR が利用できません")
    
    if issues:
        st.warning("⚠️ 環境に問題があります:")
        for issue in issues:
            st.warning(f"   • {issue}")
        st.info("💡 初期化スクリプトを実行してください:")
        st.code("python init_deepnote.py")
        
        # 続行するかの選択
        if st.button("⚠️ 問題を無視して続行"):
            st.session_state.ignore_env_issues = True
        elif not st.session_state.get("ignore_env_issues", False):
            st.stop()

# --- メイン初期化 ---
def initialize_app():
    """アプリケーションの初期化"""
    
    # 環境チェック
    check_environment()
    
    # API設定
    setup_gemini_api()
    
    # セッション状態の基本初期化
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "AIチューター"
    
    if "tutor_initialized" not in st.session_state:
        st.session_state.tutor_initialized = False
    if "tuning_initialized" not in st.session_state:
        st.session_state.tuning_initialized = False
    
    # 共通パラメータの初期化
    if "common_params_initialized" not in st.session_state:
        try:
            app_config = get_config()
            st.session_state.app_config = app_config
            
            # 画像処理設定の読み込み
            img_proc_config = app_config.get("image_processing", {})
            cv_trim_config = img_proc_config.get("opencv_trimming", {})
            
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
                "haar_apply": cv_trim_config.get("haar_apply", False),
                "haar_rect_h": cv_trim_config.get("haar_rect_h", 22),
                "haar_peak_threshold": cv_trim_config.get("haar_peak_threshold", 7.0),
                "h_proj_apply": cv_trim_config.get("h_proj_apply", False),
                "h_proj_threshold_ratio": cv_trim_config.get("h_proj_threshold_ratio", 0.15),
                "ocr_trim_padding": cv_trim_config.get("ocr_trim_padding", 0),
                "ocr_trim_lang": cv_trim_config.get("ocr_trim_lang", "eng+jpn"),
                "ocr_trim_min_conf": cv_trim_config.get("ocr_trim_min_conf", 25),
                "ocr_trim_min_box_height": cv_trim_config.get("ocr_trim_min_box_height", 5),
                "ocr_trim_max_box_height_ratio": cv_trim_config.get("ocr_trim_max_box_height_ratio", 0.3),
                "ocr_trim_min_box_width": cv_trim_config.get("ocr_trim_min_box_width", 5),
                "ocr_trim_max_box_width_ratio": cv_trim_config.get("ocr_trim_max_box_width_ratio", 0.8),
                "ocr_trim_min_aspect_ratio": cv_trim_config.get("ocr_trim_min_aspect_ratio", 0.05),
                "ocr_trim_max_aspect_ratio": cv_trim_config.get("ocr_trim_max_aspect_ratio", 20.0),
                "ocr_tesseract_config": cv_trim_config.get("ocr_tesseract_config", "--psm 6"),
            }
            
            # その他の画像処理パラメータ
            st.session_state.tuning_fixed_other_params = {
                "grayscale": img_proc_config.get("apply_grayscale", True),
                "output_format": img_proc_config.get("default_output_format", "JPEG"),
                "jpeg_quality": img_proc_config.get("default_jpeg_quality", 85),
                "max_pixels": img_proc_config.get("default_max_pixels_for_resizing", 4000000),
            }
            
            st.session_state.common_params_initialized = True
        except Exception as e:
            st.error(f"❌ 設定の読み込みに失敗: {e}")
            st.stop()

# --- UI レンダリング ---
def render_sidebar():
    """サイドバーのレンダリング"""
    with st.sidebar:
        st.title("🤖 AI学習チューター")
        
        # 環境情報の表示
        if DEEPNOTE_ENV:
            st.success("✅ Deepnote環境で実行中")
        
        if OPENCV_AVAILABLE:
            st.success("✅ OpenCV 利用可能")
        else:
            st.error("❌ OpenCV 利用不可")
        
        # モード選択
        current_mode = st.session_state.app_mode
        selected_mode = st.radio(
            "モードを選択:",
            ("AIチューター", "画像処理チューニング"),
            index=0 if current_mode == "AIチューター" else 1,
            key="mode_selector"
        )
        
        # モード変更処理
        if selected_mode != current_mode:
            st.session_state.app_mode = selected_mode
            if selected_mode == "AIチューター":
                st.session_state.tutor_initialized = False
            else:
                st.session_state.tuning_initialized = False
            st.rerun()
        
        # デバッグ情報
        render_debug_info()

def render_debug_info():
    """デバッグ情報の表示"""
    st.markdown("---")
    st.subheader("🔧 デバッグ情報")
    
    if st.session_state.app_mode == "AIチューター":
        st.write(f"**ステップ:** {state_manager.get_current_step()}")
        st.write(f"**処理中:** {st.session_state.get('processing', False)}")
        st.write(f"**曖昧フラグ:** {st.session_state.get('is_request_ambiguous', False)}")
        st.write(f"**明確化試行:** {st.session_state.get('clarification_attempts', 0)}")
        
        # 画像処理デバッグ表示
        if OPENCV_AVAILABLE:
            show_debug = st.checkbox(
                "画像処理デバッグ表示",
                value=st.session_state.get("show_img_debug_in_tutor_mode", False),
                key="debug_images_checkbox"
            )
            st.session_state.show_img_debug_in_tutor_mode = show_debug
            
            if show_debug and st.session_state.get("last_debug_images_tutor_run_final_v3"):
                with st.expander("画像処理詳細", expanded=False):
                    display_debug_images_app(
                        st.session_state.last_debug_images_tutor_run_final_v3,
                        title_prefix="固定パラメータでの"
                    )

def render_main_content():
    """メインコンテンツのレンダリング"""
    if st.session_state.app_mode == "AIチューター":
        render_tutor_mode()
    elif st.session_state.app_mode == "画像処理チューニング":
        if not OPENCV_AVAILABLE:
            st.error("❌ 画像処理チューニングモードにはOpenCVが必要です")
            st.info("💡 初期化スクリプトを実行してください:")
            st.code("python init_deepnote.py")
            return
        render_tuning_mode()

# --- メイン実行 ---
def main():
    """メイン関数"""
    try:
        # 初期化
        initialize_app()
        
        # UI レンダリング
        render_sidebar()
        render_main_content()
        
    except Exception as e:
        st.error(f"❌ アプリケーションエラー: {e}")
        st.info("💡 初期化スクリプトを実行してください:")
        st.code("python init_deepnote.py")

if __name__ == "__main__":
    main() 