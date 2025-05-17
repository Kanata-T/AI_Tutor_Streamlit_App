# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Dict, Any, List
import copy

# coreモジュールのインポート
from core import state_manager, tutor_logic

# utilsモジュールのインポート
from utils.image_processor import preprocess_uploaded_image # これは最新版を使う想定
from utils.config_loader import get_image_processing_config

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

# --- セッション状態の初期化 ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "AIチューター"

# モードごとの初期化フラグ (これは残しても良いが、主要な初期化はモード分岐の外で行う)
if "tutor_initialized" not in st.session_state: st.session_state.tutor_initialized = False
if "tuning_initialized" not in st.session_state: st.session_state.tuning_initialized = False

# --- アプリケーション全体で共通の初期化 (初回のみ) ---
if "common_params_initialized" not in st.session_state:
    config_img_proc_common = get_image_processing_config()
    config_cv_trim_defaults_common = config_img_proc_common.get("opencv_trimming", {})
    
    # AIチューターモードでも使用する「固定された」最適パラメータ
    st.session_state.tuning_fixed_cv_params = { # キー名は tuning_ ですが、実質「固定値」
        "apply": config_cv_trim_defaults_common.get("apply", True), "padding": 0,
        "adaptive_thresh_block_size": 11, "adaptive_thresh_c": 7,
        "min_contour_area_ratio": 0.00005,"gaussian_blur_kernel_width": 5,"gaussian_blur_kernel_height": 5,
        "morph_open_apply": False,
        "morph_open_kernel_size": config_cv_trim_defaults_common.get("morph_open_kernel_size", 3),
        "morph_open_iterations": config_cv_trim_defaults_common.get("morph_open_iterations", 1),
        "morph_close_apply": False,
        "morph_close_kernel_size": config_cv_trim_defaults_common.get("morph_close_kernel_size", 3),
        "morph_close_iterations": config_cv_trim_defaults_common.get("morph_close_iterations", 1),
        "haar_apply": config_cv_trim_defaults_common.get("haar_apply", True), "haar_rect_h": 22,
        "haar_peak_threshold": 7.0, "h_proj_apply": config_cv_trim_defaults_common.get("h_proj_apply", True),
        "h_proj_threshold_ratio": 0.15,
    }
    st.session_state.tuning_fixed_other_params = { # これも「固定値」
        "grayscale": config_img_proc_common.get("apply_grayscale", True),
        "output_format": config_img_proc_common.get("default_output_format", "JPEG"),
        "jpeg_quality": config_img_proc_common.get("default_jpeg_quality", 85),
        "max_pixels": config_img_proc_common.get("default_max_pixels_for_resizing", 4000000),
    }
    st.session_state.common_params_initialized = True
    print("[App] Common image processing fixed params initialized.")

# AIチューターモード用の初期化
if st.session_state.app_mode == "AIチューター" and not st.session_state.tutor_initialized:
    state_manager.initialize_session_state()
    st.session_state.tutor_initialized = True
    st.session_state.tuning_initialized = False
    print("[App] AI Tutor mode specific state initialized.")

# 画像チューニングモード専用のセッション状態初期化
if st.session_state.app_mode == "画像処理チューニング" and not st.session_state.tuning_initialized:
    st.session_state.tuning_raw_image_data = None
    st.session_state.tuning_raw_image_mime_type = None
    st.session_state.tuning_current_debug_images = []
    st.session_state.tuning_image_key_counter = 0
    st.session_state.tuning_trigger_reprocess = False
    
    # チューニングUI用の編集可能パラメータ (固定値をコピーして開始)
    # tuning_fixed_cv_params は既に共通初期化で設定されているはず
    if "tuning_editable_cv_params" not in st.session_state or not st.session_state.tuning_initialized:
        st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
    if "tuning_editable_other_params" not in st.session_state or not st.session_state.tuning_initialized:
        st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
    if "selected_param_set_name_tuning" not in st.session_state or not st.session_state.tuning_initialized:
        st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
        
    st.session_state.tuning_initialized = True
    st.session_state.tutor_initialized = False
    print("[App] Image Tuning mode specific state initialized.")


# --- モード選択 ---
current_app_mode_for_radio_ui_final_v2 = st.session_state.app_mode
new_app_mode_selected_ui_final_v2 = st.sidebar.radio(
    "アプリケーションモードを選択:", ("AIチューター", "画像処理チューニング"),
    key="app_mode_selector_radio_main_v5", # キーの一意性を保つ
    index=0 if current_app_mode_for_radio_ui_final_v2 == "AIチューター" else 1
)
if new_app_mode_selected_ui_final_v2 != current_app_mode_for_radio_ui_final_v2:
    st.session_state.app_mode = new_app_mode_selected_ui_final_v2
    if new_app_mode_selected_ui_final_v2 == "AIチューター": st.session_state.tutor_initialized = False # 再初期化を促す
    else: st.session_state.tuning_initialized = False # 再初期化を促す
    st.rerun()

# タイトル設定 (モードに応じて)
if st.session_state.app_mode == "AIチューター": st.title("AI学習チューター プロトタイプ")
else: st.title("🖼️ OpenCV 画像トリミング パラメータチューナー")


# --- UIヘルパー関数 ---
def display_analysis_result(analysis_data: Dict[str, Any], title: str = "分析結果"):
    with st.expander(title, expanded=False):
        if isinstance(analysis_data, dict):
            if "error" in analysis_data: st.error(f"分析エラー: {analysis_data['error']}")
            else: st.json(analysis_data)
            if "ocr_text_from_extraction" in analysis_data:
                st.text_area("初期分析時OCR:", analysis_data.get("ocr_text_from_extraction"), height=100, key=f"exp_ocr_{title.replace(' ','_').lower()}_disp_full")
        else: st.write(analysis_data)

def display_debug_images_app(debug_images_list: List[Dict[str, Any]], title_prefix: str = ""):
    if not debug_images_list:
        if st.session_state.app_mode == "画像処理チューニング":
             st.info("パラメータを調整して「選択した設定で再処理」ボタンを押すか、新しい画像をアップロードしてください。")
        return
    st.markdown("---"); st.subheader(f"🎨 {title_prefix}画像処理ステップ")
    tab_names = [item.get("label", f"S{i+1}") for i, item in enumerate(debug_images_list)]
    tabs = st.tabs(tab_names)
    for i, tab_container in enumerate(tabs):
        with tab_container:
            item = debug_images_list[i]
            img_data = item.get("data")
            if img_data: st.image(img_data, caption=f"M:{item.get('mode','?')},S:{item.get('size','?')}", use_container_width=True)
            else: st.warning("画像データなし")

# --- アプリケーション本体 ---
if st.session_state.app_mode == "AIチューター":
    current_step_tutor_main_final = state_manager.get_current_step()
    message_container_tutor_main = st.container(border=False)
    with message_container_tutor_main:
        if st.session_state.get("messages"):
            for i_tutor_main_msg, msg_tutor_main in enumerate(st.session_state.messages):
                with st.chat_message(msg_tutor_main["role"]):
                    content_tutor_main = msg_tutor_main.get("content")
                    if isinstance(content_tutor_main, dict) and "type" in content_tutor_main:
                        if content_tutor_main["type"] == "analysis_result": display_analysis_result(content_tutor_main["data"], content_tutor_main.get("title", f"分析結果 {i_tutor_main_msg}"))
                    elif isinstance(content_tutor_main, str): st.markdown(content_tutor_main)
                    else: st.write(str(content_tutor_main))
    if not st.session_state.get("messages") and current_step_tutor_main_final == state_manager.STEP_INPUT_SUBMISSION:
        st.info("AI学習チューターへようこそ！下の入力欄から質問をどうぞ。")

    # 1. ユーザー入力受付 (AIチューターモード)
    if current_step_tutor_main_final == state_manager.STEP_INPUT_SUBMISSION or \
       (current_step_tutor_main_final == state_manager.STEP_SESSION_END and st.session_state.get("show_new_question_form", True)):
        if current_step_tutor_main_final == state_manager.STEP_SESSION_END:
            if st.button("新しい質問の準備を始める", key="reset_main_flow_btn_end_tutor_final_v2"):
                state_manager.reset_for_new_session(); st.rerun()
        with st.form("submission_form_tutor_main_final_v2", clear_on_submit=True):
            st.markdown("#### 質問を入力してください:"); user_query_text_tf_f = st.text_input("質問テキスト", key="main_query_text_tf_f", label_visibility="collapsed")
            uploaded_file_tf_f = st.file_uploader("画像 (任意)", type=["png","jpg","jpeg","webp","gif","bmp"], key="main_file_uploader_tf_f")
            topic_opts_tf_f = ["", "文法", "語彙", "長文読解", "英作文", "その他"]; selected_topic_tf_f = st.selectbox("トピック (任意)", topic_opts_tf_f, key="main_topic_select_tf_f")
            submit_btn_tf_f = st.form_submit_button("この内容で質問する")
            if submit_btn_tf_f:
                if not user_query_text_tf_f and not uploaded_file_tf_f: st.warning("テキスト入力または画像アップロードを。")
                else:
                    if current_step_tutor_main_final == state_manager.STEP_SESSION_END: state_manager.reset_for_new_session()
                    ocr_input_image_for_llm = None # ★ OCR/LLMに渡す画像データ ★
                    debug_imgs_for_llm_run_final = []

                    if uploaded_file_tf_f:
                        print(f"[AppTutor] Preprocessing image for LLM: {uploaded_file_tf_f.name}")
                        fixed_cv = st.session_state.tuning_fixed_cv_params
                        fixed_other = st.session_state.tuning_fixed_other_params
                        
                        preprocessing_res_llm_final = preprocess_uploaded_image(
                            uploaded_file_data=uploaded_file_tf_f.getvalue(), mime_type=uploaded_file_tf_f.type,
                            max_pixels=fixed_other["max_pixels"],
                            output_format=fixed_other["output_format"],
                            jpeg_quality=fixed_other["jpeg_quality"],
                            grayscale=fixed_other["grayscale"],
                            apply_trimming_opencv_override=fixed_cv.get("apply"),
                            trim_params_override=fixed_cv
                        )
                        if preprocessing_res_llm_final and "error" not in preprocessing_res_llm_final:
                            # ★ LLM (OCR) には "ocr_input_image" を使う ★
                            ocr_input_image_for_llm = preprocessing_res_llm_final.get("ocr_input_image")
                            debug_imgs_for_llm_run_final = preprocessing_res_llm_final.get("debug_images", [])
                            if ocr_input_image_for_llm:
                                state_manager.add_message("system", f"画像処理完了 (OCR入力形式: {ocr_input_image_for_llm['mime_type']})")
                            else:
                                st.warning("OCR入力用画像の生成に問題がありましたが、処理を継続します。")
                                ocr_input_image_for_llm = preprocessing_res_llm_final.get("processed_image")
                                if ocr_input_image_for_llm:
                                    state_manager.add_message("system", f"画像処理完了 (フォールバック形式: {ocr_input_image_for_llm['mime_type']})")
                        elif preprocessing_res_llm_final: st.error(f"画像処理エラー: {preprocessing_res_llm_final['error']}")
                        else: st.error("画像処理で予期せぬエラー。")
                    st.session_state.last_debug_images_tutor_run_final_v3 = debug_imgs_for_llm_run_final # デバッグ用
                    # ★ state_manager には ocr_input_image_for_llm を渡す ★
                    state_manager.store_user_input(user_query_text_tf_f, ocr_input_image_for_llm, selected_topic_tf_f)
                    user_msg_content_final = f"質問: {user_query_text_tf_f}" + (" (画像あり)" if ocr_input_image_for_llm else "")
                    state_manager.add_message("user", user_msg_content_final)
                    state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS); st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_INITIAL_ANALYSIS:
        if st.session_state.initial_analysis_result is None and not st.session_state.get("processing"): # processingチェック
            state_manager.set_processing_status(True)
            with st.spinner("AIがあなたの質問を分析中です..."): analysis_result_ia_f = tutor_logic.perform_initial_analysis_logic()
            state_manager.set_processing_status(False)
            if analysis_result_ia_f:
                if "error" in analysis_result_ia_f:
                    st.error(f"分析エラー: {analysis_result_ia_f.get('error')}"); state_manager.add_message("system", f"エラー(初期分析): {analysis_result_ia_f.get('error')}")
                    state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION); st.session_state.show_new_question_form = True # エラー時は入力フォーム再表示
                else:
                    state_manager.store_initial_analysis_result(analysis_result_ia_f); state_manager.add_message("system", {"type": "analysis_result", "data": dict(analysis_result_ia_f), "title": "AIによる初期分析"})
                    if st.session_state.is_request_ambiguous: state_manager.set_current_step(state_manager.STEP_CLARIFICATION_NEEDED)
                    else: state_manager.add_message("assistant", "ご質問内容を理解しました。どのようなスタイルの解説がご希望ですか？"); state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                st.rerun()
            else: st.error("分析処理で予期せぬエラー。"); state_manager.add_message("system", "エラー(初期分析): 結果がNone。"); state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION); st.session_state.show_new_question_form = True; st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_CLARIFICATION_NEEDED:
        needs_clar_q_f = False
        if not st.session_state.messages or st.session_state.messages[-1]["role"] == "user" or \
           (st.session_state.messages[-1]["role"] == "system" and "analysis_result" in (st.session_state.messages[-1].get("content", {})).get("type","")):
            if st.session_state.get("clarification_attempts", 0) == 0: needs_clar_q_f = True # 初回のみ自動生成
        if needs_clar_q_f and not st.session_state.get("processing"):
            st.session_state.clarification_attempts = st.session_state.get("clarification_attempts", 0) + 1
            state_manager.set_processing_status(True)
            with st.spinner("AIが確認のための質問を準備中です..."): q_ai_f = tutor_logic.generate_clarification_question_logic()
            state_manager.set_processing_status(False)
            if q_ai_f and "エラー" not in q_ai_f: state_manager.add_message("assistant", q_ai_f); state_manager.add_clarification_history_message("assistant", q_ai_f)
            else:
                err_clar_q_f = q_ai_f or "明確化質問生成エラー。" ; state_manager.add_message("system", f"エラー(明確化質問): {err_clar_q_f}")
                state_manager.add_message("assistant", "確認質問準備に問題あり。現在の理解で進みます。スタイル選択へどうぞ。")
                st.session_state.is_request_ambiguous = False; state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
            st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_SELECT_STYLE:
        st.markdown("---"); st.subheader("解説スタイルを選択してください")
        disp_req_style_sel_f = st.session_state.clarified_request_text or (st.session_state.initial_analysis_result.get("summary") if st.session_state.initial_analysis_result else None) or st.session_state.user_query_text
        if disp_req_style_sel_f: st.info(f"現在のリクエスト: 「{disp_req_style_sel_f}」")
        style_opts_sel_f = {"detailed": "詳しく(標準)", "hint": "ヒントのみ", "socratic": "質問形式で"}
        curr_style_sel_f = st.session_state.get("selected_explanation_style", "detailed")
        sel_key_style_f = st.radio("希望スタイル:", list(style_opts_sel_f.keys()), format_func=lambda k_sf: style_opts_sel_f[k_sf], index=list(style_opts_sel_f.keys()).index(curr_style_sel_f) if curr_style_sel_f in style_opts_sel_f else 0, key="style_radio_tutor_final_v2", horizontal=True)
        if st.button("このスタイルで解説生成", key="confirm_style_tutor_btn_final_v2", type="primary"):
            state_manager.set_explanation_style(sel_key_style_f); state_manager.add_message("user", f"（スタイル「{style_opts_sel_f[sel_key_style_f]}」を選択）")
            state_manager.set_current_step(state_manager.STEP_GENERATE_EXPLANATION); st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_GENERATE_EXPLANATION:
        if st.session_state.current_explanation is None and not st.session_state.get("processing"):
            state_manager.set_processing_status(True)
            with st.spinner("AIが解説準備中..."): exp_tutor_f = tutor_logic.generate_explanation_logic()
            state_manager.set_processing_status(False)
            if exp_tutor_f and "エラー" not in exp_tutor_f: state_manager.store_generated_explanation(exp_tutor_f); state_manager.set_current_step(state_manager.STEP_FOLLOW_UP_LOOP)
            else: err_msg_exp_f2 = exp_tutor_f or "解説生成エラー。"; state_manager.add_message("system", f"エラー(解説生成): {err_msg_exp_f2}")
            st.rerun()
            
    elif current_step_tutor_main_final == state_manager.STEP_SUMMARIZE:
        if st.session_state.session_summary is None and st.session_state.student_performance_analysis is None and not st.session_state.get("processing"):
            state_manager.set_processing_status(True); sum_txt_f2, ana_rep_f2 = None, None; comb_parts_f2 = []
            with st.spinner("AIが要約と学習分析を準備中..."):
                sum_txt_f2 = tutor_logic.generate_summary_logic()
                if not sum_txt_f2 or "エラー" in sum_txt_f2: state_manager.add_message("system", f"エラー(要約): {sum_txt_f2 or '失敗'}")
                else: st.session_state.session_summary = sum_txt_f2; comb_parts_f2.append(f"【今回のまとめ】\n\n{sum_txt_f2}")
                ana_rep_f2 = tutor_logic.analyze_student_performance_logic()
                if not ana_rep_f2 or "エラー" in ana_rep_f2:
                    state_manager.add_message("system", f"エラー(分析): {ana_rep_f2 or '失敗'}"); st.session_state.student_performance_analysis = "分析失敗。"
                    comb_parts_f2.append("【学習分析】\n\n分析失敗。")
                else: st.session_state.student_performance_analysis = ana_rep_f2; comb_parts_f2.append(f"【学習分析 (β版)】\n\n{ana_rep_f2}")
            state_manager.set_processing_status(False)
            if comb_parts_f2: state_manager.add_message("assistant", "\n\n---\n\n".join(comb_parts_f2))
            state_manager.set_current_step(state_manager.STEP_SESSION_END); st.session_state.show_new_question_form = True; st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_SESSION_END:
        if not st.session_state.get("show_new_question_form"): # 新規質問フォームが表示されていない場合のみ
            if st.button("新しい質問をする", key="new_q_from_session_end_final_v2", use_container_width=True):
                state_manager.reset_for_new_session(); st.rerun()

    # AIチューターモードの共通チャット入力
    if current_step_tutor_main_final in [state_manager.STEP_CLARIFICATION_NEEDED, state_manager.STEP_FOLLOW_UP_LOOP]:
        if current_step_tutor_main_final == state_manager.STEP_FOLLOW_UP_LOOP and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            if st.button("✅ 理解しました / 要約へ", key="understood_main_btn_final_v2"):
                state_manager.add_message("user", "（理解しました）"); state_manager.set_current_step(state_manager.STEP_SUMMARIZE); st.rerun()
        chat_disabled_tf_f = st.session_state.get("processing", False)
        if user_chat_input_tf_f := st.chat_input("AIへの返答や追加質問...", disabled=chat_disabled_tf_f, key="main_tutor_chat_input_area_final_v2"):
            state_manager.add_message("user", user_chat_input_tf_f)
            if current_step_tutor_main_final == state_manager.STEP_CLARIFICATION_NEEDED:
                state_manager.add_clarification_history_message("user", user_chat_input_tf_f)
                state_manager.set_processing_status(True)
                with st.spinner("AIが応答を分析中..."): clar_res_f2 = tutor_logic.analyze_user_clarification_logic(user_chat_input_tf_f)
                state_manager.set_processing_status(False)
                if clar_res_f2:
                    state_manager.add_message("system", {"type": "analysis_result", "data": dict(clar_res_f2), "title": "明確化応答の分析結果"})
                    if "error" in clar_res_f2: state_manager.add_message("assistant", f"応答分析エラー。現在の理解で進みます。"); st.session_state.is_request_ambiguous = False; state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                    else:
                        state_manager.store_clarification_analysis(clar_res_f2)
                        if not st.session_state.is_request_ambiguous: state_manager.add_message("assistant", clar_res_f2.get("ai_response_to_user","理解しました。スタイル選択へ。")); state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                        else: state_manager.add_message("assistant", f"もう少し確認させてください: {clar_res_f2.get('remaining_issue', '不明点あり')}")
                else: state_manager.add_message("system", "エラー(明確化応答分析): 結果None。"); state_manager.add_message("assistant", "応答分析エラー。スタイル選択へ。"); st.session_state.is_request_ambiguous = False; state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                st.rerun()
            elif current_step_tutor_main_final == state_manager.STEP_FOLLOW_UP_LOOP:
                state_manager.set_processing_status(True)
                with st.spinner("AIが応答準備中..."): followup_resp_f2 = tutor_logic.generate_followup_response_logic(user_chat_input_tf_f)
                state_manager.set_processing_status(False)
                if followup_resp_f2 and "エラー" not in followup_resp_f2: state_manager.add_message("assistant", followup_resp_f2)
                else: state_manager.add_message("system", f"エラー(フォローアップ): {followup_resp_f2 or '失敗'}")
                st.rerun()


elif st.session_state.app_mode == "画像処理チューニング":
    # --- 画像処理チューニングモードのUIとロジック ---
    with st.sidebar:
        st.header("⚙️ 処理パラメータ (チューニング)")
        st.markdown("#### 1. 画像を選択")
        uploaded_file_obj_tune_complete_v2 = st.file_uploader(
            "調整対象の画像", type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
            key=f"tuning_file_uploader_{st.session_state.tuning_image_key_counter}_complete_v2"
        )
        if st.button("現在の画像をクリア", key="clear_image_button_tune_complete_v2"):
            st.session_state.tuning_raw_image_data = None; st.session_state.tuning_raw_image_mime_type = None
            st.session_state.tuning_current_debug_images = []; st.session_state.tuning_image_key_counter += 1
            st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
            st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
            st.session_state.tuning_trigger_reprocess = False; st.rerun()

        st.markdown("#### 2. パラメータセット提案")
        all_presets_tune_complete_v2 = {
            "現在の編集値": {},
            "00: 固定値 (推奨)": st.session_state.tuning_fixed_cv_params, # config/固定値を参照
            "01: 基本 (ブラーなし)": {"apply":True, "padding":15, "adaptive_thresh_block_size":11, "adaptive_thresh_c":7, "min_contour_area_ratio":0.0005, "gaussian_blur_kernel_width":0, "gaussian_blur_kernel_height":0, "morph_open_apply":False, "morph_close_apply":False, "haar_apply":False, "h_proj_apply":False},
            "02: 小さい文字 (Block小,C調整,Open有)": {"apply":True, "padding":10, "adaptive_thresh_block_size":7, "adaptive_thresh_c":3, "min_contour_area_ratio":0.0001, "gaussian_blur_kernel_width":0, "gaussian_blur_kernel_height":0, "morph_open_apply":True, "morph_open_kernel_size":2, "morph_open_iterations":1, "morph_close_apply":False, "haar_apply":True, "h_proj_apply":True},
        }
        # 他のプリセットもここに追加できます

        if "selected_param_set_name_tuning" not in st.session_state: st.session_state.selected_param_set_name_tuning = "00: 固定値 (推奨)"
        selected_set_name_tune_ui_v2 = st.selectbox("パラメータセット:", list(all_presets_tune_complete_v2.keys()), key="param_set_tune_select_complete_v2", index=list(all_presets_tune_complete_v2.keys()).index(st.session_state.selected_param_set_name_tuning))
        if selected_set_name_tune_ui_v2 != st.session_state.selected_param_set_name_tuning:
            st.session_state.selected_param_set_name_tuning = selected_set_name_tune_ui_v2
            if selected_set_name_tune_ui_v2 != "現在の編集値":
                new_editable_params_v2 = st.session_state.tuning_fixed_cv_params.copy() # 固定値をベースに
                new_editable_params_v2.update(all_presets_tune_complete_v2[selected_set_name_tune_ui_v2]) # プリセットで上書き
                st.session_state.tuning_editable_cv_params = new_editable_params_v2
            st.rerun()
        
        st.markdown("#### 3. OpenCV トリミング詳細設定")
        editable_cv_tune_v2 = st.session_state.tuning_editable_cv_params
        editable_cv_tune_v2["apply"] = st.checkbox("OpenCVトリミング適用", value=editable_cv_tune_v2.get("apply"), key="cv_apply_tune_v2")
        if editable_cv_tune_v2["apply"]:
            editable_cv_tune_v2["padding"] = st.number_input("パディング", 0, 200, editable_cv_tune_v2.get("padding"), 1, key="cv_pad_num_v2")
            bs_val_v2 = st.number_input("適応閾値ブロック(3以上奇数)", 3, value=editable_cv_tune_v2.get("adaptive_thresh_block_size"), step=2, key="cv_block_num_v2")
            editable_cv_tune_v2["adaptive_thresh_block_size"] = bs_val_v2 if bs_val_v2 % 2 != 0 else bs_val_v2 + 1
            editable_cv_tune_v2["adaptive_thresh_c"] = st.number_input("適応閾値C", value=editable_cv_tune_v2.get("adaptive_thresh_c"), step=1, key="cv_c_num_v2")
            editable_cv_tune_v2["min_contour_area_ratio"] = st.number_input("最小輪郭面積比", 0.0, 0.1, editable_cv_tune_v2.get("min_contour_area_ratio"), 0.00001, "%.5f", key="cv_area_num_v2")
            editable_cv_tune_v2["gaussian_blur_kernel_width"] = st.number_input("ブラー幅(0で無効,奇数)", 0, value=editable_cv_tune_v2.get("gaussian_blur_kernel_width"), step=1, key="cv_blurw_num_v2")
            editable_cv_tune_v2["gaussian_blur_kernel_height"] = st.number_input("ブラー高さ(0で無効,奇数)", 0, value=editable_cv_tune_v2.get("gaussian_blur_kernel_height"), step=1, key="cv_blurh_num_v2")
            st.markdown("###### モルフォロジー変換")
            editable_cv_tune_v2["morph_open_apply"] = st.checkbox("オープニング", value=editable_cv_tune_v2.get("morph_open_apply"), key="cv_mopen_cb_v2")
            if editable_cv_tune_v2.get("morph_open_apply"):
                k_op_v2=st.number_input("Openカーネル(奇数)",1,value=editable_cv_tune_v2.get("morph_open_kernel_size"),step=2,key="cv_mopenk_num_v2")
                editable_cv_tune_v2["morph_open_kernel_size"]=k_op_v2 if k_op_v2%2!=0 else k_op_v2+1
                editable_cv_tune_v2["morph_open_iterations"]=st.number_input("Openイテレーション",1,value=editable_cv_tune_v2.get("morph_open_iterations"),step=1,key="cv_mopeni_num_v2")
            editable_cv_tune_v2["morph_close_apply"] = st.checkbox("クロージング", value=editable_cv_tune_v2.get("morph_close_apply"), key="cv_mclose_cb_v2")
            if editable_cv_tune_v2.get("morph_close_apply"):
                k_cl_v2=st.number_input("Closeカーネル(奇数)",1,value=editable_cv_tune_v2.get("morph_close_kernel_size"),step=2,key="cv_mclosek_num_v2")
                editable_cv_tune_v2["morph_close_kernel_size"]=k_cl_v2 if k_cl_v2%2!=0 else k_cl_v2+1
                editable_cv_tune_v2["morph_close_iterations"]=st.number_input("Closeイテレーション",1,value=editable_cv_tune_v2.get("morph_close_iterations"),step=1,key="cv_mclosei_num_v2")
            with st.expander("補助的トリミング (実験的)", expanded=False):
                editable_cv_tune_v2["haar_apply"]=st.checkbox("Haar-Like",value=editable_cv_tune_v2.get("haar_apply"),key="cv_haar_cb_v2")
                if editable_cv_tune_v2.get("haar_apply"):
                    hrh_v2=st.number_input("Haarマスク高(偶数)",4,100,editable_cv_tune_v2.get("haar_rect_h"),2,key="cv_haarrh_num_v2")
                    editable_cv_tune_v2["haar_rect_h"]=hrh_v2 if hrh_v2%2==0 else hrh_v2+1
                    editable_cv_tune_v2["haar_peak_threshold"]=st.number_input("Haarピーク閾値",0.0,100.0,editable_cv_tune_v2.get("haar_peak_threshold"),0.5,"%.1f",key="cv_haarpt_num_v2")
                editable_cv_tune_v2["h_proj_apply"]=st.checkbox("水平射影",value=editable_cv_tune_v2.get("h_proj_apply"),key="cv_hproj_cb_v2")
                if editable_cv_tune_v2.get("h_proj_apply"):
                    editable_cv_tune_v2["h_proj_threshold_ratio"]=st.number_input("水平射影閾値比",0.001,0.5,editable_cv_tune_v2.get("h_proj_threshold_ratio"),0.001,"%.3f",key="cv_hprojtr_num_v2")

        st.markdown("#### 4. その他処理設定")
        editable_other_tune_v2 = st.session_state.tuning_editable_other_params
        editable_other_tune_v2["grayscale"] = st.checkbox("グレースケール化", value=editable_other_tune_v2.get("grayscale"), key="grayscale_tune_complete_v2")
        editable_other_tune_v2["max_pixels"] = st.number_input("リサイズ最大ピクセル(0で無効)", 0, value=editable_other_tune_v2.get("max_pixels"), step=100000, key="maxpix_tune_complete_v2")

        if st.button("この設定で画像を再処理", key="reprocess_tune_btn_complete_v2", type="primary", use_container_width=True, disabled=(st.session_state.tuning_raw_image_data is None)):
            st.session_state.tuning_trigger_reprocess = True
    
    # メインエリア表示 (チューニングモード)
    if st.session_state.tuning_raw_image_data:
        st.markdown("#### 元画像 (チューニング対象):")
        st.image(st.session_state.tuning_raw_image_data, caption=f"元画像 ({st.session_state.tuning_raw_image_mime_type})", use_container_width=True)
    display_title_tune_complete_v2 = f"「{st.session_state.selected_param_set_name_tuning}」設定での " if st.session_state.selected_param_set_name_tuning != "現在の編集値" else "現在の設定での "
    display_debug_images_app(st.session_state.tuning_current_debug_images, title_prefix=display_title_tune_complete_v2)

    # チューニング用のロジック
    should_reprocess_tuning_final_v2 = False
    if uploaded_file_obj_tune_complete_v2 and (st.session_state.tuning_raw_image_data is None or uploaded_file_obj_tune_complete_v2.getvalue() != st.session_state.tuning_raw_image_data):
        st.session_state.tuning_raw_image_data = uploaded_file_obj_tune_complete_v2.getvalue(); st.session_state.tuning_raw_image_mime_type = uploaded_file_obj_tune_complete_v2.type
        st.session_state.tuning_current_debug_images = []; should_reprocess_tuning_final_v2 = True
        st.session_state.selected_param_set_name_tuning = "現在の編集値"
        st.session_state.tuning_editable_cv_params = copy.deepcopy(st.session_state.tuning_fixed_cv_params)
        st.session_state.tuning_editable_other_params = copy.deepcopy(st.session_state.tuning_fixed_other_params)
    if st.session_state.get("tuning_trigger_reprocess"):
        should_reprocess_tuning_final_v2 = True; st.session_state.tuning_trigger_reprocess = False

    if should_reprocess_tuning_final_v2 and st.session_state.tuning_raw_image_data:
        print(f"[AppTune] Reprocessing with: CV={st.session_state.tuning_editable_cv_params}, Other={st.session_state.tuning_editable_other_params}")
        editable_cv_p_final_v2 = st.session_state.tuning_editable_cv_params; editable_o_p_final_v2 = st.session_state.tuning_editable_other_params
        params_to_pass_tune_final = {k:v for k,v in editable_cv_p_final_v2.items() if k != "apply"} # applyは別引数
        # 奇数保証などは preprocess_uploaded_image または trim_whitespace_opencv 内部で対応
        with st.spinner("チューニング画像処理中..."):
            res_tune_final_v2 = preprocess_uploaded_image(
                uploaded_file_data=st.session_state.tuning_raw_image_data, mime_type=st.session_state.tuning_raw_image_mime_type,
                max_pixels=editable_o_p_final_v2["max_pixels"], grayscale=editable_o_p_final_v2["grayscale"],
                output_format=editable_o_p_final_v2["output_format"], jpeg_quality=editable_o_p_final_v2["jpeg_quality"],
                apply_trimming_opencv_override=editable_cv_p_final_v2.get("apply"), trim_params_override=params_to_pass_tune_final
            )
        if res_tune_final_v2 and "error" not in res_tune_final_v2: st.session_state.tuning_current_debug_images = res_tune_final_v2.get("debug_images", [])
        elif res_tune_final_v2: st.error(f"画像処理エラー(Tune): {res_tune_final_v2['error']}"); st.session_state.tuning_current_debug_images = res_tune_final_v2.get("debug_images",[])
        else: st.error("画像処理で予期せぬエラー(Tune)。"); st.session_state.tuning_current_debug_images = []
        st.rerun()

# --- サイドバー下部の共通デバッグ情報 ---
with st.sidebar:
    st.markdown("---")
    if st.session_state.app_mode == "AIチューター":
        st.header("デバッグ情報 (AIチューター)")
        st.write(f"現在のステップ: `{state_manager.get_current_step()}`")
        st.write(f"処理中: `{st.session_state.get('processing', False)}`")
        st.write(f"曖昧フラグ: `{st.session_state.get('is_request_ambiguous', False)}`")
        st.write(f"明確化試行: `{st.session_state.get('clarification_attempts', 0)}`")
        st.session_state.show_img_debug_in_tutor_mode_final = st.checkbox("画像処理デバッグ表示 (AIチューター実行時)", value=st.session_state.get("show_img_debug_in_tutor_mode_final", False), key="show_img_debug_tutor_cb_final_v2")
        if st.session_state.show_img_debug_in_tutor_mode_final and st.session_state.get("last_debug_images_tutor_run_final_v3"):
            with st.expander("前回の画像処理詳細 (AIチューター)", expanded=False):
                display_debug_images_app(st.session_state.last_debug_images_tutor_run_final_v3, title_prefix="固定パラメータでの")
        with st.expander("セッションステート全体 (AIチューター)", expanded=False):
            tutor_ss_disp_f = {k:v for k,v in st.session_state.items() if not k.startswith("tuning_") and not k.startswith("editable_") and not k.startswith("ui_") and k != "all_preset_options" and k != "selected_param_set_name_tuning"}
            if "uploaded_file_data" in tutor_ss_disp_f and tutor_ss_disp_f["uploaded_file_data"] is not None:
                tutor_ss_disp_f["uploaded_file_data"] = {k_i:v_i if k_i!="data" else f"<bytes {len(v_i)}>" for k_i,v_i in tutor_ss_disp_f["uploaded_file_data"].items()}
            if "last_debug_images_tutor_run_final_v3" in tutor_ss_disp_f : tutor_ss_disp_f.pop("last_debug_images_tutor_run_final_v3")
            if "messages" in tutor_ss_disp_f and isinstance(tutor_ss_disp_f["messages"], list): tutor_ss_disp_f["messages"] = f"<List of {len(tutor_ss_disp_f['messages'])} messages>"
            st.json(tutor_ss_disp_f)

    elif st.session_state.app_mode == "画像処理チューニング":
        st.header("デバッグ情報 (チューニング)")
        with st.expander("編集中のCVパラメータ", expanded=False): st.json(st.session_state.get("tuning_editable_cv_params", {}))
        with st.expander("編集中のその他パラメータ", expanded=False): st.json(st.session_state.get("tuning_editable_other_params", {}))
        with st.expander("固定CVパラメータ (AIチューター用)", expanded=False): st.json(st.session_state.get("tuning_fixed_cv_params", {}))