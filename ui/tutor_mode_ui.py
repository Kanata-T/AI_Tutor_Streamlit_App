# ui/tutor_mode_ui.py
import streamlit as st
from typing import Dict, Any # Listは display_helpers で使われるのでここでは不要かも
from PIL import Image  # 画像向き補正用
from io import BytesIO  # バイトデータから画像を開くため

# coreモジュールのインポート
from core import state_manager, tutor_logic

# utilsモジュールのインポート
from utils.image_processor import preprocess_uploaded_image
from utils.image_processing_parts.base_image_utils import auto_orient_image_opencv, image_to_bytes
from utils.demo_case_loader import get_available_demo_cases, load_demo_case_data # ★新規インポート
from utils import config_loader # ★ config_loader をインポート ★

# uiヘルパーのインポート
from .display_helpers import display_analysis_result # 相対インポート

APP_CONFIG = config_loader.get_config()
LOGIC_CONFIG = APP_CONFIG.get("application_logic", {})
MAX_CLARIFICATION_ATTEMPTS = LOGIC_CONFIG.get("max_clarification_attempts", 1) # デフォルト値 1

def render_tutor_mode():
    """AIチューターモードのUIとロジックをレンダリングします。"""
    st.title("AI学習チューター プロトタイプ")

    # AIチューターモード用の初期化 (app.pyから移動、呼び出し時に tutor_initialized フラグで制御される想定)
    # この初期化は app.py のモード切り替え時に tutor_initialized = False とすることで再実行される
    if not st.session_state.get("tutor_initialized", False): # 既に app.py で設定されているはず
        state_manager.initialize_session_state()
        st.session_state.tutor_initialized = True # 初期化完了フラグ
        st.session_state.tuning_initialized = False # 他のモードの初期化フラグをリセット
        print("[TutorModeUI] AI Tutor mode specific state initialized via render_tutor_mode.")
        # 注意: ここで st.rerun() を呼ぶと無限ループになる可能性。
        # 初期化後にUI要素が依存するセッション状態が変わる場合、呼び出し元 (app.py) の st.rerun() で対応。
        # 今回は state_manager.initialize_session_state() がUI表示に直接影響するメッセージをクリアする等ないので大丈夫そう。

    # --- ★新規追加: デモケース選択UI (サイドバーに配置) ---
    with st.sidebar:
        st.header("デモケース選択")
        available_cases = get_available_demo_cases()
        if "selected_demo_case_id_tutor" not in st.session_state:
            st.session_state.selected_demo_case_id_tutor = None

        if available_cases:
            selected_case_id_from_ui = st.selectbox(
                "テストケースを選択:",
                options=["---"] + available_cases,
                index=0 if st.session_state.selected_demo_case_id_tutor is None \
                      else (["---"] + available_cases).index(st.session_state.selected_demo_case_id_tutor),
                key="tutor_demo_case_selector"
            )

            if selected_case_id_from_ui != "---" and selected_case_id_from_ui != st.session_state.selected_demo_case_id_tutor:
                st.session_state.selected_demo_case_id_tutor = selected_case_id_from_ui
                case_data = load_demo_case_data(selected_case_id_from_ui)
                if case_data and "error" not in case_data:
                    state_manager.reset_for_new_session()
                    st.session_state.main_query_text_ui_module = case_data.get("question_text", "")
                    
                    # ★★★ 修正点: 全ての画像情報をリストで保持 ★★★
                    st.session_state.demo_case_loaded_images = [] # 初期化
                    if case_data.get("images"):
                        st.session_state.demo_case_loaded_images = case_data["images"] # 画像情報リストをそのまま保存
                        # UI表示用 (最初の画像情報を一時的に保持、または選択式にする準備)
                        first_image_info = case_data["images"][0]
                        st.session_state.current_display_image_bytes_from_demo = first_image_info["bytes"]
                        st.session_state.current_display_image_mime_from_demo = first_image_info["mime_type"]
                        st.session_state.current_display_image_filename_from_demo = first_image_info["filename"]
                        st.info(f"デモケース '{selected_case_id_from_ui}' から {len(case_data['images'])}枚の画像がロードされました。")
                        if len(case_data['images']) > 1:
                            st.write("現在、送信時には最初の画像のみが使用されます。複数画像対応は開発中です。")
                    else:
                        # 画像がない場合のクリア処理
                        st.session_state.current_display_image_bytes_from_demo = None
                        st.session_state.current_display_image_mime_from_demo = None
                        st.session_state.current_display_image_filename_from_demo = None
                    st.session_state.demo_case_style = case_data.get("style_text")
                    st.session_state.demo_case_understood = case_data.get("understood_text")
                    state_manager.add_message("system", f"デモケース「{selected_case_id_from_ui}」がロードされました。")
                    print(f"[TutorModeUI] Demo case '{selected_case_id_from_ui}' loaded with {len(st.session_state.demo_case_loaded_images)} images and session reset.")
                    st.rerun()
                elif case_data:
                    st.error(f"デモケース '{selected_case_id_from_ui}' の読み込みエラー: {case_data['error']}")
                    st.session_state.selected_demo_case_id_tutor = None
            elif selected_case_id_from_ui == "---" and st.session_state.selected_demo_case_id_tutor is not None:
                st.session_state.selected_demo_case_id_tutor = None
                print("[TutorModeUI] Demo case selection cleared.")
        else:
            st.sidebar.info("利用可能なデモケースが `demo_cases` フォルダに見つかりません。")
        st.sidebar.markdown("---")
    # --- ここまでデモケース選択UI ---

    current_step_tutor_main_final = state_manager.get_current_step()
    message_container_tutor_main = st.container(border=False)
    
    with message_container_tutor_main:
        if st.session_state.get("messages"):
            for i_tutor_main_msg, msg_tutor_main in enumerate(st.session_state.messages):
                with st.chat_message(msg_tutor_main["role"]):
                    content_tutor_main = msg_tutor_main.get("content")
                    if isinstance(content_tutor_main, dict) and "type" in content_tutor_main:
                        if content_tutor_main["type"] == "analysis_result":
                            display_analysis_result(
                                content_tutor_main["data"], 
                                content_tutor_main.get("title", f"分析結果 {i_tutor_main_msg}")
                            )
                    elif isinstance(content_tutor_main, str):
                        st.markdown(content_tutor_main)
                    else:
                        st.write(str(content_tutor_main)) # フォールバック

    if not st.session_state.get("messages") and current_step_tutor_main_final == state_manager.STEP_INPUT_SUBMISSION:
        st.info("AI学習チューターへようこそ！下の入力欄から質問をどうぞ。")

    # 1. ユーザー入力受付 (AIチューターモード)
    if current_step_tutor_main_final == state_manager.STEP_INPUT_SUBMISSION or \
       (current_step_tutor_main_final == state_manager.STEP_SESSION_END and st.session_state.get("show_new_question_form", True)):
        
        if current_step_tutor_main_final == state_manager.STEP_SESSION_END:
            if st.button("新しい質問の準備を始める", key="reset_main_flow_btn_end_tutor_ui_module"): # キー変更
                state_manager.reset_for_new_session()
                st.session_state.selected_demo_case_id_tutor = None
                st.session_state.current_display_image_bytes_from_demo = None
                st.session_state.current_display_image_mime_from_demo = None
                st.session_state.current_display_image_filename_from_demo = None
                st.rerun()

        with st.form("submission_form_tutor_main_ui_module", clear_on_submit=False): # キー変更
            st.markdown("#### 質問を入力してください:")
            user_query_text_tf_f = st.text_input(
                "質問テキスト", 
                key="main_query_text_ui_module", 
                label_visibility="collapsed",
                value=st.session_state.get("main_query_text_ui_module","")
            )
            uploaded_files_tf_f = None # 複数形に変更
            demo_case_has_images = bool(st.session_state.get("demo_case_loaded_images"))

            if demo_case_has_images:
                st.info(f"デモケースから {len(st.session_state.demo_case_loaded_images)} 枚の画像がロード済みです。")
                # (オプション) ロードされた画像の簡易プレビューやリスト表示
                for idx, img_info in enumerate(st.session_state.demo_case_loaded_images):
                    st.caption(f"  - {img_info['filename']}")
                # 画像種別選択UIは不要になったため削除・コメントアウト
                # 例: st.selectbox("画像種別を選択", ...) など
                # 期待される画像種別 (expected_image_type) のUI表示も現時点では行わない
                uploaded_files_tf_f_manual = st.file_uploader(
                    "新しい画像をアップロード (デモ画像を全て上書き):", 
                    type=["png","jpg","jpeg","webp","gif","bmp"], 
                    key="main_file_uploader_ui_module_manual_multi", # キー変更
                    accept_multiple_files=True # ★変更
                )
                if uploaded_files_tf_f_manual: # リストで返ってくる
                    uploaded_files_tf_f = uploaded_files_tf_f_manual
                    st.session_state.demo_case_loaded_images = [] # デモ画像をクリア
                    st.session_state.selected_demo_case_id_tutor = None
                    print("[TutorModeUI] Manual image upload overrides demo case images.")
            else:
                uploaded_files_tf_f = st.file_uploader(
                    "画像 (任意、複数可):", 
                    type=["png","jpg","jpeg","webp","gif","bmp"], 
                    key="main_file_uploader_ui_module_multi", # キー変更
                    accept_multiple_files=True # ★変更
                )

            topic_opts_tf_f = ["", "文法", "語彙", "長文読解", "英作文", "その他"]
            selected_topic_tf_f = st.selectbox("トピック (任意)", topic_opts_tf_f, key="main_topic_select_ui_module")
            submit_btn_tf_f = st.form_submit_button("この内容で質問する")

            if submit_btn_tf_f:
                final_user_query_text = st.session_state.main_query_text_ui_module

                # --- ▼ 複数画像処理の準備 ▼ ---
                final_images_to_process: list = [] # [{"bytes": ..., "mime_type": ..., "filename": ...}, ...]

                if uploaded_files_tf_f: # 手動アップロードを優先 (リストで返ってくる)
                    for uploaded_file_obj in uploaded_files_tf_f:
                        final_images_to_process.append({
                            "bytes": uploaded_file_obj.getvalue(),
                            "mime_type": uploaded_file_obj.type,
                            "filename": uploaded_file_obj.name
                        })
                    print(f"[TutorModeUI] Using {len(final_images_to_process)} manually uploaded image(s).")
                elif st.session_state.get("demo_case_loaded_images"): # デモケースの画像リスト
                    for img_info in st.session_state.demo_case_loaded_images:
                        final_images_to_process.append({
                            "bytes": img_info["bytes"],
                            "mime_type": img_info["mime_type"],
                            "filename": img_info["filename"]
                        })
                    print(f"[TutorModeUI] Using {len(final_images_to_process)} image(s) from demo case.")
                # --- ▲ 複数画像処理の準備ここまで ▲ ---

                if not final_user_query_text and not final_images_to_process:
                    st.warning("テキスト入力または画像アップロードのいずれかを行ってください。")
                else:
                    if current_step_tutor_main_final == state_manager.STEP_SESSION_END:
                        state_manager.reset_for_new_session()
                        st.session_state.selected_demo_case_id_tutor = None
                        st.session_state.current_display_image_bytes_from_demo = None
                        st.session_state.current_display_image_mime_from_demo = None
                        st.session_state.current_display_image_filename_from_demo = None

                    processed_vision_images_list = [] # Vision用画像の処理結果リスト
                    processed_ocr_images_list = []    # OCR用画像の処理結果リスト (もし必要なら)
                    all_debug_images_from_processing = [] # 全画像のデバッグ情報を集約

                    if final_images_to_process:
                        fixed_cv = st.session_state.get("tuning_fixed_cv_params", {})
                        fixed_other = st.session_state.get("tuning_fixed_other_params", {})
                        app_cfg = st.session_state.get("app_config", {}) # app.pyでロード済み想定
                        img_proc_cfg = app_cfg.get("image_processing", {})
                        default_strategy = img_proc_cfg.get("default_trimming_strategy", "ocr_then_contour")

                        # 進捗バーの準備
                        progress_bar = st.progress(0, text="画像を処理中...")
                        for idx, img_data in enumerate(final_images_to_process):
                            # st.write(f"Processing image {idx+1}/{len(final_images_to_process)}: {img_data['filename']}") # 進捗表示をprogressバーに変更
                            progress_bar.progress((idx+1)/len(final_images_to_process), text=f"画像 {idx+1}/{len(final_images_to_process)}: {img_data['filename']} を処理中...")
                            # まず向き補正
                            oriented_bytes = None
                            try:
                                pil_temp = Image.open(BytesIO(img_data["bytes"]))
                                pil_oriented = auto_orient_image_opencv(pil_temp)
                                # ... (image_to_bytes で向き補正後バイト取得) ...
                                original_mime = img_data["mime_type"].lower()
                                output_fmt_session: str = "PNG"
                                if original_mime == "image/jpeg": output_fmt_session = "JPEG"
                                jpeg_q = fixed_other.get("jpeg_quality", 85)
                                oriented_bytes = image_to_bytes(pil_oriented, target_format=output_fmt_session, jpeg_quality=jpeg_q) # type: ignore
                            except Exception as e_orient:
                                st.error(f"画像 '{img_data['filename']}' の向き補正エラー: {e_orient}")
                                oriented_bytes = img_data["bytes"] # フォールバック

                            if oriented_bytes:
                                result_single_img = preprocess_uploaded_image(
                                    uploaded_file_data=oriented_bytes,
                                    mime_type=img_data["mime_type"],
                                    max_pixels=fixed_other.get("max_pixels"),
                                    output_format=fixed_other.get("output_format"),
                                    jpeg_quality=fixed_other.get("jpeg_quality"),
                                    grayscale=fixed_other.get("grayscale"),
                                    apply_trimming_opencv_override=fixed_cv.get("apply"),
                                    trim_params_override=fixed_cv,
                                    trimming_strategy_override=default_strategy
                                )
                                if result_single_img and "error" not in result_single_img:
                                    if result_single_img.get("processed_image"):
                                        processed_vision_images_list.append(result_single_img["processed_image"])
                                    if result_single_img.get("ocr_input_image"): # 必要に応じてOCR用も収集
                                        processed_ocr_images_list.append(result_single_img["ocr_input_image"])
                                    if result_single_img.get("debug_images"):
                                        # デバッグ画像にファイル名を付与して区別
                                        for dbg_img in result_single_img["debug_images"]:
                                            dbg_img["label"] = f"[{img_data['filename']}] {dbg_img.get('label', '')}"
                                        all_debug_images_from_processing.extend(result_single_img["debug_images"])
                                else:
                                    err_msg = result_single_img.get('error', '不明なエラー') if result_single_img else '不明なエラー'
                                    st.error(f"画像 '{img_data['filename']}' の処理エラー: {err_msg}")
                            else:
                                st.error(f"画像 '{img_data['filename']}' の向き補正後データがありません。")
                        progress_bar.empty() # 完了後に消す
                    # 最終的なVision用画像をセッションに保存 (リスト形式)
                    st.session_state.uploaded_file_data = processed_vision_images_list
                    # (オプション) OCR用画像もリストでセッションに保存
                    st.session_state.ocr_input_images_for_llm = processed_ocr_images_list 
                    
                    st.session_state.last_debug_images_tutor_run_final_v3 = all_debug_images_from_processing

                    st.session_state.user_query_text = final_user_query_text
                    
                    state_manager.store_user_input(
                        final_user_query_text, 
                        processed_vision_images_list, # 画像リストを渡す
                        selected_topic_tf_f
                    )
                    
                    user_msg_content_final = f"質問: {final_user_query_text}" + (f" ({len(processed_vision_images_list)}枚の画像あり)" if processed_vision_images_list else "")
                    state_manager.add_message("user", user_msg_content_final)
                    state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS)
                    
                    # デモケース関連の一時情報をクリア
                    st.session_state.demo_case_loaded_images = [] 
                    st.session_state.current_display_image_bytes_from_demo = None
                    st.session_state.current_display_image_mime_from_demo = None
                    st.session_state.current_display_image_filename_from_demo = None
                    # ... (他のデモケース一時情報もクリア) ...
                    
                    st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_INITIAL_ANALYSIS:
        if st.session_state.initial_analysis_result is None and not st.session_state.get("processing", False):
            state_manager.set_processing_status(True)
            with st.spinner("AIがあなたの質問を分析中です..."):
                analysis_result_ia_f = tutor_logic.perform_initial_analysis_logic()
            state_manager.set_processing_status(False)

            if analysis_result_ia_f:
                if "error" in analysis_result_ia_f:
                    st.error(f"分析エラー: {analysis_result_ia_f.get('error')}")
                    state_manager.add_message("system", f"エラー(初期分析): {analysis_result_ia_f.get('error')}")
                    state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
                    st.session_state.show_new_question_form = True 
                else:
                    state_manager.store_initial_analysis_result(analysis_result_ia_f)
                    state_manager.add_message("system", {"type": "analysis_result", "data": dict(analysis_result_ia_f), "title": "AIによる初期分析"})
                    if st.session_state.is_request_ambiguous:
                        state_manager.set_current_step(state_manager.STEP_CLARIFICATION_NEEDED)
                    else:
                        state_manager.add_message("assistant", "ご質問内容を理解しました。どのようなスタイルの解説がご希望ですか？")
                        state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                st.rerun()
            else:
                st.error("分析処理で予期せぬエラーが発生しました。")
                state_manager.add_message("system", "エラー(初期分析): 結果がNone。")
                state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
                st.session_state.show_new_question_form = True
                st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_CLARIFICATION_NEEDED:
        needs_clar_q_f = False
        # 最後のメッセージがユーザーか、システムによる分析結果表示の場合に、AIからの確認質問をトリガー
        if not st.session_state.messages or \
           st.session_state.messages[-1]["role"] == "user" or \
           (st.session_state.messages[-1]["role"] == "system" and \
            isinstance(st.session_state.messages[-1].get("content"), dict) and \
            st.session_state.messages[-1]["content"].get("type") == "analysis_result"):
            
            if st.session_state.get("clarification_attempts", 0) == 0: # 初回のみ自動で質問生成
                needs_clar_q_f = True
        
        if needs_clar_q_f and not st.session_state.get("processing", False):
            st.session_state.clarification_attempts = st.session_state.get("clarification_attempts", 0) + 1
            state_manager.set_processing_status(True)
            with st.spinner("AIが確認のための質問を準備中です..."):
                q_ai_f = tutor_logic.generate_clarification_question_logic()
            state_manager.set_processing_status(False)

            if q_ai_f and "エラー" not in q_ai_f: # 成功時
                state_manager.add_message("assistant", q_ai_f)
                state_manager.add_clarification_history_message("assistant", q_ai_f)
            else: # エラーまたは不適切な応答時
                err_clar_q_f = q_ai_f or "明確化質問生成エラー。"
                state_manager.add_message("system", f"エラー(明確化質問): {err_clar_q_f}")
                state_manager.add_message("assistant", "確認質問の準備に問題がありました。現在の理解で進めさせていただきます。解説スタイルを選択してください。")
                st.session_state.is_request_ambiguous = False # 強制的に曖昧でない状態に
                state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
            st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_SELECT_STYLE:
        st.markdown("---")
        st.subheader("解説スタイルを選択してください")
        
        disp_req_style_sel_f = st.session_state.clarified_request_text or \
                               (st.session_state.initial_analysis_result.get("summary") if st.session_state.initial_analysis_result else None) or \
                               st.session_state.user_query_text
        if disp_req_style_sel_f:
            st.info(f"現在のリクエスト: 「{disp_req_style_sel_f}」")

        style_opts_sel_f = {"detailed": "詳しく(標準)", "hint": "ヒントのみ", "socratic": "質問形式で"}
        curr_style_sel_f = st.session_state.get("selected_explanation_style", "detailed")
        
        # インデックスの安全な取得
        current_style_index = 0
        try:
            current_style_index = list(style_opts_sel_f.keys()).index(curr_style_sel_f)
        except ValueError:
            # curr_style_sel_f が style_opts_sel_f のキーにない場合はデフォルトの0を使用
            pass

        sel_key_style_f = st.radio(
            "希望スタイル:", 
            list(style_opts_sel_f.keys()), 
            format_func=lambda k_sf: style_opts_sel_f[k_sf], 
            index=current_style_index,
            key="style_radio_tutor_ui_module", # キー変更
            horizontal=True
        )
        if st.button("このスタイルで解説生成", key="confirm_style_tutor_btn_ui_module", type="primary"): # キー変更
            state_manager.set_explanation_style(sel_key_style_f)
            state_manager.add_message("user", f"（スタイル「{style_opts_sel_f[sel_key_style_f]}」を選択）")
            state_manager.set_current_step(state_manager.STEP_GENERATE_EXPLANATION)
            st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_GENERATE_EXPLANATION:
        if st.session_state.current_explanation is None and not st.session_state.get("processing", False):
            state_manager.set_processing_status(True)
            with st.spinner("AIが解説準備中..."):
                exp_tutor_f = tutor_logic.generate_explanation_logic()
            state_manager.set_processing_status(False)
            
            if exp_tutor_f and "エラー" not in exp_tutor_f:
                state_manager.store_generated_explanation(exp_tutor_f)
                # 解説が生成されたら、それを表示するためにメッセージとして追加
                # state_manager.add_message("assistant", exp_tutor_f) # tutor_logic側でやるかここでやるか検討
                state_manager.set_current_step(state_manager.STEP_FOLLOW_UP_LOOP)
            else:
                err_msg_exp_f2 = exp_tutor_f or "解説生成エラー。"
                state_manager.add_message("system", f"エラー(解説生成): {err_msg_exp_f2}")
                # エラーの場合、ユーザーにフィードバックし、次のアクションを促す
                state_manager.add_message("assistant", "申し訳ありません、解説の生成中に問題が発生しました。もう一度試すか、質問を変えてみてください。")
                state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION) # または適切なエラーハンドリングステップへ
            st.rerun()
            
    elif current_step_tutor_main_final == state_manager.STEP_SUMMARIZE:
        if st.session_state.session_summary is None and \
           st.session_state.student_performance_analysis is None and \
           not st.session_state.get("processing", False):
            
            state_manager.set_processing_status(True)
            sum_txt_f2, ana_rep_f2 = None, None
            comb_parts_f2 = []

            with st.spinner("AIが要約と学習分析を準備中..."):
                sum_txt_f2 = tutor_logic.generate_summary_logic()
                if not sum_txt_f2 or "エラー" in sum_txt_f2:
                    state_manager.add_message("system", f"エラー(要約): {sum_txt_f2 or '失敗'}")
                else:
                    st.session_state.session_summary = sum_txt_f2
                    comb_parts_f2.append(f"【今回のまとめ】\n\n{sum_txt_f2}")
                
                ana_rep_f2 = tutor_logic.analyze_student_performance_logic()
                if not ana_rep_f2 or "エラー" in ana_rep_f2:
                    state_manager.add_message("system", f"エラー(分析): {ana_rep_f2 or '失敗'}")
                    st.session_state.student_performance_analysis = "分析失敗。" # エラーメッセージを格納
                    comb_parts_f2.append("【学習分析】\n\n申し訳ありません、学習分析の生成に失敗しました。")
                else:
                    st.session_state.student_performance_analysis = ana_rep_f2
                    comb_parts_f2.append(f"【学習分析 (β版)】\n\n{ana_rep_f2}")
            
            state_manager.set_processing_status(False)
            if comb_parts_f2:
                state_manager.add_message("assistant", "\n\n---\n\n".join(comb_parts_f2))
            
            state_manager.set_current_step(state_manager.STEP_SESSION_END)
            st.session_state.show_new_question_form = True # 新規質問フォームを自動表示
            st.rerun()

    elif current_step_tutor_main_final == state_manager.STEP_SESSION_END:
        # show_new_question_form が True の場合、上の入力フォームが表示される
        # そうでない場合（例：要約直後でまだボタンを押していない）はこちらのボタンが表示される
        if not st.session_state.get("show_new_question_form", False):
            if st.button("新しい質問をする", key="new_q_from_session_end_ui_module", use_container_width=True): # キー変更
                state_manager.reset_for_new_session()
                st.rerun()

    # AIチューターモードの共通チャット入力
    if current_step_tutor_main_final in [state_manager.STEP_CLARIFICATION_NEEDED, state_manager.STEP_FOLLOW_UP_LOOP]:
        # 「理解しました」ボタンの表示条件
        if current_step_tutor_main_final == state_manager.STEP_FOLLOW_UP_LOOP and \
           st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            if st.button("✅ 理解しました / 要約へ", key="understood_main_btn_ui_module"):
                state_manager.add_message("user", "（理解しました）")
                state_manager.set_current_step(state_manager.STEP_SUMMARIZE)
                st.rerun()

        chat_disabled = st.session_state.get("processing", False)
        user_chat_input = st.chat_input(
            "AIへの返答や追加質問など", 
            disabled=chat_disabled, 
            key="main_tutor_chat_input_area_ui_module"
        )

        if user_chat_input:
            state_manager.add_message("user", user_chat_input)
            
            if current_step_tutor_main_final == state_manager.STEP_CLARIFICATION_NEEDED:
                # ★明確化応答の分析を完全にスキップし、ユーザー応答を clarified_request_text に直接セットして次のステップへ★
                st.session_state.clarified_request_text = user_chat_input
                st.session_state.is_request_ambiguous = False # 応答があった時点で曖昧さ解消とみなす
                state_manager.add_clarification_history_message("user", user_chat_input)
                state_manager.set_current_step(state_manager.STEP_PLAN_GUIDANCE)
                st.rerun()

            elif current_step_tutor_main_final == state_manager.STEP_FOLLOW_UP_LOOP:
                state_manager.set_processing_status(True)
                with st.spinner("AIが応答準備中..."):
                    followup_response = tutor_logic.generate_followup_response_logic(user_chat_input)
                state_manager.set_processing_status(False)
                
                if followup_response and "エラー" not in followup_response:
                    state_manager.add_message("assistant", followup_response)
                else:
                    error_msg_fu = followup_response or "フォローアップ応答の生成に失敗しました。"
                    state_manager.add_message("system", f"エラー(フォローアップ): {error_msg_fu}")
                    state_manager.add_message("assistant", "申し訳ありません、応答の準備中に問題が発生しました。")
                st.rerun()