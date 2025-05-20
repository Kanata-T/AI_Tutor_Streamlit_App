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

# uiヘルパーのインポート
from .display_helpers import display_analysis_result # 相対インポート

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
                st.rerun()

        with st.form("submission_form_tutor_main_ui_module", clear_on_submit=True): # キー変更
            st.markdown("#### 質問を入力してください:")
            user_query_text_tf_f = st.text_input("質問テキスト", key="main_query_text_ui_module", label_visibility="collapsed")
            uploaded_file_tf_f = st.file_uploader(
                "画像 (任意)", 
                type=["png","jpg","jpeg","webp","gif","bmp"], 
                key="main_file_uploader_ui_module"
            )
            topic_opts_tf_f = ["", "文法", "語彙", "長文読解", "英作文", "その他"]
            selected_topic_tf_f = st.selectbox("トピック (任意)", topic_opts_tf_f, key="main_topic_select_ui_module")
            submit_btn_tf_f = st.form_submit_button("この内容で質問する")

            if submit_btn_tf_f:
                if not user_query_text_tf_f and not uploaded_file_tf_f:
                    st.warning("テキスト入力または画像アップロードのいずれかを行ってください。")
                else:
                    if current_step_tutor_main_final == state_manager.STEP_SESSION_END:
                        state_manager.reset_for_new_session() # 新しいセッションのためにリセット

                    ocr_input_image_for_llm = None 
                    debug_imgs_for_llm_run_final = []

                    if uploaded_file_tf_f:
                        raw_bytes_tutor = uploaded_file_tf_f.getvalue()
                        mime_type_tutor = uploaded_file_tf_f.type
                        processed_image_bytes_for_preprocessing = None
                        try:
                            print(f"[TutorModeUI] Attempting OpenCV auto orientation for tutor mode: {uploaded_file_tf_f.name}")
                            img_pil_tutor_temp = Image.open(BytesIO(raw_bytes_tutor))
                            img_pil_auto_oriented_tutor = auto_orient_image_opencv(img_pil_tutor_temp)
                            original_mime_tutor = mime_type_tutor.lower()
                            output_fmt_for_session_tutor: str = "PNG"
                            if original_mime_tutor == "image/jpeg":
                                output_fmt_for_session_tutor = "JPEG"
                            elif original_mime_tutor == "image/webp":
                                try:
                                    Image.open(BytesIO(b'')).save(BytesIO(), format='WEBP')
                                    output_fmt_for_session_tutor = "WEBP"
                                except Exception:
                                    output_fmt_for_session_tutor = "PNG"
                            jpeg_q_tutor = st.session_state.get("tuning_fixed_other_params", {}).get("jpeg_quality", 85)
                            processed_image_bytes_for_preprocessing = image_to_bytes(
                                img_pil_auto_oriented_tutor,
                                target_format=output_fmt_for_session_tutor,
                                jpeg_quality=jpeg_q_tutor
                            )
                            if not processed_image_bytes_for_preprocessing:
                                st.warning("AIチューター: 画像の向き補正後のバイト変換に失敗。元の画像で処理試行。")
                                processed_image_bytes_for_preprocessing = raw_bytes_tutor
                            else:
                                print("[TutorModeUI] Auto orientation successful for tutor mode.")
                        except Exception as e_orient_tutor:
                            st.error(f"AIチューター: 画像の自動向き補正中にエラー: {e_orient_tutor}")
                            processed_image_bytes_for_preprocessing = raw_bytes_tutor
                        if processed_image_bytes_for_preprocessing:
                            print(f"[TutorModeUI] Preprocessing image for LLM (after auto-orientation attempt): {uploaded_file_tf_f.name}")
                            fixed_cv = st.session_state.tuning_fixed_cv_params
                            fixed_other = st.session_state.tuning_fixed_other_params
                            preprocessing_res_llm_final = preprocess_uploaded_image(
                                uploaded_file_data=processed_image_bytes_for_preprocessing,
                                mime_type=mime_type_tutor,
                                max_pixels=fixed_other["max_pixels"],
                                output_format=fixed_other["output_format"],
                                jpeg_quality=fixed_other["jpeg_quality"],
                                grayscale=fixed_other["grayscale"],
                                apply_trimming_opencv_override=fixed_cv.get("apply"),
                                trim_params_override=fixed_cv
                            )
                            if preprocessing_res_llm_final and "error" not in preprocessing_res_llm_final:
                                ocr_input_image_for_llm = preprocessing_res_llm_final.get("ocr_input_image")
                                debug_imgs_for_llm_run_final = preprocessing_res_llm_final.get("debug_images", [])
                                if ocr_input_image_for_llm:
                                    state_manager.add_message("system", f"画像処理完了 (OCR入力形式: {ocr_input_image_for_llm.get('mime_type', 'N/A')})")
                                else:
                                    st.warning("OCR入力用画像の生成に問題がありましたが、処理を継続します。")
                                    ocr_input_image_for_llm = preprocessing_res_llm_final.get("processed_image")
                                    if ocr_input_image_for_llm:
                                        state_manager.add_message("system", f"画像処理完了 (フォールバック形式: {ocr_input_image_for_llm.get('mime_type', 'N/A')})")
                            elif preprocessing_res_llm_final and "error" in preprocessing_res_llm_final:
                                st.error(f"画像処理エラー: {preprocessing_res_llm_final['error']}")
                            else:
                                st.error("画像処理で予期せぬエラーが発生しました。")
                        else:
                            st.error("AIチューター: 画像データの準備に失敗しました。")
                    
                    st.session_state.last_debug_images_tutor_run_final_v3 = debug_imgs_for_llm_run_final # デバッグ用
                    state_manager.store_user_input(user_query_text_tf_f, ocr_input_image_for_llm, selected_topic_tf_f)
                    
                    user_msg_content_final = f"質問: {user_query_text_tf_f}" + (" (画像あり)" if ocr_input_image_for_llm else "")
                    state_manager.add_message("user", user_msg_content_final)
                    state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS)
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
        if current_step_tutor_main_final == state_manager.STEP_FOLLOW_UP_LOOP and \
           st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            # 「理解しました」ボタンは、AIの応答後に表示
            if st.button("✅ 理解しました / 要約へ", key="understood_main_btn_ui_module"): # キー変更
                state_manager.add_message("user", "（理解しました）")
                state_manager.set_current_step(state_manager.STEP_SUMMARIZE)
                st.rerun()

        chat_disabled_tf_f = st.session_state.get("processing", False)
        user_chat_input_tf_f = st.chat_input(
            "AIへの返答や追加質問など", 
            disabled=chat_disabled_tf_f, 
            key="main_tutor_chat_input_area_ui_module" # キー変更
        )

        if user_chat_input_tf_f:
            state_manager.add_message("user", user_chat_input_tf_f)
            
            if current_step_tutor_main_final == state_manager.STEP_CLARIFICATION_NEEDED:
                state_manager.add_clarification_history_message("user", user_chat_input_tf_f)
                state_manager.set_processing_status(True)
                with st.spinner("AIがあなたの応答を分析中です..."):
                    clar_res_f2 = tutor_logic.analyze_user_clarification_logic(user_chat_input_tf_f)
                state_manager.set_processing_status(False)

                if clar_res_f2:
                    state_manager.add_message("system", {"type": "analysis_result", "data": dict(clar_res_f2), "title": "明確化応答の分析結果"})
                    if "error" in clar_res_f2:
                        state_manager.add_message("assistant", "応答の分析中に問題がありました。現在の理解で進めさせていただきます。")
                        st.session_state.is_request_ambiguous = False
                        state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                    else:
                        state_manager.store_clarification_analysis(clar_res_f2) # 分析結果を保存
                        if not st.session_state.is_request_ambiguous: # 曖昧さが解消された
                            state_manager.add_message("assistant", clar_res_f2.get("ai_response_to_user","理解しました。解説のスタイルを選択してください。"))
                            state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                        else: # まだ曖昧さが残っている
                            state_manager.add_message("assistant", f"もう少し確認させてください: {clar_res_f2.get('remaining_issue', '不明な点が残っています。')}")
                            # clarification_attempts をここでインクリメントしても良い（tutor_logic側でも制御可能）
                else: # clar_res_f2 が None の場合など
                    state_manager.add_message("system", "エラー(明確化応答分析): 分析結果がありませんでした。")
                    state_manager.add_message("assistant", "応答の分析に問題がありました。現在の理解で進めさせていただきます。")
                    st.session_state.is_request_ambiguous = False
                    state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                st.rerun()

            elif current_step_tutor_main_final == state_manager.STEP_FOLLOW_UP_LOOP:
                state_manager.set_processing_status(True)
                with st.spinner("AIが応答準備中..."):
                    followup_resp_f2 = tutor_logic.generate_followup_response_logic(user_chat_input_tf_f)
                state_manager.set_processing_status(False)
                
                if followup_resp_f2 and "エラー" not in followup_resp_f2:
                    state_manager.add_message("assistant", followup_resp_f2)
                else:
                    err_msg_fu = followup_resp_f2 or "フォローアップ応答の生成に失敗しました。"
                    state_manager.add_message("system", f"エラー(フォローアップ): {err_msg_fu}")
                    state_manager.add_message("assistant", "申し訳ありません、応答の準備中に問題が発生しました。")
                st.rerun()