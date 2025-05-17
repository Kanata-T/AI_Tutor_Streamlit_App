# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
# from PIL import Image # Image を直接ここで使う必要はなくなるかも
from typing import Optional, Dict, Any # Dict, Any を追加

# coreモジュールのインポート (まとめてインポートできるように __init__.py を利用)
from core import state_manager, tutor_logic # tutor_logic をインポート
from core.type_definitions import UploadedFileData, InitialAnalysisResult, ClarificationAnalysisResult, ChatMessage # 型定義をインポート

# utilsモジュールのインポート (image_processorから必要な関数をインポート)
from utils.image_processor import preprocess_uploaded_image
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
st.set_page_config(page_title="AI学習チューター", layout="wide")
st.title("AI学習チューター プロトタイプ")

# セッション状態の初期化
state_manager.initialize_session_state()

# --- UIヘルパー関数 (任意) ---
def display_analysis_result(analysis_data: Dict[str, Any], title: str = "分析結果"):
    """分析結果を整形してエキスパンダー内に表示する"""
    with st.expander(title, expanded=False):
        if isinstance(analysis_data, dict):
            if "error" in analysis_data:
                st.error(f"分析中にエラーが発生しました: {analysis_data['error']}")
            else:
                st.json(analysis_data)
            if "ocr_text_from_extraction" in analysis_data:
                st.text_area(
                    "初期分析時のOCR抽出テキスト:",
                    analysis_data.get("ocr_text_from_extraction"),
                    height=100,
                    key=f"expander_ocr_{title.replace(' ', '_').lower()}"
                )
        else:
            st.write(analysis_data)

# --- 会話履歴表示コンテナ --- 
# 必ず最初に配置し、このコンテナにメッセージが追加されていく
message_container = st.container(height=500, border=False)
with message_container:
    if not st.session_state.messages:
        st.info("AI学習チューターへようこそ！下の入力欄から質問をどうぞ。")
    else:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                content = message.get("content")
                if isinstance(content, dict) and "type" in content:
                    if content["type"] == "analysis_result":
                        display_analysis_result(content["data"], content.get("title", f"分析結果 {i}"))
                elif isinstance(content, str):
                    st.markdown(content)
                else:
                    st.write(str(content))

# --- 現在のステップに応じた処理と入力UIの表示 --- 
current_step = state_manager.get_current_step()

# 1. ユーザー入力受付ステップ (またはセッション終了後の新規質問受付)
if current_step == state_manager.STEP_INPUT_SUBMISSION or \
   (current_step == state_manager.STEP_SESSION_END and st.session_state.get("show_new_question_form", True)):

    if current_step == state_manager.STEP_SESSION_END:
        st.success("学習セッションが終了しました。新しい質問があれば、以下に入力してください。")
        if st.button("新しい質問の準備 (状態リセット)", key="reset_from_end_form_prep"):
            state_manager.reset_for_new_session()
            st.session_state.show_new_question_form = True
            st.rerun()

    with st.form("submission_form", clear_on_submit=True):
        user_query_text_input = st.text_input(
            "質問内容を入力してください:",
            key="query_text_main_input_form"
        )
        uploaded_file_obj = st.file_uploader(
            "画像をアップロード (任意):",
            type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
            key="file_uploader_main_form"
        )
        topic_options = ["", "文法", "語彙", "長文読解", "英作文", "その他"]
        selected_topic_input = st.selectbox(
            "トピックを選択 (任意):",
            topic_options,
            key="topic_main_input_form"
        )
        submitted = st.form_submit_button("この内容で質問する")

        if submitted:
            if not user_query_text_input and not uploaded_file_obj:
                st.warning("質問内容のテキスト入力または画像のアップロードのいずれかを行ってください。")
            else:
                if current_step == state_manager.STEP_SESSION_END:
                    state_manager.reset_for_new_session()

                processed_image_data_for_state = None
                if uploaded_file_obj is not None:
                    raw_image_bytes = uploaded_file_obj.getvalue()
                    raw_mime_type = uploaded_file_obj.type
                    print(f"[App] Preprocessing image: {uploaded_file_obj.name}, type: {raw_mime_type}, size: {len(raw_image_bytes)/1024:.1f}KB")
                    processed_image_info = preprocess_uploaded_image(
                        uploaded_file_data=raw_image_bytes,
                        mime_type=raw_mime_type
                    )

                    if processed_image_info and "error" not in processed_image_info:
                        processed_image_data_for_state = processed_image_info
                        state_manager.add_message("system", f"システム: 画像を処理しました (形式: {processed_image_info['mime_type']}, サイズ: {len(processed_image_info['data'])/1024:.1f}KB)。")
                        st.caption(f"DEBUG - Processed Image: MIME Type: {processed_image_info['mime_type']}, Data Size: {len(processed_image_info['data'])} bytes")
                    elif processed_image_info and "error" in processed_image_info:
                        st.error(f"画像処理エラー: {processed_image_info['error']}")
                        state_manager.add_message("system", f"システムエラー(画像処理): {processed_image_info['error']}")
                    else:
                        st.error("画像処理中に予期せぬエラーが発生しました。")
                        state_manager.add_message("system", "システムエラー(画像処理): 予期せぬエラー。")
                
                state_manager.store_user_input(
                    user_query_text_input,
                    processed_image_data_for_state,
                    selected_topic_input
                )
                
                user_message_content = f"質問: {user_query_text_input}"
                if processed_image_data_for_state:
                    user_message_content += " (画像あり)"
                state_manager.add_message("user", user_message_content)
                
                state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS)
                st.session_state.show_new_question_form = False
                st.rerun()

# 2. 初期分析ステップ
elif current_step == state_manager.STEP_INITIAL_ANALYSIS:
    if st.session_state.initial_analysis_result is None and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True)
        with st.spinner("AIがあなたの質問を分析中です..."):
            analysis_result = tutor_logic.perform_initial_analysis_logic()
        state_manager.set_processing_status(False)

        if analysis_result:
            if "error" in analysis_result:
                err_msg = analysis_result.get('error', "不明な分析エラー")
                st.error(f"分析エラー: {err_msg}")
                state_manager.add_message("system", f"エラー(初期分析): {err_msg}")
                state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
                st.session_state.show_new_question_form = True
            else:
                state_manager.store_initial_analysis_result(analysis_result)
                state_manager.add_message(
                    "system",
                    {"type": "analysis_result", "data": dict(analysis_result), "title": "AIによる初期分析結果"}
                )
                if st.session_state.is_request_ambiguous:
                    state_manager.set_current_step(state_manager.STEP_CLARIFICATION_NEEDED)
                else:
                    state_manager.add_message("assistant", "ご質問内容を理解しました。どのようなスタイルの解説がご希望ですか？ (この下に選択肢が表示されます)")
                    state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
            st.rerun()
        else:
            st.error("質問の分析処理中に予期せぬエラーが発生しました。お手数ですが、もう一度お試しください。")
            state_manager.add_message("system", "エラー(初期分析): 分析結果がNoneでした。")
            state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
            st.session_state.show_new_question_form = True
            st.rerun()
    # --- デバッグ表示 (任意) ---
    if st.session_state.get("debug_show_input", False): # デバッグ表示用のフラグ (任意)
        st.write("ユーザーの質問 (セッションステートより):")
        st.write(f"- テキスト: {st.session_state.user_query_text}")
        st.write(f"- トピック: {st.session_state.selected_topic}")
        if st.session_state.uploaded_file_data and "data" in st.session_state.uploaded_file_data: # ★ 形式変更に対応 ★
            st.write("- 処理済みアップロード画像:")
            st.image(st.session_state.uploaded_file_data["data"], caption=f"Type: {st.session_state.uploaded_file_data['mime_type']}")

# 3. 明確化ステップ (AIからの質問表示は会話履歴で、ユーザー応答は下部のチャット入力)
elif current_step == state_manager.STEP_CLARIFICATION_NEEDED:
    needs_clarification_q_generation = False
    if not st.session_state.messages or st.session_state.messages[-1]["role"] == "user" or \
       (st.session_state.messages[-1]["role"] == "system" and "analysis_result" in st.session_state.messages[-1].get("content", {}).get("type","")):
        if st.session_state.get("clarification_attempts", 0) == 0:
            needs_clarification_q_generation = True

    if needs_clarification_q_generation and not st.session_state.get("processing", False):
        st.session_state.clarification_attempts = st.session_state.get("clarification_attempts", 0) + 1
        state_manager.set_processing_status(True)
        with st.spinner("AIが確認のための質問を準備中です..."):
            question_from_ai = tutor_logic.generate_clarification_question_logic()
        state_manager.set_processing_status(False)

        if question_from_ai and "システムエラー" not in question_from_ai and "生成できませんでした" not in question_from_ai:
            state_manager.add_message("assistant", question_from_ai)
            state_manager.add_clarification_history_message("assistant", question_from_ai)
        else:
            error_msg = question_from_ai or "明確化のための質問をAIが生成できませんでした。"
            state_manager.add_message("system", f"エラー(明確化質問生成失敗): {error_msg}")
            state_manager.add_message("assistant", "申し訳ありません、確認のための質問準備に問題がありました。現在の情報で解説に進むか、スタイルを選択してください。")
            st.session_state.is_request_ambiguous = False
            state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
        st.rerun()

# 4. スタイル選択ステップ
elif current_step == state_manager.STEP_SELECT_STYLE:
    st.markdown("---")
    st.subheader("解説スタイルを選択してください")
    display_request = st.session_state.clarified_request_text or \
                      (st.session_state.initial_analysis_result.get("summary") if st.session_state.initial_analysis_result else None) or \
                      st.session_state.user_query_text
    if display_request:
        st.info(f"現在のリクエスト: 「{display_request}」")

    style_options = {
        "detailed": "詳しく解説してほしい (標準)",
        "hint": "ヒントだけ教えてほしい",
        "socratic": "質問形式で一緒に考えてほしい"
    }
    current_style = st.session_state.get("selected_explanation_style", "detailed")
    
    selected_style_key = st.radio(
        "希望する解説のスタイルを選んでください:",
        options=list(style_options.keys()),
        format_func=lambda key: style_options[key],
        index=list(style_options.keys()).index(current_style) if current_style in style_options else 0,
        key="style_radio_main",
        horizontal=True,
    )
    
    if st.button("このスタイルで解説を生成する", key="confirm_style_button_main", type="primary"):
        state_manager.set_explanation_style(selected_style_key)
        state_manager.add_message("user", f"（解説スタイルとして「{style_options[selected_style_key]}」を選択）")
        state_manager.set_current_step(state_manager.STEP_GENERATE_EXPLANATION)
        st.rerun()

# 5. 解説生成ステップ
elif current_step == state_manager.STEP_GENERATE_EXPLANATION:
    if st.session_state.current_explanation is None and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True)
        with st.spinner("AIが解説を準備中です... しばらくお待ちください。"):
            explanation = tutor_logic.generate_explanation_logic()
        state_manager.set_processing_status(False)

        if explanation and "システムエラー" not in explanation and "生成できませんでした" not in explanation:
            state_manager.store_generated_explanation(explanation)
            state_manager.set_current_step(state_manager.STEP_FOLLOW_UP_LOOP)
        else:
            error_msg = explanation or "解説の生成に失敗しました。もう一度スタイル選択からお試しください。"
            state_manager.add_message("system", f"エラー(解説生成失敗): {error_msg}")
        st.rerun()

# 6. フォローアップループ (ユーザーからの追加質問待ち)
# 7. 理解確認 (現在はフォローアップループ内で「理解しました」ボタンにより遷移)
# 8. 要約 (「理解しました」ボタンから遷移)
elif current_step == state_manager.STEP_SUMMARIZE:
    # st.header("8. 要約と持ち帰りメッセージ") # チャット形式ではヘッダーは不要かも
    if st.session_state.session_summary is None and \
       st.session_state.student_performance_analysis is None and \
       not st.session_state.get("processing", False):

        state_manager.set_processing_status(True)
        summary_text = None
        analysis_report = None
        combined_message_parts = []
        error_occurred_summary = False
        error_occurred_analysis = False

        with st.spinner("AIがセッションの要約と学習分析を準備中です..."):
            summary_text = tutor_logic.generate_summary_logic()
            if not summary_text or "システムエラー" in summary_text or "生成できませんでした" in summary_text:
                error_msg_summary = summary_text or "要約の生成に失敗しました。"
                state_manager.add_message("system", f"エラー(要約生成失敗): {error_msg_summary}")
                error_occurred_summary = True
            else:
                st.session_state.session_summary = summary_text
                combined_message_parts.append(f"【今回の学習のまとめ】\n\n{summary_text}")

            analysis_report = tutor_logic.analyze_student_performance_logic()
            if not analysis_report or "システムエラー" in analysis_report or "生成できませんでした" in analysis_report:
                error_msg_analysis = analysis_report or "学習分析レポートの生成に失敗しました。"
                state_manager.add_message("system", f"エラー(学習分析失敗): {error_msg_analysis}")
                st.session_state.student_performance_analysis = "分析レポートの生成に失敗しました。"
                combined_message_parts.append("【学習の理解度分析】\n\n分析レポートの生成に失敗しました。")
                error_occurred_analysis = True
            else:
                st.session_state.student_performance_analysis = analysis_report
                combined_message_parts.append(f"【学習の理解度分析とCEFRレベル推定 (β版)】\n\n{analysis_report}")
        
        state_manager.set_processing_status(False)

        if combined_message_parts:
            final_message_to_user = "\n\n---\n\n".join(combined_message_parts)
            state_manager.add_message("assistant", final_message_to_user)
        
        state_manager.set_current_step(state_manager.STEP_SESSION_END)
        st.session_state.show_new_question_form = True
        st.rerun()

# 9. セッション終了ステップ (新しい質問または分析の選択)
elif current_step == state_manager.STEP_SESSION_END:
    # st.success("学習セッションが終了しました。お疲れ様でした！") # メッセージは会話履歴に追加済み
    # このステップでは、新しい質問を始めるためのUIのみ表示
    # 学習のまとめと分析は、会話履歴に表示されている想定

    if st.button("新しい質問を始める", key="new_session_button_main_end", use_container_width=True):
        state_manager.reset_for_new_session() 
        # student_performance_analysis は reset_for_new_session 内でリセットされる
        st.rerun()

# --- 共通チャット入力UI --- 
# 明確化ループ、フォローアップループ、理解しましたボタン(フォローアップ中のみ)
if current_step == state_manager.STEP_FOLLOW_UP_LOOP:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button("✅ 理解しました / セッションを終了して要約へ", key="understood_button_main_chat"):
            state_manager.add_message("user", "（理解しました）")
            state_manager.set_current_step(state_manager.STEP_SUMMARIZE)
            st.rerun()

if current_step in [state_manager.STEP_CLARIFICATION_NEEDED, state_manager.STEP_FOLLOW_UP_LOOP]:
    is_chat_disabled = st.session_state.get("processing", False)
    if user_input := st.chat_input("AIへの返答や追加の質問を入力してください...", disabled=is_chat_disabled, key="main_chat_input_area"):
        state_manager.add_message("user", user_input)
        
        if current_step == state_manager.STEP_CLARIFICATION_NEEDED:
            state_manager.add_clarification_history_message("user", user_input)
            state_manager.set_processing_status(True)
            with st.spinner("AIがあなたの応答を分析中です..."):
                clarification_analysis_result = tutor_logic.analyze_user_clarification_logic(user_input)
            state_manager.set_processing_status(False)
            
            if clarification_analysis_result:
                state_manager.add_message(
                    "system",
                    {"type": "analysis_result", "data": dict(clarification_analysis_result), "title": "明確化応答の分析結果"}
                )
                if "error" in clarification_analysis_result:
                    err_msg = clarification_analysis_result.get('error', "応答分析エラー")
                    state_manager.add_message("assistant", f"申し訳ありません、応答の分析に問題がありました ({err_msg})。現在の情報で解説に進みます。")
                    st.session_state.is_request_ambiguous = False
                    state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                else:
                    state_manager.store_clarification_analysis(clarification_analysis_result)
                    if not st.session_state.is_request_ambiguous:
                        ai_resp = clarification_analysis_result.get(
                            "ai_response_to_user",
                            "ありがとうございます、理解が深まりました。どのようなスタイルの解説が良いか教えてください。"
                        )
                        state_manager.add_message("assistant", ai_resp)
                        state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                    else:
                        reason_remaining = clarification_analysis_result.get("remaining_issue", "まだ少し確認したい点があります。")
                        state_manager.add_message("assistant", f"ありがとうございます。もう少しだけ確認させてください: {reason_remaining}")
            else:
                state_manager.add_message("system", "エラー(明確化応答分析): 結果がNoneでした。")
                state_manager.add_message("assistant", "申し訳ありません、応答の分析に問題がありました。現在の情報で解説に進みます。")
                st.session_state.is_request_ambiguous = False
                state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
            st.rerun()

        elif current_step == state_manager.STEP_FOLLOW_UP_LOOP:
            state_manager.set_processing_status(True)
            with st.spinner("AIがフォローアップの応答を準備中です..."):
                followup_response = tutor_logic.generate_followup_response_logic(user_input)
            state_manager.set_processing_status(False)
            
            if followup_response and "システムエラー" not in followup_response and "生成できませんでした" not in followup_response :
                state_manager.add_message("assistant", followup_response)
            else:
                error_msg = followup_response or "AIが応答を生成できませんでした。"
                state_manager.add_message("system", f"エラー(フォローアップ): {error_msg}")
            st.rerun()

# --- デバッグ情報 (サイドバーに表示) ---
with st.sidebar:
    st.header("デバッグ情報")
    st.write(f"現在のステップ: `{current_step}`")
    st.write(f"処理中フラグ: `{st.session_state.get('processing', False)}`")
    st.write(f"曖昧フラグ: `{st.session_state.get('is_request_ambiguous', False)}`")
    st.write(f"明確化試行回数: `{st.session_state.get('clarification_attempts', 0)}`")

    with st.expander("セッションステート全体 (JSON)", expanded=False):
        try:
            session_state_display = {
                k: (v if k != "uploaded_file_data" or v is None else
                    {kk: (vv if kk != "data" else f"<bytes data of length {len(vv)}>" if isinstance(vv, bytes) else vv)
                     for kk, vv in v.items()}
                   )
                for k, v in st.session_state.items()
            }
            st.json(session_state_display)
        except Exception as e:
            st.error(f"セッション状態の表示中にエラー: {e}")