# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image # Pillowをインポート

# coreモジュールのインポート (まとめてインポートできるように __init__.py を利用)
from core import state_manager, tutor_logic # tutor_logic をインポート

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

# --- Streamlit UI ---
st.set_page_config(page_title="AI学習チューター", layout="wide")
st.title("AI学習チューター プロトタイプ")

# セッション状態の初期化
state_manager.initialize_session_state()

current_step = state_manager.get_current_step()

if current_step == state_manager.STEP_INPUT_SUBMISSION:
    st.header("1. 質問を入力してください")
    with st.form("submission_form"):
        user_query_text_input = st.text_input(
            "質問内容を入力してください",
            value=st.session_state.user_query_text
        )
        uploaded_file_obj = st.file_uploader(
            "画像をアップロード (任意)",
            type=["png", "jpg", "jpeg"]
        )
        selected_topic_input = st.selectbox(
            "トピックを選択 (任意)",
            ["", "文法", "語彙", "長文読解", "英作文", "その他"],
            index=["", "文法", "語彙", "長文読解", "英作文", "その他"].index(st.session_state.selected_topic) if st.session_state.selected_topic else 0
        )
        submitted = st.form_submit_button("質問を送信")

        if submitted:
            if not user_query_text_input and not uploaded_file_obj:
                st.warning("質問内容のテキスト入力または画像のアップロードのいずれかを行ってください。")
            else:
                image_data_for_state = None
                if uploaded_file_obj is not None:
                    try:
                        image_data_for_state = {
                            "mime_type": uploaded_file_obj.type,
                            "data": uploaded_file_obj.getvalue()
                        }
                        # st.success("画像を読み込みました。") # 送信ボタン押下後のメッセージは集約
                    except Exception as e:
                        st.error(f"画像の読み込みに失敗しました: {e}")
                        image_data_for_state = None
                
                state_manager.store_user_input(
                    user_query_text_input,
                    image_data_for_state,
                    selected_topic_input
                )
                
                user_message_content = f"質問: {user_query_text_input}"
                if image_data_for_state:
                    user_message_content += " (画像あり)"
                state_manager.add_message("user", user_message_content)
                
                state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS)
                st.rerun()

elif current_step == state_manager.STEP_INITIAL_ANALYSIS:
    st.header("2. 初期分析中です...")
    # (デバッグ用) 保存されたユーザー情報を表示
    # st.write("ユーザーの質問 (セッションステートより):")
    # st.write(f"- テキスト: {st.session_state.user_query_text}")
    # st.write(f"- トピック: {st.session_state.selected_topic}")
    # if st.session_state.uploaded_file_data:
    #     st.write("- アップロードされた画像:")
    #     st.image(st.session_state.uploaded_file_data["data"])

    # 初期分析処理がまだ実行されていない場合のみ実行する (st.rerun対策)
    if st.session_state.initial_analysis_result is None and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True) # 処理中フラグを立てる
        with st.spinner("AIが分析中です... しばらくお待ちください。"):
            # tutor_logic の関数を呼び出す
            analysis_result = tutor_logic.perform_initial_analysis_logic()
        state_manager.set_processing_status(False) # 処理中フラグを下ろす

        if analysis_result:
            if "error" in analysis_result:
                st.error(f"分析エラー: {analysis_result['error']}")
                # エラー内容によっては前のステップに戻すなどの処理も検討
                # state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
                # st.session_state.error_message = analysis_result['error'] # state_managerに関数作っても良い
            else:
                state_manager.store_initial_analysis_result(analysis_result)
                # st.success("初期分析が完了しました！") # デバッグ用
                # 曖昧さに基づいて次のステップを決定
                if st.session_state.is_request_ambiguous:
                    state_manager.add_message("assistant", "ご質問について、もう少し詳しく教えていただけますか？ (初期分析の結果、曖昧な点がありました)") # 仮のメッセージ
                    state_manager.set_current_step(state_manager.STEP_CLARIFICATION_NEEDED)
                else:
                    state_manager.add_message("assistant", "ご質問内容を理解しました。解説の準備をします。 (初期分析完了)") # 仮のメッセージ
                    state_manager.set_current_step(state_manager.STEP_SELECT_STYLE) # 仮にスタイル選択へ
                st.rerun() # 分析結果を反映して再描画
        else: # analysis_result が None の場合 (API呼び出し以前の致命的エラーなど)
            st.error("分析処理中に予期せぬエラーが発生しました。")
            # state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
            # st.rerun()

    # 分析結果の表示 (デバッグ用)
    if st.session_state.initial_analysis_result and "error" not in st.session_state.initial_analysis_result:
        st.subheader("初期分析結果:")
        st.json(st.session_state.initial_analysis_result)
    elif st.session_state.initial_analysis_result and "error" in st.session_state.initial_analysis_result:
        st.warning("分析結果にエラーが含まれています。上記のエラーメッセージを確認してください。")
    
    # 次のステップに進むためのボタンなどをここに配置しても良い (例: エラー時に再試行ボタン)

elif current_step == state_manager.STEP_CLARIFICATION_NEEDED:
    st.header("3. ご質問の明確化")

    # AIからの明確化質問がまだ表示されていなければ生成・表示
    needs_clarification_q_generation = True
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant":
            needs_clarification_q_generation = False

    if needs_clarification_q_generation and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True)
        with st.spinner("AIが確認のための質問を準備中です..."):
            question_from_ai = tutor_logic.generate_clarification_question_logic()
        state_manager.set_processing_status(False)

        if question_from_ai and "システムエラー" not in question_from_ai and "生成できませんでした" not in question_from_ai:
            state_manager.add_message("assistant", question_from_ai)
            st.rerun()
        else:
            error_msg = question_from_ai or "明確化のための質問を生成できませんでした。"
            state_manager.add_message("system", f"エラー: {error_msg}")
            st.error(error_msg)

elif current_step == state_manager.STEP_SELECT_STYLE:
    st.header("4. 解説スタイルを選択してください")

    # ユーザーに明確化されたリクエスト内容を表示 (確認のため)
    if st.session_state.clarified_request_text:
        st.info(f"明確化されたご要望: 「{st.session_state.clarified_request_text}」")
    elif st.session_state.initial_analysis_result and not st.session_state.is_request_ambiguous:
        # 初期分析で曖昧でなかった場合
        st.info(f"ご要望: 「{st.session_state.initial_analysis_result.get('summary', st.session_state.user_query_text)}」")
    else:
        st.warning("解説対象のリクエストが不明です。最初からやり直してください。")
        if st.button("最初に戻る"):
            state_manager.reset_for_new_session()
            st.rerun()
        st.stop()

    style_options = {
        "detailed": "詳しく解説 (Detailed)",
        "hint": "ヒントのみ (Hint Only)",
        "socratic": "ソクラテス的対話 (Socratic Questioning)"
    }
    
    # 現在選択されているスタイルを取得
    current_selected_style_key = st.session_state.get("selected_explanation_style", "detailed")
    # style_optionsのキーのリストを取得し、現在のスタイルのインデックスを見つける
    options_keys = list(style_options.keys())
    current_index = options_keys.index(current_selected_style_key) if current_selected_style_key in options_keys else 0

    selected_style_key = st.radio(
        "希望する解説のスタイルを選んでください:",
        options=options_keys, # 辞書のキーをオプションとして渡す
        format_func=lambda key: style_options[key], # 表示は辞書のバリューを使う
        index=current_index, # デフォルトで選択されている項目
        horizontal=True,
    )

    if st.button("このスタイルで解説を生成する"):
        state_manager.set_explanation_style(selected_style_key)
        state_manager.add_message("user", f"解説スタイルとして「{style_options[selected_style_key]}」を選択しました。")
        state_manager.set_current_step(state_manager.STEP_GENERATE_EXPLANATION)
        st.rerun()

# --- 会話履歴の表示 (変更なし) ---
if st.session_state.messages:
    st.subheader("会話履歴")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- チャット入力 (ユーザーが応答や追加質問をするためのもの) ---
if current_step in [state_manager.STEP_CLARIFICATION_NEEDED, state_manager.STEP_FOLLOW_UP_LOOP]:
    user_response = st.chat_input("AIへの返答や追加の質問を入力してください...")
    if user_response:
        state_manager.add_message("user", user_response)
        # state_manager.add_clarification_history_message("user", user_response) # 専用履歴

        if current_step == state_manager.STEP_CLARIFICATION_NEEDED:
            # ユーザーが明確化質問に応答したので、その応答を分析する
            state_manager.set_processing_status(True)
            with st.spinner("AIが応答を分析中です..."):
                clarification_analysis_result = tutor_logic.analyze_user_clarification_logic(user_response)
            state_manager.set_processing_status(False)

            if clarification_analysis_result:
                if "error" in clarification_analysis_result:
                    st.error(f"応答分析エラー: {clarification_analysis_result['error']}")
                    state_manager.add_message("system", f"エラー: {clarification_analysis_result['error']}")
                else:
                    state_manager.store_clarification_analysis(clarification_analysis_result)
                    if not st.session_state.is_request_ambiguous:
                        success_msg = f"ありがとうございます、理解が深まりました！明確化されたご要望: 「{st.session_state.clarified_request_text}」"
                        state_manager.add_message("assistant", success_msg)
                        state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                    else:
                        remaining_issue_msg = st.session_state.initial_analysis_result.get("reason_for_ambiguity", "もう少し情報が必要です。")
                        state_manager.add_message("assistant", f"ありがとうございます。ただ、「{remaining_issue_msg}」という点がまだ気になります。")
                st.rerun()
            else:
                st.error("応答分析処理中に予期せぬエラーが発生しました。")
                state_manager.add_message("system", "エラー: 応答分析中に予期せぬエラー。")
        # elif current_step == state_manager.STEP_FOLLOW_UP_LOOP:
            # ...

# --- デバッグ情報 (変更なし) ---
with st.sidebar:
    st.header("デバッグ情報 (Session State)")
    st.json(st.session_state)