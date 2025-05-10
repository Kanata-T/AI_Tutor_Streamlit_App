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

# 会話履歴の表示 (常に表示する部分)
if st.session_state.messages:
    st.subheader("会話履歴")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# チャット入力 (フォローアップ用、今はまだ機能しない)
# prompt = st.chat_input("追加の質問や返答があれば入力してください...")
# if prompt:
#    # ... (省略) ...

# デバッグ用にセッションステートを表示
with st.sidebar:
    st.header("デバッグ情報 (Session State)")
    st.json(st.session_state)