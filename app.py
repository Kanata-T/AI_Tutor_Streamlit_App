# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image # Pillowをインポート

# coreモジュールのインポート
from core import state_manager # state_managerをインポート

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

# セッション状態の初期化 (state_managerの関数を呼び出す)
state_manager.initialize_session_state()

# --- メイン処理 ---
current_step = state_manager.get_current_step() # 現在のステップを取得

if current_step == state_manager.STEP_INPUT_SUBMISSION:
    st.header("1. 質問を入力してください")
    with st.form("submission_form"):
        user_query_text_input = st.text_input(
            "質問内容を入力してください",
            value=st.session_state.user_query_text # 前回の値を表示する場合
        )
        uploaded_file_obj = st.file_uploader( # 変数名を変更 (streamlitのオブジェクト)
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
                        # Pillowを使って画像を開く必要はない (getvalueでバイト列取得可能)
                        # img = Image.open(uploaded_file_obj) # これは表示や加工するなら必要
                        image_data_for_state = {
                            "mime_type": uploaded_file_obj.type,
                            "data": uploaded_file_obj.getvalue() # バイトデータを取得
                        }
                        st.success("画像を読み込みました。")
                    except Exception as e:
                        st.error(f"画像の読み込みに失敗しました: {e}")
                        image_data_for_state = None
                
                # state_manager経由でユーザー入力を保存
                state_manager.store_user_input(
                    user_query_text_input,
                    image_data_for_state,
                    selected_topic_input
                )
                
                state_manager.add_message("user", f"質問: {user_query_text_input}" + (f" (画像あり)" if image_data_for_state else "")) # ユーザーの質問を履歴に追加
                state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS)
                st.rerun() # 状態変更を即時UIに反映させる

elif current_step == state_manager.STEP_INITIAL_ANALYSIS:
    st.header("2. 初期分析中です...")
    # (デバッグ用) 保存されたユーザー情報を表示
    st.write("ユーザーの質問 (セッションステートより):")
    st.write(f"- テキスト: {st.session_state.user_query_text}")
    st.write(f"- トピック: {st.session_state.selected_topic}")
    if st.session_state.uploaded_file_data:
        st.write("- アップロードされた画像:")
        st.image(st.session_state.uploaded_file_data["data"])
    
    # ここで実際にGemini APIを呼び出す処理を後でtutor_logic.pyに実装し、ここから呼び出す
    # with st.spinner("AIが分析中です..."):
    #     analysis_result = tutor_logic.perform_initial_analysis() # 仮の呼び出し
    #     state_manager.store_initial_analysis_result(analysis_result)
    #     if state_manager.is_request_ambiguous():
    #         state_manager.set_current_step(state_manager.STEP_CLARIFICATION_NEEDED)
    #     else:
    #         state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
    # st.rerun()
    st.info("（現在、初期分析ロジックは未実装です。次のステップで実装します。）")


# 会話履歴の表示 (常に表示する部分)
if st.session_state.messages: # messagesが空でない場合のみ表示
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