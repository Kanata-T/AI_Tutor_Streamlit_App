# core/state_manager.py
import streamlit as st
from typing import List, Dict, Any, Optional, Literal

# ステップの定義 (より厳密にするためにEnumを使うことも推奨されます)
STEP_INPUT_SUBMISSION = "INPUT_SUBMISSION"
STEP_INITIAL_ANALYSIS = "INITIAL_ANALYSIS"
STEP_CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED" # 曖昧さあり、明確化ループ開始前
STEP_CLARIFICATION_LOOP = "CLARIFICATION_LOOP"     # 明確化の質問と応答のループ中
STEP_SELECT_STYLE = "SELECT_STYLE"
STEP_GENERATE_EXPLANATION = "GENERATE_EXPLANATION"
STEP_FOLLOW_UP_LOOP = "FOLLOW_UP_LOOP"
STEP_CONFIRM_UNDERSTANDING = "CONFIRM_UNDERSTANDING"
STEP_SUMMARIZE = "SUMMARIZE"
STEP_SESSION_END = "SESSION_END" # 要約表示後、新規質問待ち

def initialize_session_state():
    """セッション状態のキーを初期化する"""
    default_values = {
        "current_step": STEP_INPUT_SUBMISSION,
        "messages": [],  # メインのチャット履歴: List[Dict[str, str]] (role, content)
        "user_query_text": "",
        "uploaded_file_data": None, # Dict[str, Any] (mime_type, data) or None
        "selected_topic": "",
        "initial_analysis_result": None, # Dict[str, Any] LLMからの初期分析結果
        "is_request_ambiguous": False,
        "clarification_history": [], # 明確化ループ中の会話履歴
        "clarified_request_text": None, # 明確化後のリクエスト
        "selected_explanation_style": "detailed", # デフォルトスタイル
        "current_explanation": None,
        "current_followup_response": None,
        "session_summary": None,
        "error_message": None,
        "processing": False, # LLM処理中などのフラグ
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_current_step() -> str:
    """現在のステップを取得する"""
    return st.session_state.get("current_step", STEP_INPUT_SUBMISSION)

def set_current_step(step: str):
    """現在のステップを設定する"""
    st.session_state.current_step = step

def add_message(role: Literal["user", "assistant", "system"], content: str):
    """メインの会話履歴にメッセージを追加する"""
    st.session_state.messages.append({"role": role, "content": content})

def store_user_input(query_text: str, uploaded_file: Optional[Dict[str, Any]], topic: str):
    """ユーザーの初期入力を保存する"""
    st.session_state.user_query_text = query_text
    st.session_state.uploaded_file_data = uploaded_file
    st.session_state.selected_topic = topic

def store_initial_analysis_result(result: Dict[str, Any]):
    """初期分析の結果を保存する"""
    st.session_state.initial_analysis_result = result
    st.session_state.is_request_ambiguous = (result.get("ambiguity") == "ambiguous")

def set_processing_status(is_processing: bool):
    """処理中ステータスを設定する"""
    st.session_state.processing = is_processing

def reset_for_new_session():
    """新しいセッションのために主要な状態をリセットする"""
    keys_to_clear_or_reset = [
        "messages", "user_query_text", "uploaded_file_data", "selected_topic",
        "initial_analysis_result", "is_request_ambiguous", "clarification_history",
        "clarified_request_text", "current_explanation", "current_followup_response",
        "session_summary", "error_message"
    ]
    for key in keys_to_clear_or_reset:
        if key in st.session_state:
            # 型に応じて適切な初期値にリセットする
            if isinstance(st.session_state[key], list):
                st.session_state[key] = []
            elif isinstance(st.session_state[key], dict):
                 st.session_state[key] = {} # またはNoneなど、キーによる
            elif isinstance(st.session_state[key], bool):
                st.session_state[key] = False # キーによる
            else:
                st.session_state[key] = None # または空文字列など
    # current_stepとデフォルトのスタイルはリセット後に再設定
    st.session_state.current_step = STEP_INPUT_SUBMISSION
    st.session_state.selected_explanation_style = "detailed"
    st.session_state.processing = False
    # initialize_session_state() を再度呼び出して、すべてのキーがデフォルト値を持つことを保証する
    initialize_session_state()


# --- 今後追加する可能性のある関数 ---
# def get_user_query() -> Dict[str, Any]:
#     return {
#         "text": st.session_state.user_query_text,
#         "image_data": st.session_state.uploaded_file_data,
#         "topic": st.session_state.selected_topic
#     }

# def add_clarification_message(role: str, content: str):
#     st.session_state.clarification_history.append({"role": role, "content": content})

# def set_clarified_request(request_text: str):
#    st.session_state.clarified_request_text = request_text
#    st.session_state.is_request_ambiguous = False # 明確化された