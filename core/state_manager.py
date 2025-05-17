# core/state_manager.py
import streamlit as st
from typing import List, Dict, Any, Optional, Literal
from .type_definitions import ChatMessage, UploadedFileData, InitialAnalysisResult, ClarificationAnalysisResult

# ステップの定義 (より厳密にするためにEnumを使うことも推奨されます)
# 各ステップはアプリケーションのUIとロジックのフローを示す
STEP_INPUT_SUBMISSION = "INPUT_SUBMISSION"       # ユーザーからの質問入力待ち
STEP_INITIAL_ANALYSIS = "INITIAL_ANALYSIS"       # 入力された質問の初期分析処理中
STEP_CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED" # 初期分析の結果、質問が曖昧で明確化が必要な状態
STEP_CLARIFICATION_LOOP = "CLARIFICATION_LOOP"     # (現在は未使用)明確化の質問と応答のループ専用ステップとして定義していたが、
                                                 # STEP_CLARIFICATION_NEEDED とチャット入力でループを表現している
STEP_SELECT_STYLE = "SELECT_STYLE"               # 解説スタイルの選択待ち
STEP_GENERATE_EXPLANATION = "GENERATE_EXPLANATION" # 選択されたスタイルに基づき解説を生成中
STEP_FOLLOW_UP_LOOP = "FOLLOW_UP_LOOP"             # 生成された解説に対するフォローアップ質問の受付中
STEP_CONFIRM_UNDERSTANDING = "CONFIRM_UNDERSTANDING" # ユーザーの理解確認後、要約へ移行する中間ステップ
STEP_SUMMARIZE = "SUMMARIZE"                       # セッションの要約を生成中
STEP_SESSION_END = "SESSION_END"                   # 要約表示後、新しいセッションの開始待ち

# セッション状態のデフォルト値
# Streamlitのセッション状態で管理されるキーとその初期値を定義
DEFAULT_SESSION_VALUES: Dict[str, Any] = {
    "current_step": STEP_INPUT_SUBMISSION,              # 現在のアプリケーションのステップ
    "messages": [],                                     # メインの会話履歴 (ユーザーとAIの発言)
    "user_query_text": "",                             # ユーザーが入力したテキスト質問
    "uploaded_file_data": None,                         # アップロード画像はDict[str, Any] (mime_type, data)またはNoneを想定
    "selected_topic": "",                              # ユーザーが選択したトピック
    "initial_analysis_result": None,                    # LLMによる初期分析結果 (曖昧さ、カテゴリなど)
    "is_request_ambiguous": False,                      # 初期分析の結果、リクエストが曖昧かどうかのフラグ
    "clarification_history": [],                        # 明確化ループ専用の会話履歴。メインの `messages` とは別に管理し、
                                                        # 明確化が解決した際にクリアされる。これにより、AIが明確化質問を
                                                        # 正確に参照し、ループが終了した後は影響を残さないようにする。
    "clarified_request_text": None,                     # 明確化後のユーザーリクエストテキスト
    "selected_explanation_style": "detailed",           # ユーザーが選択した解説スタイル
    "current_explanation": None,                        # 生成された解説テキスト
    "current_followup_response": None,                  # 生成されたフォローアップ応答テキスト
    "session_summary": None,                            # 生成されたセッション要約テキスト
    "error_message": None,                              # UIに表示するための一時的なエラーメッセージ
    "processing": False,                                # LLM呼び出しなど時間のかかる処理が実行中かを示すフラグ
    "student_performance_analysis": None,               # 生徒のパフォーマンス分析結果
    "clarification_attempts": 0,                        # 明確化の試行回数 (0から開始)
}

def initialize_session_state():
    """セッション状態のキーを初期化する。
    デフォルト値が未設定のキーのみ初期値を設定する。
    uploaded_file_dataはDict[str, Any] (mime_type, data)またはNoneを想定。
    """
    for key, value in DEFAULT_SESSION_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_current_step() -> str:
    """現在のアプリケーションステップを取得する。"""
    return st.session_state.get("current_step", STEP_INPUT_SUBMISSION)

def set_current_step(step: str):
    """現在のアプリケーションステップを設定する。"""
    st.session_state.current_step = step

def add_message(role: Literal["user", "assistant", "system"], content: str):
    """メインの会話履歴 (`st.session_state.messages`) にメッセージを追加する。"""
    new_message: ChatMessage = {"role": role, "content": content}
    st.session_state.messages.append(new_message)

def store_user_input(query_text: str, uploaded_file: Optional[Dict[str, Any]], topic: str):
    """ユーザーの初期入力を保存する (uploaded_fileは処理済みの画像情報辞書を期待)"""
    st.session_state.user_query_text = query_text
    st.session_state.uploaded_file_data = uploaded_file
    st.session_state.selected_topic = topic

def store_initial_analysis_result(result: InitialAnalysisResult):
    """LLMによる初期分析の結果をセッション状態に保存し、曖昧性フラグも更新する。"""
    st.session_state.initial_analysis_result = result
    st.session_state.is_request_ambiguous = (result.get("ambiguity") == "ambiguous")

def set_processing_status(is_processing: bool):
    """LLM呼び出しなどの処理中フラグをセッション状態に設定する。"""
    st.session_state.processing = is_processing

def reset_for_new_session():
    """新しいセッションのために、セッション状態の主要な値をデフォルト値にリセットする。"""
    for key, value in DEFAULT_SESSION_VALUES.items():
        st.session_state[key] = value
    # 特定のキーをデフォルト値とは異なる値で明示的にリセットする必要がある場合はここで行う
    st.session_state.processing = False
    st.session_state.student_performance_analysis = None
    # clarification_attempts は DEFAULT_SESSION_VALUES に含まれるため、ループでリセットされる

def store_clarification_analysis(analysis_data: ClarificationAnalysisResult):
    """ユーザーの明確化応答に対するLLMの分析結果をセッション状態に保存する。
    曖昧さが解消された場合は、関連フラグを更新し、明確化履歴もクリアする。
    """
    if "resolved" in analysis_data and analysis_data.get("resolved"):
        st.session_state.is_request_ambiguous = False
        st.session_state.clarified_request_text = analysis_data.get("clarified_request", "不明（明確化成功）")
        # 初期分析結果の曖昧さ理由もクリア（あるいは更新）することが望ましい場合がある
        if isinstance(st.session_state.initial_analysis_result, dict):
            st.session_state.initial_analysis_result["reason_for_ambiguity"] = None 
        st.session_state.clarification_history = [] # 明確化解決時に専用履歴をクリア
    else: 
        st.session_state.is_request_ambiguous = True
        # 曖昧さが解消されなかった場合、残っている曖昧さの理由を更新
        if "remaining_issue" in analysis_data and analysis_data.get("remaining_issue"):
            if not isinstance(st.session_state.initial_analysis_result, dict):
                # 通常はinitial_analysis_resultは存在するはずだが、念のため初期化
                st.session_state.initial_analysis_result = {} 
            st.session_state.initial_analysis_result["reason_for_ambiguity"] = analysis_data.get("remaining_issue")
        else: 
            # remaining_issue がない場合は一般的なメッセージで理由を更新
            if not isinstance(st.session_state.initial_analysis_result, dict):
                st.session_state.initial_analysis_result = {}
            st.session_state.initial_analysis_result["reason_for_ambiguity"] = "まだ不明瞭な点があります。もう少し詳しく教えてください。"

def add_clarification_history_message(role: Literal["user", "assistant"], content: str):
    """明確化ループ専用の会話履歴 (`st.session_state.clarification_history`) にメッセージを追加する。
    この履歴は、AIが正確に明確化の文脈を追跡するために使用される。
    """
    if "clarification_history" not in st.session_state:
        st.session_state.clarification_history = []
    new_message: ChatMessage = {"role": role, "content": content}
    st.session_state.clarification_history.append(new_message)

def set_explanation_style(style: str):
    """ユーザーが選択した解説スタイルをセッション状態に保存する。"""
    st.session_state.selected_explanation_style = style

def store_generated_explanation(explanation: str):
    """LLMによって生成された解説テキストをセッション状態に保存し、メインの会話履歴にも追加する。"""
    st.session_state.current_explanation = explanation
    add_message("assistant", explanation)