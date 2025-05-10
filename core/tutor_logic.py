# core/tutor_logic.py
import streamlit as st # st.session_state にアクセスするために必要
from typing import Dict, Any, Optional
from services import gemini_service # services/__init__.py 経由でインポート
from . import state_manager # 同じディレクトリの state_manager をインポート

def perform_initial_analysis_logic() -> Optional[Dict[str, Any]]:
    """
    ユーザーの入力に基づいて初期分析を実行し、結果を返す。
    st.session_state から必要な情報を取得し、gemini_service を呼び出す。
    """
    query_text = st.session_state.get("user_query_text", "")
    image_data = st.session_state.get("uploaded_file_data", None) # mime_type, data を含む辞書
    # selected_topic = st.session_state.get("selected_topic", "") # LLMに渡す情報として含めても良い

    if not query_text and not image_data:
        # st.error("分析するテキストまたは画像がありません。") # UI側で表示するのでここでは不要かも
        print("Error in tutor_logic: No query text or image data for analysis.")
        return {"error": "入力がありません。"}


    # gemini_service の analyze_initial_input を呼び出す
    # この関数はJSONレスポンス(辞書)またはエラー情報(辞書)またはNoneを返す想定
    print(f"Tutor Logic: Calling Gemini for initial analysis. Query: '{query_text[:50]}...', Image: {'Yes' if image_data else 'No'}")
    analysis_result = gemini_service.analyze_initial_input(
        query_text=query_text,
        image_data=image_data
    )
    print(f"Tutor Logic: Received analysis result: {analysis_result}")


    if analysis_result is None:
        # API呼び出し自体が失敗した場合など
        return {"error": "AIによる分析に失敗しました。時間をおいて再度お試しください。"}
    
    if "error" in analysis_result:
        # APIは成功したが、レスポンスのパースエラーなど、サービス層でエラーがセットされた場合
        return analysis_result # エラー情報をそのまま返す

    # 成功した場合、結果を返す (この辞書が state_manager.store_initial_analysis_result に渡される)
    return analysis_result


def analyze_user_clarification_logic(user_response: str) -> Optional[Dict[str, Any]]:
    """
    ユーザーの明確化応答を分析し、結果 (resolved, clarified_request, remaining_issue) を返す。
    st.session_state から必要な情報を取得する。
    """
    original_query = st.session_state.get("user_query_text", "")
    initial_analysis = st.session_state.get("initial_analysis_result", {})
    reason_ambiguity = initial_analysis.get("reason_for_ambiguity", "詳細不明")
    image_provided = st.session_state.get("uploaded_file_data") is not None
    
    # AIからの直近の明確化質問を取得 (messagesの最後がassistantであると仮定)
    # より堅牢にするには、clarification_history を使うか、メッセージにタイプを付与する
    llm_question = "不明な質問" # デフォルト
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant": # 最後のAIの発言を明確化質問とする
                llm_question = msg["content"]
                break
    
    # あるいは、明確化ループ専用の履歴 `clarification_history` があればそこから取得
    # if st.session_state.get("clarification_history"):
    #    last_clarif_q_entry = next((item for item in reversed(st.session_state.clarification_history) if item["role"] == "assistant"), None) # もしclarification_historyを使うなら
    #    if last_clarif_q_entry:
    #        llm_question = last_clarif_q_entry["content"]


    if not all([original_query, reason_ambiguity, llm_question, user_response]):
        print("Error in tutor_logic: Missing data for analyzing user clarification.")
        return {"error": "ユーザー応答の分析に必要な情報が不足しています。"}

    print(f"Tutor Logic: Analyzing user clarification. User response: '{user_response[:50]}...'")
    analysis_result = gemini_service.analyze_user_clarification_llm(
        original_query_text=original_query,
        original_image_provided=image_provided,
        reason_for_ambiguity=reason_ambiguity,
        llm_clarification_question=llm_question,
        user_response_text=user_response
    )
    print(f"Tutor Logic: Received clarification analysis result: {analysis_result}")
    
    return analysis_result

def generate_clarification_question_logic() -> Optional[str]:
    """
    LLMに明確化のための質問を生成させ、その質問文字列を返す。
    st.session_state から必要な情報を取得する。
    """
    original_query = st.session_state.get("user_query_text", "")
    initial_analysis = st.session_state.get("initial_analysis_result", {})
    reason_ambiguity = initial_analysis.get("reason_for_ambiguity", "詳細不明")
    image_provided = st.session_state.get("uploaded_file_data") is not None

    if not original_query or not reason_ambiguity: # reason_ambiguityが空でも質問は作れるかもしれないので、original_queryだけでも良いかも
        print("Error in tutor_logic: Missing original query or ambiguity reason for clarification.")
        # ユーザーに表示するエラーメッセージは呼び出し元(app.py)で制御した方が良い場合もある
        return "明確化のための情報を取得できませんでした。" 

    print(f"Tutor Logic: Generating clarification question. Reason: {reason_ambiguity}")
    clarification_q = gemini_service.generate_clarification_question_llm(
        original_query_text=original_query,
        reason_for_ambiguity=reason_ambiguity,
        image_provided=image_provided
    )
    print(f"Tutor Logic: Generated clarification question: {clarification_q}")
    
    return clarification_q


def generate_explanation_logic() -> Optional[str]:
    """
    保存されたリクエスト、カテゴリ、スタイルに基づき、LLMに解説を生成させる。
    st.session_state から必要な情報を取得する。
    """
    clarified_request = st.session_state.get("clarified_request_text")
    # 曖昧でなかった場合は、初期分析のサマリーを使う
    if not clarified_request and st.session_state.get("initial_analysis_result"):
        initial_summary = st.session_state.initial_analysis_result.get("summary")
        if initial_summary:
            clarified_request = initial_summary
        else: # サマリーもなければ元のクエリ
            clarified_request = st.session_state.get("user_query_text", "不明なリクエスト")
    elif not clarified_request: # それでもなければ
         clarified_request = st.session_state.get("user_query_text", "不明なリクエスト")


    request_category = "不明"
    if st.session_state.get("initial_analysis_result"):
        request_category = st.session_state.initial_analysis_result.get("request_category", "不明")
    
    explanation_style = st.session_state.get("selected_explanation_style", "detailed")
    
    # 関連コンテキスト (OCR結果など)
    relevant_context_ocr = None
    if st.session_state.get("initial_analysis_result"):
        relevant_context_ocr = st.session_state.initial_analysis_result.get("ocr_text")
    
    # 会話履歴の要約 (簡易的に直近数件のメッセージ、または明確化ループの履歴など)
    # 今回は明確化が完了した時点での全メッセージ履歴を渡してみる (トークン数に注意)
    # より高度な場合は、関連性の高い部分だけを抽出・要約する
    conversation_history_for_llm = st.session_state.get("messages", [])
    # トークン数削減のため、システムメッセージは除外しても良いかもしれない
    # conversation_history_for_llm = [msg for msg in conversation_history_for_llm if msg["role"] != "system"]


    if not clarified_request or clarified_request == "不明なリクエスト":
        print("Error in tutor_logic: Clarified request is missing for explanation generation.")
        return "解説を生成するためのリクエスト内容が確定していません。"

    print(f"Tutor Logic: Generating explanation. Request: '{clarified_request[:50]}...', Style: {explanation_style}")
    explanation_text = gemini_service.generate_explanation_llm(
        clarified_request=clarified_request,
        request_category=request_category,
        explanation_style=explanation_style,
        relevant_context=relevant_context_ocr, # OCRテキストをコンテキストとして渡す
        conversation_history=conversation_history_for_llm # これまでの会話履歴
    )
    print(f"Tutor Logic: Generated explanation (first 100 chars): {explanation_text[:100] if explanation_text else 'None'}")
    
    return explanation_text

def generate_followup_response_logic(user_latest_input: str) -> Optional[str]:
    """
    ユーザーのフォローアップ入力に基づき、LLMに適切な応答を生成させる。
    """
    # ユーザーの最新入力は引数で受け取るが、それを含む全会話履歴を渡す
    full_conversation_history = st.session_state.get("messages", [])
    
    if not full_conversation_history: # 履歴がないことは通常ありえないが念のため
        print("Error in tutor_logic: Conversation history is empty for followup.")
        return "会話の文脈がありません。"

    print(f"Tutor Logic: Generating followup response. User input: '{user_latest_input[:50]}...'")
    followup_response = gemini_service.generate_followup_response_llm(
        conversation_history=full_conversation_history, # 全履歴を渡す
        user_latest_input=user_latest_input # 最新の入力も渡す (プロンプト内で利用)
    )
    print(f"Tutor Logic: Generated followup response (first 100 chars): {followup_response[:100] if followup_response else 'None'}")
    
    return followup_response