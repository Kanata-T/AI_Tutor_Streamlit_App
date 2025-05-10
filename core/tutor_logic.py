# core/tutor_logic.py
import streamlit as st # st.session_state にアクセスするために必要
from typing import Dict, Any, Optional
from services import gemini_service # services/__init__.py 経由でインポート
from . import state_manager # 同じディレクトリの state_manager をインポート

def perform_initial_analysis_logic() -> Optional[Dict[str, Any]]:
    """
    ユーザーの入力に基づいて初期分析を実行し、結果を返す。
    1. 画像があればOCRを実行。
    2. OCR結果とテキストクエリで分析を実行。
    """
    query_text = st.session_state.get("user_query_text", "")
    image_data_dict = st.session_state.get("uploaded_file_data", None) # st.session_stateのキー名を確認

    if not query_text and not image_data_dict:
        print("Error in tutor_logic: No query text or image data for analysis.")
        return {"error": "入力がありません。"}

    extracted_ocr_text: Optional[str] = None
    if image_data_dict:
        print(f"Tutor Logic: Performing OCR. Image MIME: {image_data_dict.get('mime_type')}")
        # state_manager.set_processing_status(True) # 必要なら個別のスピナー制御
        # with st.spinner("画像を解析中 (OCR)..."): # app.py側で全体スピナーを出すのでここでは不要かも
        extracted_ocr_text = gemini_service.extract_text_from_image_llm(image_data_dict)
        # state_manager.set_processing_status(False)
        if extracted_ocr_text is None:
            print("Warning in tutor_logic: OCR failed or returned no text.")
            # OCR失敗をエラーとして扱うか、テキストなしとして分析に進むか選択
            # extracted_ocr_text = "" # 空文字として進む
            return {"error": "画像の文字抽出に失敗しました。"} # 今回はエラーとして一旦止める

        print(f"Tutor Logic: OCR Result: '{extracted_ocr_text[:100]}...'")
        # (オプション) OCR結果をst.session_stateに保存してもよい
        # st.session_state.ocr_result_text = extracted_ocr_text
    
    # OCR結果（あれば）とテキストクエリで分析
    print(f"Tutor Logic: Calling analysis with OCR. Query: '{query_text[:50]}...', OCR: '{str(extracted_ocr_text)[:50] if extracted_ocr_text else 'No OCR'}'")
    analysis_result = gemini_service.analyze_initial_input_with_ocr(
        query_text=query_text,
        ocr_text_from_image=extracted_ocr_text
    )
    print(f"Tutor Logic: Received final analysis result: {analysis_result}")

    if analysis_result is None:
        return {"error": "AIによる分析に失敗しました。"}
    if "error" in analysis_result:
        return analysis_result
    
    # 成功した場合、元のinitial_analysis_resultにOCR結果も追加して返す（もし必要なら）
    # 今回のanalyze_query_with_ocrプロンプトではOCR結果は出力JSONに含まれない想定
    # 必要であれば、ここでanalysis_resultに "ocr_text": extracted_ocr_text を追加する
    if extracted_ocr_text and "ocr_text" not in analysis_result: # 出力に含まれないが保持したい場合
         analysis_result["ocr_text_from_extraction"] = extracted_ocr_text


    return analysis_result


def analyze_user_clarification_logic(user_response: str) -> Optional[Dict[str, Any]]:
    """
    ユーザーの明確化応答を分析し、結果 (resolved, clarified_request, remaining_issue) を返す。
    st.session_state から必要な情報を取得する。
    """
    original_query = st.session_state.get("user_query_text", "")
    initial_analysis = st.session_state.get("initial_analysis_result", {}) # これが必要
    reason_ambiguity_initial = initial_analysis.get("reason_for_ambiguity", "詳細不明") # 初期分析時の理由
    image_provided = st.session_state.get("uploaded_file_data") is not None
    
    # AIからの直近の明確化質問を取得
    llm_question = "不明な質問"
    current_messages = st.session_state.get("messages", []) # 全会話履歴
    if current_messages:
        # ユーザーの応答(user_response)の直前のAIの発言が明確化質問であると仮定
        # ユーザーの応答はmessagesの最後に追加されているはずなので、その一つ前を見る
        if len(current_messages) >= 2 and current_messages[-2]["role"] == "assistant":
             llm_question = current_messages[-2]["content"]
        elif current_messages[-1]["role"] == "assistant": # ユーザー入力直前にAI発言がなかった場合(ありえないはずだが念のため)
             llm_question = current_messages[-1]["content"]


    if not all([original_query, reason_ambiguity_initial, llm_question != "不明な質問", user_response]):
        missing_parts = []
        if not original_query: missing_parts.append("元の質問")
        if not reason_ambiguity_initial: missing_parts.append("初期の曖昧さの理由")
        if llm_question == "不明な質問": missing_parts.append("AIの明確化質問")
        if not user_response: missing_parts.append("ユーザーの応答")
        print(f"Error in tutor_logic: Missing data for analyzing user clarification. Missing: {', '.join(missing_parts)}")
        return {"error": f"ユーザー応答の分析に必要な情報が不足しています ({', '.join(missing_parts)})。"}

    print(f"Tutor Logic: Analyzing user clarification. User response: '{user_response[:50]}...'")
    analysis_result = gemini_service.analyze_user_clarification_llm(
        original_query_text=original_query,
        original_image_provided=image_provided,
        reason_for_ambiguity=reason_ambiguity_initial, # 初期分析時の曖昧さの理由
        llm_clarification_question=llm_question,
        user_response_text=user_response,
        conversation_history=current_messages # 全会話履歴を渡す
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
    
    # 明確化質問生成のコンテキストとして渡す会話履歴
    # (ユーザーの最初の質問と、AIの初期応答(曖昧さ指摘)までが含まれている想定)
    current_messages = st.session_state.get("messages", [])


    if not original_query: # 曖昧さの理由がなくても、元のクエリがあれば質問は作れる
        print("Error in tutor_logic: Missing original query for clarification.")
        return "明確化のための元の質問情報がありません。"
    if not reason_ambiguity and current_messages: # 理由が不明でも履歴があれば試みる
        reason_ambiguity = "（具体的な曖昧理由は不明ですが、追加情報が必要です）"


    print(f"Tutor Logic: Generating clarification question. Reason: {reason_ambiguity}")
    clarification_q = gemini_service.generate_clarification_question_llm(
        original_query_text=original_query,
        reason_for_ambiguity=reason_ambiguity, # 必須
        image_provided=image_provided,
        conversation_history=current_messages # ★ 会話履歴を渡す ★
    )
    print(f"Tutor Logic: Generated clarification question: {clarification_q}")
    
    return clarification_q


def generate_explanation_logic() -> Optional[str]:
    """
    保存されたリクエスト、カテゴリ、スタイルに基づき、LLMに解説を生成させる。
    st.session_state から必要な情報を取得する。
    """
    clarified_request = st.session_state.get("clarified_request_text")
    if not clarified_request and st.session_state.get("initial_analysis_result"):
        clarified_request = st.session_state.initial_analysis_result.get("summary", st.session_state.user_query_text)
    elif not clarified_request:
         clarified_request = st.session_state.get("user_query_text", "不明なリクエスト")

    request_category = "不明"
    if st.session_state.get("initial_analysis_result"):
        request_category = st.session_state.initial_analysis_result.get("request_category", "不明")
    
    explanation_style = st.session_state.get("selected_explanation_style", "detailed")
    
    relevant_context_ocr = None
    if st.session_state.get("initial_analysis_result"):
        relevant_context_ocr = st.session_state.initial_analysis_result.get("ocr_text_from_extraction") # 前回の修正でキー名変更
        if not relevant_context_ocr: # フォールバック
            relevant_context_ocr = st.session_state.initial_analysis_result.get("ocr_text")

    # 解説生成のコンテキストとして渡す会話履歴
    # プロンプトの指示通り、スタイル選択までのやり取りを含める
    # (ユーザーがスタイルを選択したメッセージも含む)
    conversation_history_for_llm = st.session_state.get("messages", [])

    if not clarified_request or clarified_request == "不明なリクエスト":
        print("Error in tutor_logic: Clarified request is missing for explanation generation.")
        return "解説を生成するためのリクエスト内容が確定していません。"

    print(f"Tutor Logic: Generating explanation. Request: '{clarified_request[:50]}...', Style: {explanation_style}")
    explanation_text = gemini_service.generate_explanation_llm(
        clarified_request=clarified_request,
        request_category=request_category,
        explanation_style=explanation_style,
        relevant_context=relevant_context_ocr,
        conversation_history=conversation_history_for_llm
    )
    print(f"Tutor Logic: Generated explanation (first 100 chars): {explanation_text[:100] if explanation_text else 'None'}")
    return explanation_text

def generate_followup_response_logic(user_latest_input: str) -> Optional[str]:
    """
    ユーザーのフォローアップ入力に基づき、LLMに適切な応答を生成させる。
    """
    full_conversation_history = st.session_state.get("messages", [])
    
    if not full_conversation_history:
        print("Error in tutor_logic: Conversation history is empty for followup.")
        return "会話の文脈がありません。"

    # ユーザーの最新入力は full_conversation_history の最後にも含まれているはず
    # なので、user_latest_input はそのまま渡す
    print(f"Tutor Logic: Generating followup response. User input: '{user_latest_input[:50]}...'")
    followup_response = gemini_service.generate_followup_response_llm(
        conversation_history=full_conversation_history, # ユーザーの最新入力を含む全履歴
        user_latest_input=user_latest_input # プロンプト用に別途渡す
    )
    print(f"Tutor Logic: Generated followup response (first 100 chars): {followup_response[:100] if followup_response else 'None'}")
    return followup_response

def generate_summary_logic() -> Optional[str]:
    """
    現在の会話履歴全体とセッション情報を基に、セッションの要約をLLMに生成させる。
    """
    full_conversation_history = st.session_state.get("messages", [])
    
    request_category = "不明なカテゴリ"
    topic = "不明なトピック"
    if st.session_state.get("initial_analysis_result"):
        request_category = st.session_state.initial_analysis_result.get("request_category", request_category)
        topic = st.session_state.initial_analysis_result.get("topic", topic)
    
    # もしclarified_request_textがあれば、それをtopicとして優先するなども考えられる
    if st.session_state.get("clarified_request_text"):
        topic = st.session_state.clarified_request_text # こちらの方がより具体的な場合がある


    if not full_conversation_history:
        print("Error in tutor_logic: Conversation history is empty for summary.")
        return "要約するための会話履歴がありません。"

    print(f"Tutor Logic: Generating session summary. Category: {request_category}, Topic: {topic[:50]}...")
    summary_text = gemini_service.generate_summary_llm(
        request_category=request_category,
        topic=topic,
        conversation_history=full_conversation_history
    )
    print(f"Tutor Logic: Generated summary (first 100 chars): {summary_text[:100] if summary_text else 'None'}")
    return summary_text