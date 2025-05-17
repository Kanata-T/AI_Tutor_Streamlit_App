# core/tutor_logic.py
import streamlit as st # st.session_state にアクセスするために必要
from typing import Dict, Any, Optional, List
from services import gemini_service # services/__init__.py 経由でインポート
from . import state_manager # 同じディレクトリの state_manager をインポート
from .type_definitions import ChatMessage, UploadedFileData, InitialAnalysisResult, ClarificationAnalysisResult # 型定義をインポート

def perform_initial_analysis_logic() -> Optional[InitialAnalysisResult]:
    """
    ユーザーの初期入力 (テキストクエリ、画像) に基づいて初期分析を実行する。

    処理フロー:
    1. セッション状態からユーザーの質問テキストとアップロードされた画像データを取得。
    2. 入力がない場合はエラーを返す。
    3. 画像データが存在すれば、`gemini_service.extract_text_from_image_llm` を呼び出してOCRを実行。
       - OCR失敗時はエラーを返す。
    4. ユーザーの質問テキストとOCR結果 (あれば) を用いて、
       `gemini_service.analyze_initial_input_with_ocr` を呼び出し、質問の分析を依頼。
       - 分析結果には、曖昧さ、カテゴリ、トピック、要約などが含まれることを期待。
    5. 分析結果にOCRテキストを (もしあれば) `ocr_text_from_extraction`として追加し、結果を返す。
       - LLMからの直接の分析結果にOCRテキストが含まれない場合に備え、別途追加する。

    Returns:
        Optional[InitialAnalysisResult]: 分析結果。エラー時は "error" キーを含む辞書。
    """
    query_text: str = st.session_state.get("user_query_text", "")
    image_data_dict = st.session_state.get("uploaded_file_data", None) # これは {"mime_type": ..., "data": ...} 形式になっているはず

    if not query_text and not image_data_dict:
        print("Error in tutor_logic: No query text or image data for analysis.")
        return {"error": "質問内容のテキスト入力または画像のアップロードのいずれかを行ってください。"} # type: ignore

    extracted_ocr_text: Optional[str] = None
    if image_data_dict: # この image_data_dict をそのまま gemini_service.extract_text_from_image_llm に渡す
        print(f"Tutor Logic: Performing OCR. Image MIME: {image_data_dict.get('mime_type')}")
        extracted_ocr_text = gemini_service.extract_text_from_image_llm(image_data_dict) # 引数形式は合致
        if extracted_ocr_text is None:
            print("Warning in tutor_logic: OCR failed or returned no text.")
            return {"error": "アップロードされた画像から文字を抽出できませんでした。画像をご確認いただくか、テキストで質問を入力してください。"} # type: ignore
        print(f"Tutor Logic: OCR Result (first 100 chars): '{extracted_ocr_text[:100]}...'")
    
    print(f"Tutor Logic: Calling analysis with OCR. Query: '{query_text[:50]}...', OCR: '{str(extracted_ocr_text)[:50] if extracted_ocr_text else 'No OCR'}'")
    analysis_result: Optional[InitialAnalysisResult] = gemini_service.analyze_initial_input_with_ocr(
        query_text=query_text,
        ocr_text_from_image=extracted_ocr_text
    )
    print(f"Tutor Logic: Received final analysis result from LLM: {analysis_result}")

    if analysis_result is None:
        # gemini_service内でAPI呼び出し自体に失敗した場合など
        return {"error": "AIによる質問の分析処理中に予期せぬエラーが発生しました。"} # type: ignore
    if "error" in analysis_result:
        # gemini_service内のプロンプト読み込みやフォーマットエラー、JSONパースエラーなど
        return analysis_result # エラー情報をそのまま上位に伝搬
    
    # LLMの分析結果に、こちらで抽出したOCRテキスト情報を追加する
    # これにより、後続の処理でOCR結果を別途参照する手間を省き、分析結果オブジェクトに集約する
    if extracted_ocr_text:
         analysis_result["ocr_text_from_extraction"] = extracted_ocr_text # type: ignore

    return analysis_result


def analyze_user_clarification_logic(user_response: str) -> Optional[ClarificationAnalysisResult]:
    """
    ユーザーの明確化応答を分析し、曖昧さが解消されたかを判定する。

    必要なセッション状態:
    - `user_query_text`: 元のユーザーの質問テキスト。
    - `initial_analysis_result`: 初期分析の結果 (特に `reason_for_ambiguity`)。
    - `uploaded_file_data`: 元の質問で画像が提供されたかの情報。
    - `clarification_history`: AIの明確化質問とそれに対するユーザー応答の履歴。
                         この履歴の最後から2番目のアシスタントの発言をAIの質問として使用する。
    - `messages`: メインの会話履歴 (LLMへのコンテキストとして渡すため)。

    Args:
        user_response (str): ユーザーによる明確化の応答テキスト。

    Returns:
        Optional[ClarificationAnalysisResult]: 分析結果。エラー時は "error" キーを含む辞書。
    """
    original_query: str = st.session_state.get("user_query_text", "")
    initial_analysis: Optional[InitialAnalysisResult] = st.session_state.get("initial_analysis_result", None)
    
    if not initial_analysis:
        # この状況は通常発生しないはずだが、防御的にチェック
        print("Error in tutor_logic: Initial analysis result not found for clarification.")
        return {"error": "ユーザー応答を分析するための元の質問情報が見つかりません。"} # type: ignore
        
    reason_ambiguity_initial: str = initial_analysis.get("reason_for_ambiguity", "（詳細不明な曖昧さ）")
    image_provided: bool = st.session_state.get("uploaded_file_data") is not None
    
    llm_question = "不明な質問" # デフォルト値
    # 明確化ループ専用の履歴 `clarification_history` からAIの最後の質問を取得する。
    # `app.py` のロジックにより、この関数が呼ばれる時点で `clarification_history` には
    # [..., AIの質問, ユーザーの最新応答] の順でメッセージが格納されているはず。
    clarification_hist: List[ChatMessage] = st.session_state.get("clarification_history", [])
    if len(clarification_hist) >= 2 and clarification_hist[-2]["role"] == "assistant":
        llm_question = clarification_hist[-2]["content"]
    elif len(clarification_hist) == 1 and clarification_hist[0]["role"] == "assistant":
        # ユーザー応答がclarification_historyに追加される「前」に呼ばれた場合のフォールバック(通常はない)
        llm_question = clarification_hist[0]["content"]
    else:
        # AIの質問が見つからない場合はエラーログを残し、デフォルトの「不明な質問」を使用
        print(f"Warning in tutor_logic: Could not retrieve AI's clarification question from clarification_history. History: {clarification_hist}")

    # LLMに渡すメインの会話履歴 (デバッグやより広範な文脈理解のため)
    main_conversation_history: List[ChatMessage] = st.session_state.get("messages", [])

    if not all([original_query, user_response]): # llm_question は見つからなくても許容する（プロンプト側で対応想定）
        missing_parts = []
        if not original_query: missing_parts.append("元の質問")
        # if llm_question == "不明な質問": missing_parts.append("AIの明確化質問") # なくても進める
        if not user_response: missing_parts.append("ユーザーの応答")
        if missing_parts: # reason_ambiguity_initial はなくても許容
            print(f"Error in tutor_logic: Missing critical data for analyzing user clarification. Missing: {', '.join(missing_parts)}")
            return {"error": f"ユーザー応答の分析に必要な情報 ({', '.join(missing_parts)}) が不足しています。"} # type: ignore

    print(f"Tutor Logic: Analyzing user clarification. User response: '{user_response[:50]}...' for AI question: '{llm_question[:50]}...'")
    analysis_result: Optional[ClarificationAnalysisResult] = gemini_service.analyze_user_clarification_llm(
        original_query_text=original_query,
        original_image_provided=image_provided,
        reason_for_ambiguity=reason_ambiguity_initial,
        llm_clarification_question=llm_question,
        user_response_text=user_response,
        conversation_history=main_conversation_history 
    )
    print(f"Tutor Logic: Received clarification analysis result: {analysis_result}")
    st.session_state.debug_last_clarification_analysis = analysis_result # デバッグ用に保存
    return analysis_result


def generate_clarification_question_logic() -> Optional[str]:
    """
    ユーザーの初期質問が曖昧だった場合に、LLMに明確化のための質問を生成させる。

    必要なセッション状態:
    - `user_query_text`: 元のユーザーの質問テキスト。
    - `initial_analysis_result`: 初期分析の結果 (特に `reason_for_ambiguity`)。
    - `uploaded_file_data`: 元の質問で画像が提供されたかの情報。
    - `messages`: メインの会話履歴 (LLMへのコンテキストとして渡すため)。

    Returns:
        Optional[str]: 生成された明確化質問の文字列。エラー時はエラーメッセージ文字列。
    """
    original_query: str = st.session_state.get("user_query_text", "")
    initial_analysis: Optional[InitialAnalysisResult] = st.session_state.get("initial_analysis_result", None)
    
    if not initial_analysis:
        print("Error in tutor_logic: Initial analysis result not found for generating clarification question.")
        return "明確化質問を生成するための元の質問の分析情報が見つかりません。"
        
    reason_ambiguity: str = initial_analysis.get("reason_for_ambiguity", "（具体的な曖昧性の理由は分析されていませんが、追加情報が必要です）")
    image_provided: bool = st.session_state.get("uploaded_file_data") is not None
    main_conversation_history: List[ChatMessage] = st.session_state.get("messages", [])

    if not original_query:
        print("Error in tutor_logic: Missing original query for clarification question generation.")
        return "明確化のための元の質問情報が不足しています。"

    print(f"Tutor Logic: Generating clarification question. Original query: '{original_query[:50]}...' Reason: {reason_ambiguity}")
    clarification_q = gemini_service.generate_clarification_question_llm(
        original_query_text=original_query,
        reason_for_ambiguity=reason_ambiguity,
        image_provided=image_provided,
        conversation_history=main_conversation_history
    )
    print(f"Tutor Logic: Generated clarification question: '{str(clarification_q)[:100]}...'")
    return clarification_q


def generate_explanation_logic() -> Optional[str]:
    """ユーザーのリクエストと選択されたスタイルに基づき、LLMに解説文を生成させる。"""
    clarified_request = st.session_state.get("clarified_request_text")
    if not clarified_request and st.session_state.get("initial_analysis_result"):
        initial_res = st.session_state.initial_analysis_result
        clarified_request = initial_res.get("summary", st.session_state.user_query_text)
    elif not clarified_request:
         clarified_request = st.session_state.get("user_query_text", "不明なリクエスト")

    request_category = "不明"
    if st.session_state.get("initial_analysis_result"):
        request_category = st.session_state.initial_analysis_result.get("request_category", "不明")
    
    explanation_style = st.session_state.get("selected_explanation_style", "detailed")
    
    relevant_context_ocr = None
    if st.session_state.get("initial_analysis_result"):
        initial_res_for_ocr = st.session_state.initial_analysis_result
        # ocr_text_from_extraction は perform_initial_analysis_logic で追加されるカスタムキー
        relevant_context_ocr = initial_res_for_ocr.get("ocr_text_from_extraction") 
        if not relevant_context_ocr:
            relevant_context_ocr = initial_res_for_ocr.get("ocr_text") # プロンプトからの直接的なocr_textキーも考慮

    conversation_history_for_llm: List[ChatMessage] = st.session_state.get("messages", [])

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
    print(f"Tutor Logic: Generated explanation (first 100 chars): {str(explanation_text)[:100] if explanation_text else 'None'}")
    return explanation_text

def generate_followup_response_logic(user_latest_input: str) -> Optional[str]:
    """ユーザーのフォローアップ入力に対し、LLMに応答を生成させる。"""
    full_conversation_history: List[ChatMessage] = st.session_state.get("messages", [])
    
    if not full_conversation_history:
        print("Error in tutor_logic: Conversation history is empty for followup.")
        return "AIと応答するための会話の文脈がありません。"

    print(f"Tutor Logic: Generating followup response. User input: '{user_latest_input[:50]}...'")
    followup_response = gemini_service.generate_followup_response_llm(
        conversation_history=full_conversation_history,
        user_latest_input=user_latest_input
    )
    print(f"Tutor Logic: Generated followup response (first 100 chars): {str(followup_response)[:100] if followup_response else 'None'}")
    return followup_response

def generate_summary_logic() -> Optional[str]:
    """現在のセッションの会話履歴に基づき、LLMに要約を生成させる。"""
    full_conversation_history: List[ChatMessage] = st.session_state.get("messages", [])
    request_category = "不明なカテゴリ"
    topic = "不明なトピック"

    initial_res_summary: Optional[InitialAnalysisResult] = st.session_state.get("initial_analysis_result")
    if initial_res_summary:
        request_category = initial_res_summary.get("request_category", request_category)
        topic = initial_res_summary.get("topic", topic)
    
    clarified_text_topic = st.session_state.get("clarified_request_text")
    if clarified_text_topic:
        topic = clarified_text_topic # 明確化されたテキストがあれば、それをトピックとして優先

    if not full_conversation_history:
        print("Error in tutor_logic: Conversation history is empty for summary.")
        return "要約を生成するための会話履歴がありません。"

    print(f"Tutor Logic: Generating session summary. Category: {request_category}, Topic: {topic[:50]}...")
    summary_text = gemini_service.generate_summary_llm(
        request_category=request_category,
        topic=topic,
        conversation_history=full_conversation_history
    )
    print(f"Tutor Logic: Generated summary (first 100 chars): {str(summary_text)[:100] if summary_text else 'None'}")
    return summary_text

def format_conversation_for_analysis(conversation_history: List[ChatMessage]) -> str:
    """提供された会話履歴を、生徒のパフォーマンス分析プロンプトに適した文字列形式に整形する。
    システムエラーメッセージは除外する。
    """
    performance_data_str = "セッション会話履歴:\n"
    if not conversation_history:
        return "セッション会話履歴: (履歴なし)\n"
        
    for message in conversation_history:
        if message["role"] == "system" and "エラー" in message["content"]:
            continue
        role_display = "生徒" if message["role"] == "user" else "AIチューター"
        # contentがNoneの場合も考慮 (TypedDictではcontentは必須だが念のため)
        content_text = message.get("content", "") 
        performance_data_str += f"- {role_display}: {content_text}\n"
    
    # (オプション) ここに `initial_analysis_result` の内容などを追加することも可能
    # 例: initial_res_format = st.session_state.get("initial_analysis_result")
    # if initial_res_format:
    #     try:
    #         performance_data_str += "\n初期分析結果:\n"
    #         performance_data_str += json.dumps(dict(initial_res_format), indent=2, ensure_ascii=False) + "\n"
    #     except TypeError as e:
    #         performance_data_str += f"初期分析結果の表示に失敗: {e}\n"

    return performance_data_str


def analyze_student_performance_logic() -> Optional[str]:
    """現在のセッションの会話履歴を基に、生徒のパフォーマンス分析レポートをLLMに生成させる。"""
    full_conversation_history: List[ChatMessage] = st.session_state.get("messages", [])
    
    if not full_conversation_history:
        print("Error in tutor_logic: Conversation history is empty for performance analysis.")
        return "パフォーマンス分析の対象となる会話履歴がありません。"

    performance_data_str = format_conversation_for_analysis(full_conversation_history)

    print("Tutor Logic: Analyzing student performance.")
    analysis_report = gemini_service.analyze_student_performance_llm(
        student_performance_data_str=performance_data_str
    )
    # print(f"Tutor Logic: Generated student performance analysis report: {analysis_report}")
    return analysis_report