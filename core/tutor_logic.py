# core/tutor_logic.py
import streamlit as st # st.session_state にアクセスするために必要
from typing import Dict, Any, Optional, List
from services import gemini_service # services/__init__.py 経由でインポート
from . import state_manager # 同じディレクトリの state_manager をインポート
from .type_definitions import ChatMessage, UploadedFileData, InitialAnalysisResult, ClarificationAnalysisResult, ProcessedImageInfo, ImageType # 型定義をインポート

def perform_initial_analysis_logic() -> Optional[InitialAnalysisResult]:
    """
    ユーザーの初期入力 (テキストクエリ、複数画像) に基づいて初期分析を実行する。
    画像がある場合は、OCRと画像種別判別を行い、その情報を初期分析プロンプトに含める。
    """
    query_text: str = st.session_state.get("user_query_text", "")
    raw_image_data_list: Optional[List[Dict[str, Any]]] = st.session_state.get("uploaded_file_data", None)

    if not query_text and not raw_image_data_list:
        print("Error in tutor_logic: No query text or image data for analysis.")
        return {"error": "質問内容のテキスト入力または画像のアップロードのいずれかを行ってください。"}

    combined_ocr_and_type_info_for_prompt: Optional[str] = None
    st.session_state.processed_image_details_list = [] # 初期化

    if raw_image_data_list and len(raw_image_data_list) > 0:
        print(f"Tutor Logic: Performing OCR and Type Classification for {len(raw_image_data_list)} image(s).")
        processed_image_info_list: List[ProcessedImageInfo] = gemini_service.extract_text_and_type_from_image_llm(raw_image_data_list)
        st.session_state.processed_image_details_list = processed_image_info_list

        if not processed_image_info_list:
            print("Warning in tutor_logic: OCR and Type Classification returned no results.")
            return {"error": "アップロードされた画像から情報を抽出できませんでした。"}

        ocr_texts_for_prompt_parts = []
        has_successful_ocr = False # 初期値は False
        for proc_info in processed_image_info_list:
            filename = proc_info.get("original_filename", "不明なファイル")
            img_type_str = str(proc_info.get("image_type", ImageType.OTHER))
            ocr_text_single = proc_info.get("ocr_text", "[テキスト抽出なし]") # デフォルトを明示

            ocr_text_display = ocr_text_single # 表示用はそのまま

            # 成功したOCRかどうかを判定するフラグ
            current_ocr_is_successful = False
            if ocr_text_single: # まずNoneや空文字列でないこと
                # エラーを示す典型的な文字列や、実質的に内容がない文字列を含まない場合に成功とみなす
                error_indicators = [
                    "[エラーのため",
                    "[画像データエラー",
                    "[テキスト抽出なし]",
                    "[テキスト抽出キーなし]",
                    "[APIエラーのため抽出失敗",
                    "[LLM応答形式エラーのため抽出失敗"
                ]
                if not any(indicator in ocr_text_single for indicator in error_indicators):
                    if len(ocr_text_single.strip()) > 0:
                        current_ocr_is_successful = True
            if current_ocr_is_successful:
                has_successful_ocr = True # 一つでも成功があればフラグを立てる

            ocr_texts_for_prompt_parts.append(
                f"--- 画像「{filename}」(種別: {img_type_str}) の抽出テキスト ---\n{ocr_text_display}"
            )
        if not has_successful_ocr:
            print("Warning in tutor_logic: No successful OCR results from any image.")
            # 必要に応じてここで return {"error": "全ての画像から有効なテキストを抽出できませんでした。"} なども可
            # combined_ocr_and_type_info_for_prompt = "なし" はこの時点ではセットしない

        if ocr_texts_for_prompt_parts:
            combined_ocr_and_type_info_for_prompt = "\n\n".join(ocr_texts_for_prompt_parts)
            print(f"Tutor Logic: Combined OCR & Type Info (first 200 chars for prompt): '{combined_ocr_and_type_info_for_prompt[:200]}...'")
        else:
            print("Warning in tutor_logic: No OCR parts to combine, though images were processed. Setting combined info to 'なし'.")
            combined_ocr_and_type_info_for_prompt = "なし"

    analysis_input_ocr_text = combined_ocr_and_type_info_for_prompt
    print(f"Tutor Logic: Calling initial analysis. Query: '{query_text[:50]}...', Combined OCR/Type Info: '{str(analysis_input_ocr_text)[:50] if analysis_input_ocr_text else 'No OCR/Type Info'}'")
    analysis_result: Optional[InitialAnalysisResult] = gemini_service.analyze_initial_input_with_ocr(
        query_text=query_text,
        combined_ocr_and_type_info=analysis_input_ocr_text
    )
    print(f"Tutor Logic: Received final analysis result from LLM: {analysis_result}")

    if analysis_result is None:
        return {"error": "AIによる質問の分析処理中に予期せぬエラーが発生しました（API応答なし）。"}
    if not isinstance(analysis_result, dict):
        return {"error": f"AIによる分析結果が予期しない形式です: {type(analysis_result)}"}
    if "error" in analysis_result:
        return analysis_result
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
        
    reason_ambiguity: str = initial_analysis.get("reason_for_ambiguity", "（追加情報が必要です）")
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
    initial_res_expl: Optional[InitialAnalysisResult] = st.session_state.get("initial_analysis_result")
    if not clarified_request and initial_res_expl:
        clarified_request = initial_res_expl.get("summary", st.session_state.user_query_text)
    elif not clarified_request:
         clarified_request = st.session_state.get("user_query_text", "不明なリクエスト")

    request_category = "不明"
    if initial_res_expl:
        request_category = initial_res_expl.get("request_category", "不明")
    
    explanation_style = st.session_state.get("selected_explanation_style", "detailed")
    
    relevant_context_ocr: Optional[str] = None
    processed_images: Optional[List[ProcessedImageInfo]] = st.session_state.get("processed_image_details_list")
    if processed_images:
        context_parts = []
        for proc_info in processed_images:
            filename = proc_info.get("original_filename", "不明なファイル")
            img_type_str = str(proc_info.get("image_type", ImageType.OTHER))
            ocr_text_single = proc_info.get("ocr_text", "[テキスト抽出なし]")
            context_parts.append(
                f"--- 画像「{filename}」(種別: {img_type_str}) の抽出テキスト ---\n{ocr_text_single}"
            )
        if context_parts:
            relevant_context_ocr = "\n\n".join(context_parts)
    if not relevant_context_ocr and initial_res_expl:
        relevant_context_ocr = initial_res_expl.get("ocr_text_from_extraction_combined")

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