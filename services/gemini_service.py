# services/gemini_service.py
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path
from utils import config_loader  # ★ config_loaderをインポート ★
# core から型定義をインポート (循環依存に注意しつつ、今回は型定義のみなので許容範囲とする)
# 本来は services と core が依存しない共通の型定義モジュールが良い
from core.type_definitions import ChatMessage, UploadedFileData, InitialAnalysisResult, ClarificationAnalysisResult

# --- モデル名とプロンプト名をconfigから取得 ---
TEXT_MODEL_NAME = config_loader.get_text_model_name()
VISION_MODEL_NAME = config_loader.get_vision_model_name()

# プロンプトディレクトリのパス設定
# このファイル(gemini_service.py)の親ディレクトリ(services)の親ディレクトリ(プロジェクトルート)の下のprompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SYSTEM_PROMPT_FILENAME = config_loader.get_prompt_template_name(config_loader.PROMPT_KEY_SYSTEM)
if SYSTEM_PROMPT_FILENAME is None:
    raise ValueError("System prompt filename not found in config.yaml")

# --- システムプロンプトの読み込み ---
def load_system_prompt() -> Optional[str]:
    """システムプロンプトファイルを読み込む。見つからない場合やエラー時はNoneを返す。"""
    system_prompt_path = PROMPTS_DIR / f"{SYSTEM_PROMPT_FILENAME}.md"
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # f-string内でのバックスラッシュエスケープに注意 (直接 Path オブジェクトを使う方が安全)
        print(f"CRITICAL ERROR: System prompt file not found at {system_prompt_path.resolve()}")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Error reading system prompt file {system_prompt_path.resolve()}: {e}")
        return None

SYSTEM_PROMPT_CONTENT = load_system_prompt()
if SYSTEM_PROMPT_CONTENT is None:
    # 起動時に警告を出すが、処理は続行させる（フォールバックがある場合を想定）
    print("WARNING: Failed to load system prompt. API calls may not behave as expected or may lack core instructions.")

def _format_conversation_history_for_prompt(
    conversation_history: Optional[List[ChatMessage]], # TypedDict List を受け取るように変更
    default_empty_message: str = "（会話履歴なし）",
    exclude_system_messages: bool = True,
    user_role_name: str = "生徒",
    assistant_role_name: str = "AI"
) -> str:
    """指定された会話履歴オブジェクトを、LLMプロンプト用の単一文字列に整形するヘルパー関数。

    Args:
        conversation_history: ChatMessageオブジェクトのリスト、またはNone。
        default_empty_message: 履歴が空の場合に使用する文字列。
        exclude_system_messages: Trueの場合、roleが"system"のメッセージを除外する。
        user_role_name: "user"ロールの表示名。
        assistant_role_name: "assistant"ロールの表示名。

    Returns:
        整形された会話履歴文字列。
    """
    if not conversation_history:
        return default_empty_message

    history_text = ""
    for message in conversation_history:
        if not isinstance(message, dict):
            # 予期せぬ型の場合のスキップ処理（より堅牢にするならログ出力）
            print(f"Warning (_format_conversation_history_for_prompt): Skipping non-dict message: {message}")
            continue
            
        role = message.get("role")
        content = message.get("content", "") # contentがない場合は空文字

        if exclude_system_messages and role == "system":
            continue
        
        role_display = ""
        if role == "user":
            role_display = user_role_name
        elif role == "assistant":
            role_display = assistant_role_name
        elif role == "system":
            role_display = "システム"
        else: 
            role_display = str(role) # 未知のロールはそのまま表示

        history_text += f"{role_display}: {content}\n"
    
    formatted_history = history_text.strip()
    return formatted_history if formatted_history else default_empty_message

def load_prompt_template(template_key: str) -> Optional[str]:
    """指定されたキーに対応するプロンプトテンプレートファイルを読み込む。
    config_loaderを通じてファイル名を取得し、PROMPTS_DIRからファイルを読み取る。
    ファイルが見つからない、または読み込みエラーの場合はNoneを返す。
    """
    template_filename = config_loader.get_prompt_template_name(template_key)
    if template_filename is None:
        # このエラーは config_loader 側でもログ出力されるが、こちらでもキー名を添えて明確にする
        print(f"Critical Error: Prompt template key '{template_key}' not found in config.yaml.")
        return None
        
    prompt_file_path = PROMPTS_DIR / f"{template_filename}.md"
    # print(f"Attempting to load prompt from: {prompt_file_path.resolve()}") # デバッグ用ログ
    if not prompt_file_path.exists():
        print(f"Critical Error: Prompt template file does not exist at {prompt_file_path.resolve()}")
        return None
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Critical Error: Error reading prompt template file {prompt_file_path.resolve()}: {e}")
        return None

def call_gemini_api(
    prompt_or_contents: Union[str, List[Any]], # 文字列プロンプトまたはVision API用のcontentsリスト
    model_name: Optional[str] = None, 
    generation_config: Optional[genai.types.GenerationConfigDict] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    is_json_output: bool = False # Trueの場合、レスポンスをJSONとしてパース試行
) -> Union[str, Dict[str, Any], None]:
    """Google Generative AI API (Gemini) を呼び出す共通関数。

    Args:
        prompt_or_contents: LLMに渡すプロンプト文字列、または画像を含む場合はcontentsリスト。
        model_name: 使用するモデル名。Noneの場合はTEXT_MODEL_NAMEが使用される。
        generation_config: 生成時の設定 (temperature, top_pなど)。
        safety_settings: 安全性設定。
        is_json_output: レスポンスがJSON形式であることを期待し、パースするかどうか。

    Returns:
        Union[str, Dict[str, Any], None]: LLMからのレスポンステキスト、パースされたJSON辞書、またはエラー時にNone。
                                           JSONパース失敗時は{"error": ..., "raw_response": ...} を返す。
    """
    final_model_name = model_name if model_name else TEXT_MODEL_NAME
    
    # システムプロンプトがロードされていれば、それをモデルに設定
    current_system_instruction = SYSTEM_PROMPT_CONTENT if SYSTEM_PROMPT_CONTENT else None
    if current_system_instruction is None:
        print(f"Warning: System prompt content is not loaded. Calling API for model {final_model_name} without system instruction.")

    try:
        model = genai.GenerativeModel(
            final_model_name,
            system_instruction=current_system_instruction
        )
        
        # APIに渡す形式をcontentsリストに統一
        final_contents = [prompt_or_contents] if isinstance(prompt_or_contents, str) else prompt_or_contents
        
        response = model.generate_content(
            final_contents,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if is_json_output:
            text_response = response.text.strip()
            # LLMがJSONをマークダウンブロックで囲む場合があるため、除去処理
            if text_response.startswith("```json"):
                text_response = text_response[len("```json"):].strip()
            if text_response.endswith("```"):
                text_response = text_response[:-len("```")].strip()
            try:
                return json.loads(text_response)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from LLM response (model: {final_model_name}): {e}")
                print(f"LLM Raw Response for JSON (model: {final_model_name}): {response.text}")
                return {"error": "LLMからのJSON応答の解析に失敗しました。", "raw_response": response.text}
        else:
            return response.text
    except Exception as e:
        # API呼び出し中の予期せぬエラー (認証エラー、クォータ超過、ネットワーク問題など)
        print(f"Error calling Gemini API (model: {final_model_name}): {e}")
        return None # エラー時はNoneを返す (呼び出し元で処理)

def extract_text_from_image_llm(
        image_data: UploadedFileData 
    ) -> Optional[str]:
    """画像データからテキストを抽出する (OCR)。Visionモデルを使用。
    プロンプト: "extract_text_from_image"
    """
    prompt_text_from_file = load_prompt_template(config_loader.PROMPT_KEY_EXTRACT_TEXT_FROM_IMAGE)
    if prompt_text_from_file is None:
            # ログはload_prompt_template内で出力されるため、ここでは追加のログは不要
            print(f"Error: OCR prompt template {config_loader.PROMPT_KEY_EXTRACT_TEXT_FROM_IMAGE} could not be loaded. OCR will fail.")
            return None 
    prompt_text = prompt_text_from_file

    contents_for_api = [
        {'mime_type': image_data['mime_type'], 'data': image_data['data']},
        prompt_text
    ]

    ocr_result_text = call_gemini_api(
        contents_for_api,
        model_name=VISION_MODEL_NAME, 
        is_json_output=False # OCR結果はプレーンテキストを期待
    )
    
    if ocr_result_text is None or isinstance(ocr_result_text, dict):
        print(f"OCR attempt with model {VISION_MODEL_NAME} failed or returned unexpected format.")
        return None 
    return ocr_result_text.strip()


def analyze_initial_input_with_ocr(
        query_text: str,
        ocr_text_from_image: Optional[str] = None
    ) -> Optional[InitialAnalysisResult]:
    """ユーザーのテキストクエリとOCR結果（あれば）を分析し、曖昧さなどを判定する。
    プロンプト: "analyze_query_with_ocr" (JSON出力を期待)
    期待するプレースホルダ: {query_text}, {ocr_text_from_image}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_ANALYZE_QUERY_WITH_OCR)
    if prompt_template is None:
        return {"error": f"システムエラー: 分析プロンプト '{config_loader.PROMPT_KEY_ANALYZE_QUERY_WITH_OCR}' を読み込めませんでした。"} # type: ignore

    ocr_text_to_pass = ocr_text_from_image if ocr_text_from_image else "なし"

    try:
        prompt = prompt_template.format(
            query_text=query_text,
            ocr_text_from_image=ocr_text_to_pass
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_ANALYZE_QUERY_WITH_OCR
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return {"error": error_message} # type: ignore

    result = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=True
    )
    if isinstance(result, dict):
        # errorキーが内部に含まれる場合も、そのままTypedDictとしてキャストされることを期待
        return result # type: ignore 
    # JSON出力を期待しているので、文字列が返ってきた場合はエラーとして扱う
    print(f"Warning: analyze_initial_input_with_ocr expected JSON but received non-dict: {result}")
    return None

def analyze_user_clarification_llm(
        original_query_text: str,
        original_image_provided: bool,
        reason_for_ambiguity: str,
        llm_clarification_question: str,
        user_response_text: str,
        conversation_history: List[ChatMessage]
    ) -> Optional[ClarificationAnalysisResult]:
    """ユーザーの明確化応答を分析し、曖昧さが解消されたか判定する。
    プロンプト: "analyze_clarification_response" (JSON出力を期待)
    期待するプレースホルダ: {original_user_query}, {original_image_info}, {reason_for_ambiguity}, 
                         {llm_clarification_question}, {user_response_text}, {conversation_history}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_ANALYZE_CLARIFICATION_RESPONSE)
    if prompt_template is None:
        # print(...) はload_prompt_template内で行われる
        return {"error": f"システムエラー: 応答分析プロンプト '{config_loader.PROMPT_KEY_ANALYZE_CLARIFICATION_RESPONSE}' を読み込めませんでした。"} # type: ignore

    original_image_info_text = "(画像も提供されていました)" if original_image_provided else "(画像はありませんでした)"
    history_text_for_prompt = _format_conversation_history_for_prompt(
        conversation_history,
        default_empty_message="（会話履歴なし）",
        exclude_system_messages=True
    )

    try:
        prompt = prompt_template.format(
            original_user_query=original_query_text,
            original_image_info=original_image_info_text,
            reason_for_ambiguity=reason_for_ambiguity,
            llm_clarification_question=llm_clarification_question,
            user_response_text=user_response_text,
            conversation_history=history_text_for_prompt
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_ANALYZE_CLARIFICATION_RESPONSE
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return {"error": error_message} # type: ignore

    result = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=True
    )
    if isinstance(result, dict):
        return result # type: ignore
    print(f"Warning: analyze_user_clarification_llm expected JSON but received non-dict: {result}")
    return None

def generate_clarification_question_llm(
        original_query_text: str,
        reason_for_ambiguity: str,
        image_provided: bool,
        conversation_history: List[ChatMessage]
    ) -> Optional[str]:
    """曖昧なユーザーの質問に対し、明確化のための質問をLLMに生成させる。
    プロンプト: "clarification_question"
    期待するプレースホルダ: {original_user_query}, {original_image_info}, {reason_for_ambiguity}, {conversation_history}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_CLARIFICATION_QUESTION)
    if prompt_template is None:
        return f"システムエラー: 明確化質問プロンプト '{config_loader.PROMPT_KEY_CLARIFICATION_QUESTION}' を読み込めませんでした。"

    original_image_info_text = "(画像も提供されています)" if image_provided else "(画像はありませんでした)"
    history_text_for_prompt = _format_conversation_history_for_prompt(
        conversation_history,
        default_empty_message="（まだ会話はありません）",
        exclude_system_messages=True
    )

    try:
        prompt = prompt_template.format(
            original_user_query=original_query_text,
            original_image_info=original_image_info_text,
            reason_for_ambiguity=reason_for_ambiguity,
            # _format_conversation_history_for_prompt でstripされるので、ここでは不要
            conversation_history=history_text_for_prompt 
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_CLARIFICATION_QUESTION
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return error_message

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         # API呼び出し失敗または予期せぬJSON応答
         print(f"Warning: generate_clarification_question_llm received None or dict: {response_text}")
         return "AIが明確化のための質問を生成できませんでした。もう一度お試しいただくか、質問を変えてみてください。"
    
    return response_text.strip()

def generate_explanation_llm(
        clarified_request: str,
        request_category: str,
        explanation_style: str,
        relevant_context: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> Optional[str]:
    """明確化されたリクエストと選択されたスタイルに基づき、解説をLLMに生成させる。
    プロンプト: "generate_explanation"
    期待するプレースホルダ: {clarified_request}, {request_category}, {explanation_style}, {relevant_context}, {conversation_history}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_GENERATE_EXPLANATION)
    if prompt_template is None:
        return f"システムエラー: 解説生成プロンプト '{config_loader.PROMPT_KEY_GENERATE_EXPLANATION}' を読み込めませんでした。"

    history_text_for_prompt = _format_conversation_history_for_prompt(
        conversation_history,
        default_empty_message="（これまでの会話はありません）",
        exclude_system_messages=True
    )
    context_text_to_pass = relevant_context if relevant_context else "なし"

    try:
        prompt = prompt_template.format(
            clarified_request=clarified_request,
            request_category=request_category,
            explanation_style=explanation_style,
            relevant_context=context_text_to_pass,
            conversation_history=history_text_for_prompt
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_GENERATE_EXPLANATION
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return error_message

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         print(f"Warning: generate_explanation_llm received None or dict: {response_text}")
         return "AIが解説を生成できませんでした。お手数ですが、もう一度お試しください。"
    
    return response_text.strip()

def generate_followup_response_llm(
        conversation_history: List[ChatMessage],
        user_latest_input: str
    ) -> Optional[str]:
    """会話履歴とユーザーの最新の入力に基づき、フォローアップ応答をLLMに生成させる。
    プロンプト: "generate_followup"
    期待するプレースホルダ: {conversation_history}, {user_followup_text}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_GENERATE_FOLLOWUP)
    if prompt_template is None:
        return f"システムエラー: フォローアップ応答プロンプト '{config_loader.PROMPT_KEY_GENERATE_FOLLOWUP}' を読み込めませんでした。"

    history_text_for_prompt = _format_conversation_history_for_prompt(
        conversation_history,
        default_empty_message="（会話履歴なし）",
        exclude_system_messages=True
    )

    try:
        prompt = prompt_template.format(
            conversation_history=history_text_for_prompt,
            user_followup_text=user_latest_input 
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_GENERATE_FOLLOWUP
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return error_message

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         print(f"Warning: generate_followup_response_llm received None or dict: {response_text}")
         return "AIが応答を生成できませんでした。もう一度入力してください。"
    
    return response_text.strip()

def generate_summary_llm(
        request_category: str,
        topic: str,
        conversation_history: List[ChatMessage]
    ) -> Optional[str]:
    """会話履歴に基づき、セッションの要約と持ち帰りメッセージをLLMに生成させる。
    プロンプト: "generate_summary"
    期待するプレースホルダ: {request_category}, {topic}, {relevant_history_summary}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_GENERATE_SUMMARY)
    if prompt_template is None:
        return f"システムエラー: 要約生成プロンプト '{config_loader.PROMPT_KEY_GENERATE_SUMMARY}' を読み込めませんでした。"

    history_text_for_prompt = _format_conversation_history_for_prompt(
        conversation_history,
        default_empty_message="（要約対象の会話履歴がありません）",
        exclude_system_messages=True
    )

    try:
        prompt = prompt_template.format(
            request_category=request_category,
            topic=topic,
            relevant_history_summary=history_text_for_prompt
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_GENERATE_SUMMARY
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return error_message

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         print(f"Warning: generate_summary_llm received None or dict: {response_text}")
         return "AIが要約を生成できませんでした。"
    
    return response_text.strip()

def analyze_student_performance_llm(
        student_performance_data_str: str
    ) -> Optional[str]:
    """生徒のパフォーマンスデータを分析し、理解度やCEFRレベルを推定する。
    プロンプト: "analyze_student_performance"
    期待するプレースホルダ: {student_performance_data}
    """
    prompt_template = load_prompt_template(config_loader.PROMPT_KEY_ANALYZE_STUDENT_PERFORMANCE)
    if prompt_template is None:
        return f"システムエラー: 生徒能力分析プロンプト '{config_loader.PROMPT_KEY_ANALYZE_STUDENT_PERFORMANCE}' を読み込めませんでした。"

    try:
        prompt = prompt_template.format(
            student_performance_data=student_performance_data_str
        )
    except KeyError as e:
        template_key_name = config_loader.PROMPT_KEY_ANALYZE_STUDENT_PERFORMANCE
        error_message = f"システムエラー: プロンプト '{template_key_name}' のフォーマットに失敗しました。プレースホルダ {e} が不足しています。"
        print(f"KeyError during prompt formatting for {template_key_name}: Missing key {e}")
        return error_message

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         print(f"Warning: analyze_student_performance_llm received None or dict: {response_text}")
         return "AIが生徒の能力分析を生成できませんでした。"
    
    return response_text.strip()