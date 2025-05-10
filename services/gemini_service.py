# services/gemini_service.py
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path
from utils import config_loader  # ★ config_loaderをインポート ★

# --- モデル名とプロンプト名をconfigから取得 ---
TEXT_MODEL_NAME = config_loader.get_text_model_name()
VISION_MODEL_NAME = config_loader.get_vision_model_name()

# プロンプトディレクトリのパス設定
# このファイル(gemini_service.py)の親ディレクトリ(services)の親ディレクトリ(プロジェクトルート)の下のprompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SYSTEM_PROMPT_FILENAME = config_loader.get_prompt_template_name("system_prompt")
if SYSTEM_PROMPT_FILENAME is None:
    raise ValueError("System prompt filename not found in config.yaml")

# --- システムプロンプトの読み込み ---
def load_system_prompt() -> Optional[str]:
    system_prompt_path = PROMPTS_DIR / f"{SYSTEM_PROMPT_FILENAME}.md"  # ★ configから取得したファイル名を使用 ★
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: System prompt file not found at {system_prompt_path}")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Error reading system prompt file {system_prompt_path}: {e}")
        return None

SYSTEM_PROMPT_CONTENT = load_system_prompt()
if SYSTEM_PROMPT_CONTENT is None:
    print("Failed to load system prompt. API calls may not behave as expected.")

def load_prompt_template(template_key: str) -> Optional[str]:  # ★ 引数をキーに変更 ★
    """
    promptsディレクトリから指定されたキーに対応するプロンプトテンプレートを読み込む。
    キーはconfig.yamlのprompt_templatesで定義されているもの。
    """
    template_filename = config_loader.get_prompt_template_name(template_key)
    if template_filename is None:
        print(f"Error: Prompt template key '{template_key}' not found in config.")
        return None
    prompt_file_path = PROMPTS_DIR / f"{template_filename}.md"
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at {prompt_file_path}")
        return None
    except Exception as e:
        print(f"Error reading prompt template file {prompt_file_path}: {e}")
        return None

def call_gemini_api(
    prompt_or_contents: Union[str, List[Any]],
    model_name: Optional[str] = None,  # ★ Optionalに変更し、デフォルトをNoneに ★
    generation_config: Optional[genai.types.GenerationConfigDict] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    is_json_output: bool = False
) -> Union[str, Dict[str, Any], None]:
    final_model_name = model_name if model_name else TEXT_MODEL_NAME  # ★ デフォルトモデルを設定 ★
    if SYSTEM_PROMPT_CONTENT is None:
        print("Warning: System prompt is not loaded. Proceeding without system instruction.")
        current_system_instruction = None
    else:
        current_system_instruction = SYSTEM_PROMPT_CONTENT
    try:
        model = genai.GenerativeModel(
            final_model_name,  # ★ final_model_name を使用 ★
            system_instruction=current_system_instruction
        )
        if isinstance(prompt_or_contents, str):
            final_contents = [prompt_or_contents]
        else:
            final_contents = prompt_or_contents
        response = model.generate_content(
            final_contents,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        if is_json_output:
            text_response = response.text.strip()
            if text_response.startswith("```json"):
                text_response = text_response[len("```json"):].strip()
            if text_response.endswith("```"):
                text_response = text_response[:-len("```")].strip()
            try:
                return json.loads(text_response)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from LLM response: {e}")
                print(f"LLM Raw Response: {response.text}")
                return {"error": "Failed to parse JSON response", "raw_response": response.text}
        else:
            return response.text
    except Exception as e:
        print(f"Error calling Gemini API ({final_model_name}) with system instruction: {e}")  # ★ final_model_name を使用 ★
        return None

def extract_text_from_image_llm(
        image_data: Dict[str, Any] # mime_type, data
    ) -> Optional[str]:
    """
    画像データからテキストを抽出する (OCR)。
    """
    prompt_template = config_loader.get_prompt_template_name("extract_text_from_image")
    if prompt_template is None: # config_loaderからキー名で取得
        prompt_text = "この画像からテキストを抽出してください。" # フォールバックプロンプト
        print("Warning: OCR prompt template key 'extract_text_from_image' not found in config, using default prompt.")
    else:
        prompt_text_from_file = load_prompt_template("extract_text_from_image") # キー名でテンプレートロード
        if prompt_text_from_file is None:
             print("Error: OCR prompt template could not be loaded.")
             return None # またはエラーを示す文字列
        prompt_text = prompt_text_from_file


    # Visionモデルを使用
    # call_gemini_api に渡すのは contents リスト
    contents_for_api = [
        {'mime_type': image_data['mime_type'], 'data': image_data['data']},
        prompt_text # プロンプトをテキストとして追加
    ]

    ocr_result_text = call_gemini_api(
        contents_for_api,
        model_name=VISION_MODEL_NAME, # 必ずVisionモデル
        is_json_output=False # OCR結果はプレーンテキストを期待
    )
    
    if ocr_result_text is None or isinstance(ocr_result_text, dict): # dictはエラーの場合
        return None # エラー時はNone
    return ocr_result_text.strip()


def analyze_initial_input_with_ocr( # 関数名を変更 (旧 analyze_initial_input)
        query_text: str,
        ocr_text_from_image: Optional[str] = None # OCR結果を引数で受け取る
    ) -> Optional[Dict[str, Any]]:
    """
    ユーザーのテキストクエリとOCR結果（あれば）を分析し、曖昧さなどを判定する。
    """
    prompt_template = load_prompt_template("analyze_query_with_ocr") # 新しいプロンプトキー
    if prompt_template is None:
        return {"error": "システムエラー: 分析プロンプトを読み込めませんでした。"}

    ocr_text_to_pass = ocr_text_from_image if ocr_text_from_image else "なし"

    try:
        prompt = prompt_template.format(
            query_text=query_text,
            ocr_text_from_image=ocr_text_to_pass
        )
    except KeyError as e:
        return {"error": f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"}

    # この分析はテキストベースでOK (OCR結果は既にテキストとして入力されているため)
    # ただし、元の画像もコンテキストとしてLLMに見せたい場合は、VISION_MODELと画像データを渡す必要があるが、
    # 今回はOCRテキストのみを利用する方針。
    result = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME, # テキストモデルでOK
        is_json_output=True
    )
    return result


def analyze_user_clarification_llm(
        original_query_text: str,
        original_image_provided: bool,
        reason_for_ambiguity: str,         # 初期分析での曖昧さの理由
        llm_clarification_question: str,   # AIがした明確化質問
        user_response_text: str,           # それに対するユーザーの応答
        conversation_history: List[Dict[str, str]] # これまでの全会話履歴
    ) -> Optional[Dict[str, Any]]:
    """
    ユーザーの明確化応答を分析し、曖昧さが解消されたか判定する。
    """
    prompt_template = load_prompt_template("analyze_clarification_response") # config経由
    if prompt_template is None:
        print("Error: Analyze clarification response prompt template could not be loaded.")
        return {"error": "システムエラー: 応答分析プロンプトを読み込めませんでした。"}

    original_image_info_text = "(画像も提供されていました)" if original_image_provided else "(画像はありませんでした)"

    # 会話履歴をテキスト形式に整形 (プロンプトの{conversation_history}用)
    # この履歴には、AIの明確化質問とユーザーの最新応答も含まれているべき
    history_text_for_prompt = ""
    for message in conversation_history:
        role_display = "生徒" if message["role"] == "user" else "AI"
        if message["role"] == "system": continue
        history_text_for_prompt += f"{role_display}: {message['content']}\n"
    if not history_text_for_prompt:
        history_text_for_prompt = "（会話履歴なし）"


    try:
        prompt = prompt_template.format(
            original_user_query=original_query_text,
            original_image_info=original_image_info_text,
            reason_for_ambiguity=reason_for_ambiguity,
            llm_clarification_question=llm_clarification_question,
            user_response_text=user_response_text,
            conversation_history=history_text_for_prompt.strip()
        )
    except KeyError as e:
        print(f"Error formatting analyze clarification response prompt. Missing key: {e}")
        return {"error": f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"}

    result = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=True
    )
    return result

def generate_clarification_question_llm(
        original_query_text: str,
        reason_for_ambiguity: str,
        image_provided: bool,
        conversation_history: List[Dict[str, str]] # 会話履歴を引数に追加
    ) -> Optional[str]:
    """
    曖昧なユーザーの質問に対して、明確化のための質問をLLMに生成させる。
    """
    prompt_template = load_prompt_template("clarification_question") # config経由でファイル名取得
    if prompt_template is None:
        print("Error: Clarification question prompt template could not be loaded.")
        return "システムエラー: 明確化質問プロンプトを読み込めませんでした。"

    original_image_info_text = "(画像も提供されています)" if image_provided else "(画像はありませんでした)" # プロンプトに合わせて調整

    # 会話履歴をテキスト形式に整形
    history_text_for_prompt = ""
    # このプロンプトのコンテキストでは、直近のユーザーの質問とAIの初期応答が重要
    # 全履歴を渡すか、関連部分を抽出するかはプロンプトの設計とトークン数による
    # 今回は、ユーザーの最初の質問とAIの初期応答（曖昧さ指摘）が含まれる履歴を想定
    # 簡易的に、渡されたconversation_historyをそのまま使う
    for message in conversation_history:
        role_display = "生徒" if message["role"] == "user" else "AI"
        if message["role"] == "system": # システムメッセージは含めない
            continue
        history_text_for_prompt += f"{role_display}: {message['content']}\n"
    if not history_text_for_prompt: # 履歴が空なら (通常はユーザーの最初の質問があるはず)
        history_text_for_prompt = "（まだ会話はありません）"


    try:
        prompt = prompt_template.format(
            original_user_query=original_query_text,
            original_image_info=original_image_info_text,
            reason_for_ambiguity=reason_for_ambiguity,
            conversation_history=history_text_for_prompt.strip() # 整形した会話履歴
        )
    except KeyError as e:
        print(f"Error formatting clarification prompt. Missing key: {e}")
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         return "AIが明確化のための質問を生成できませんでした。もう一度試しますか？"
    
    return response_text.strip()


def generate_explanation_llm(
        clarified_request: str,
        request_category: str,
        explanation_style: str,
        relevant_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str,str]]] = None
    ) -> Optional[str]:
    """
    明確化されたリクエストと選択されたスタイルに基づき、解説をLLMに生成させる。
    """
    prompt_template = load_prompt_template("generate_explanation")  # ★ キーで指定 ★
    if prompt_template is None:
        return "システムエラー: 解説生成プロンプトを読み込めませんでした。"

    style_instructions = ""
    if explanation_style == "detailed":
        style_instructions = "このリクエストに対して、関連するルール、定義、具体例、ステップバイステップの手順などを網羅的に、しかし段階的に分かりやすく説明してください。必要であれば、関連する文法事項や語彙についても触れてください。"
    elif explanation_style == "hint":
        style_instructions = "このリクエストを解決するための、核心に迫るヒントや手がかりを1つか2つ提示してください。直接的な答えや詳細な説明は避け、生徒自身が考えるきっかけとなるようにしてください。"
    elif explanation_style == "socratic":
        style_instructions = "このリクエストについて生徒自身の思考を促すような、核心的な理解を問う質問をいくつか投げかけてください。生徒が自ら答えにたどり着くための誘導的な問いかけを、段階的に行ってください。"
    else:
        style_instructions = "このリクエストに対して、関連するルール、定義、具体例、ステップバイステップの手順などを網羅的に、しかし段階的に分かりやすく説明してください。"

    history_summary_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history]) if conversation_history else "なし"
    context_text = relevant_context if relevant_context else "なし"

    try:
        prompt = prompt_template.format(
            clarified_request=clarified_request,
            request_category=request_category,
            explanation_style=explanation_style,
            relevant_context=context_text,
            conversation_history_summary=history_summary_text,
            style_specific_instructions=style_instructions
        )
    except KeyError as e:
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,  # 明示的に指定
        is_json_output=False
    )
    if response_text is None or isinstance(response_text, dict):
         return "AIが解説を生成できませんでした。"
    return response_text.strip()

def generate_followup_response_llm(
        conversation_history: List[Dict[str, str]],
        user_latest_input: str
    ) -> Optional[str]:
    """
    会話履歴とユーザーの最新の入力に基づき、フォローアップ応答をLLMに生成させる。
    """
    prompt_template = load_prompt_template("generate_followup")  # ★ キーで指定 ★
    if prompt_template is None:
        return "システムエラー: フォローアップ応答プロンプトを読み込めませんでした。"

    history_text = ""
    for message in conversation_history:
        role_display = "生徒" if message["role"] == "user" else "AI"
        history_text += f"{role_display}: {message['content']}\n"
    if conversation_history and conversation_history[-1]["role"] == "user" and conversation_history[-1]["content"] == user_latest_input:
         history_for_prompt = conversation_history[:-1]
    else:
         history_for_prompt = conversation_history
    history_text_for_prompt = ""
    for message in history_for_prompt:
        role_display = "生徒" if message["role"] == "user" else "AI"
        history_text_for_prompt += f"{role_display}: {message['content']}\n"
    try:
        prompt = prompt_template.format(
            conversation_history=history_text_for_prompt.strip(),
            user_latest_input=user_latest_input
        )
    except KeyError as e:
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,  # 明示的に指定
        is_json_output=False
    )
    if response_text is None or isinstance(response_text, dict):
         return "AIが応答を生成できませんでした。"
    return response_text.strip()

def generate_summary_llm(
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
    """
    会話履歴に基づき、セッションの要約と持ち帰りメッセージをLLMに生成させる。
    """
    prompt_template = load_prompt_template("generate_summary")  # ★ キーで指定 ★
    if prompt_template is None:
        return "システムエラー: 要約生成プロンプトを読み込めませんでした。"

    history_text = ""
    for message in conversation_history:
        if message["role"] != "system":
            role_display = "生徒" if message["role"] == "user" else "AI"
            history_text += f"{role_display}: {message['content']}\n"
    try:
        prompt = prompt_template.format(conversation_history=history_text.strip())
    except KeyError as e:
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,  # 明示的に指定
        is_json_output=False
    )
    if response_text is None or isinstance(response_text, dict):
         return "AIが要約を生成できませんでした。"
    return response_text.strip()