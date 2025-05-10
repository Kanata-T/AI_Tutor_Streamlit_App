# services/gemini_service.py
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional, Union
from PIL import Image # 画像のMIMEタイプ判定や前処理に使う可能性
import io # バイトデータを扱うため
import os  # osモジュールをインポート
from pathlib import Path  # pathlibをインポート

# Geminiモデルの設定 (後で設定ファイルなどから読み込めるようにしても良い)
TEXT_MODEL_NAME = "gemini-1.5-flash-latest" # または "gemini-pro"
VISION_MODEL_NAME = "gemini-1.5-flash-latest" # Vision APIが使えるモデル (例: "gemini-pro-vision")
                                         # Gemini 1.5 FlashもVisionをサポート

# プロンプトディレクトリのパス設定
# このファイル(gemini_service.py)の親ディレクトリ(services)の親ディレクトリ(プロジェクトルート)の下のprompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt_template(template_name: str) -> Optional[str]:
    """
    promptsディレクトリから指定された名前のプロンプトテンプレート (.md ファイル) を読み込む。
    ファイルが見つからない場合は None を返す。
    """
    prompt_file_path = PROMPTS_DIR / f"{template_name}.md"
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
    prompt: str,
    model_name: str = TEXT_MODEL_NAME,
    image_data: Optional[Dict[str, Any]] = None, # {"mime_type": "image/jpeg", "data": bytes_data}
    generation_config: Optional[genai.types.GenerationConfigDict] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    is_json_output: bool = False
) -> Union[str, Dict[str, Any], None]:
    """
    Gemini APIを呼び出す汎用関数。
    :param prompt: LLMへのプロンプト文字列。
    :param model_name: 使用するモデル名。
    :param image_data: 画像データ (MIMEタイプとバイトデータを含む辞書)。Visionモデルの場合に指定。
    :param generation_config: 生成設定 (temperatureなど)。
    :param safety_settings: 安全性設定。
    :param is_json_output: 出力がJSON形式であることを期待するかどうか。
    :return: LLMからのレスポンス (テキストまたはパースされたJSON)、エラー時はNone。
    """
    try:
        model = genai.GenerativeModel(model_name)
        
        contents = []
        if image_data and image_data.get("data") and image_data.get("mime_type"):
            # 画像データを追加 (Pillowオブジェクトではなく、バイトデータとMIMEタイプを直接渡す)
            img_blob = {'mime_type': image_data['mime_type'], 'data': image_data['data']}
            contents.append(img_blob)
        
        contents.append(prompt) # テキストプロンプトを追加

        # デバッグ用に送信するコンテンツを表示
        # print(f"--- Sending to Gemini ({model_name}) ---")
        # print(f"Prompt part: {prompt[:200]}...") # プロンプトの一部
        # if image_data:
        #     print(f"Image MIME type: {image_data['mime_type']}")
        # print("------------------------------------")

        response = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # print(f"--- Received from Gemini ---") # デバッグ用
        # print(response.text)
        # print("-----------------------------")


        if is_json_output:
            # LLMの出力からJSON部分を抽出する（Markdownのコードブロック ```json ... ``` を考慮）
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
                return {"error": "Failed to parse JSON response", "raw_response": response.text} # エラー情報を返す
        else:
            return response.text

    except Exception as e:
        print(f"Error calling Gemini API ({model_name}): {e}")
        # TODO: ここでStreamlitのst.errorなどを使ってユーザーに通知することも検討
        return None


def analyze_initial_input(
        query_text: str,
        image_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
    """
    ユーザーの初期入力（テキスト＋画像）を分析し、曖昧さなどを判定する。
    """
    prompt_template = load_prompt_template("initial_analysis")  # ここでファイルから読み込む
    if prompt_template is None:  # Noneが返ってきたらエラー処理
        print("Error: Initial analysis prompt template could not be loaded.")
        return {"error": "システムエラー: 初期分析プロンプトを読み込めませんでした。"}

    image_instructions_part = ""
    if image_data:
        image_instructions_part = "提供された画像をOCRし、内容を理解してください。"
        model_to_use = VISION_MODEL_NAME
    else:
        model_to_use = TEXT_MODEL_NAME

    try:
        prompt = prompt_template.format(
            query_text=query_text,
            image_instructions=image_instructions_part
        )
    except KeyError as e:
        print(f"Error formatting prompt template. Missing key: {e}")
        return {"error": f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"}

    result = call_gemini_api(
        prompt,
        model_name=model_to_use,
        image_data=image_data,
        is_json_output=True
    )
    return result

def analyze_user_clarification_llm(
        original_query_text: str,
        original_image_provided: bool,
        reason_for_ambiguity: str,
        llm_clarification_question: str,
        user_response_text: str
    ) -> Optional[Dict[str, Any]]:
    """
    ユーザーの明確化応答を分析し、曖昧さが解消されたか判定する。
    """
    prompt_template = load_prompt_template("analyze_clarification_response")
    if prompt_template is None:
        print("Error: Analyze clarification response prompt template could not be loaded.")
        return {"error": "システムエラー: 応答分析プロンプトを読み込めませんでした。"}

    original_image_info_text = "(画像も提供されていました)" if original_image_provided else "(画像はありませんでした)"

    try:
        prompt = prompt_template.format(
            original_user_query=original_query_text,
            original_image_info=original_image_info_text,
            reason_for_ambiguity=reason_for_ambiguity,
            llm_clarification_question=llm_clarification_question,
            user_response_text=user_response_text
        )
    except KeyError as e:
        print(f"Error formatting analyze clarification response prompt. Missing key: {e}")
        return {"error": f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"}

    # この応答はJSON形式を期待する
    result = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=True
    )
    return result

def generate_clarification_question_llm(
        original_query_text: str,
        reason_for_ambiguity: str,
        image_provided: bool # 元の質問に画像があったかどうか
    ) -> Optional[str]:
    """
    曖昧なユーザーの質問に対して、明確化のための質問をLLMに生成させる。
    """
    prompt_template = load_prompt_template("clarification_question")
    if prompt_template is None:
        print("Error: Clarification question prompt template could not be loaded.")
        return "システムエラー: 明確化質問プロンプトを読み込めませんでした。" # エラーメッセージを返す

    original_image_info_text = "(画像も提供されています)" if image_provided else ""

    try:
        prompt = prompt_template.format(
            original_user_query=original_query_text,
            original_image_info=original_image_info_text,
            reason_for_ambiguity=reason_for_ambiguity
        )
    except KeyError as e:
        print(f"Error formatting clarification prompt. Missing key: {e}")
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    # この場合、応答は単純なテキストを期待する
    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME, # テキスト生成モデル
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict): # dictはエラーの場合
         return "AIが明確化のための質問を生成できませんでした。もう一度試しますか？"
    
    return response_text.strip()

def generate_explanation_llm(
        clarified_request: str,
        request_category: str,
        explanation_style: str,
        relevant_context: Optional[str] = None, # OCR結果など
        conversation_history: Optional[List[Dict[str,str]]] = None # 要約された会話履歴
    ) -> Optional[str]:
    """
    明確化されたリクエストと選択されたスタイルに基づき、解説をLLMに生成させる。
    """
    prompt_template = load_prompt_template("generate_explanation")
    if prompt_template is None:
        print("Error: Generate explanation prompt template could not be loaded.")
        return "システムエラー: 解説生成プロンプトを読み込めませんでした。"

    # スタイルに応じた具体的な指示を生成
    style_instructions = ""
    if explanation_style == "detailed":
        style_instructions = "このリクエストに対して、関連するルール、定義、具体例、ステップバイステップの手順などを網羅的に、しかし段階的に分かりやすく説明してください。必要であれば、関連する文法事項や語彙についても触れてください。"
    elif explanation_style == "hint":
        style_instructions = "このリクエストを解決するための、核心に迫るヒントや手がかりを1つか2つ提示してください。直接的な答えや詳細な説明は避け、生徒自身が考えるきっかけとなるようにしてください。"
    elif explanation_style == "socratic":
        style_instructions = "このリクエストについて生徒自身の思考を促すような、核心的な理解を問う質問をいくつか投げかけてください。生徒が自ら答えにたどり着くための誘導的な問いかけを、段階的に行ってください。"
    else: # デフォルトは詳細解説
        style_instructions = "このリクエストに対して、関連するルール、定義、具体例、ステップバイステップの手順などを網羅的に、しかし段階的に分かりやすく説明してください。"

    history_summary_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history]) if conversation_history else "なし"
    context_text = relevant_context if relevant_context else "なし"


    try:
        prompt = prompt_template.format(
            clarified_request=clarified_request,
            request_category=request_category,
            explanation_style=explanation_style, # スタイル名自体も渡す
            relevant_context=context_text,
            conversation_history_summary=history_summary_text,
            style_specific_instructions=style_instructions
        )
    except KeyError as e:
        print(f"Error formatting explanation prompt. Missing key: {e}")
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False # 解説はマークダウン形式のテキストを期待
    )

    if response_text is None or isinstance(response_text, dict): # dictはエラーの場合
         return "AIが解説を生成できませんでした。"
    
    return response_text.strip()


def generate_followup_response_llm(
        conversation_history: List[Dict[str, str]], # "role", "content" のリスト
        user_latest_input: str
    ) -> Optional[str]:
    """
    会話履歴とユーザーの最新の入力に基づき、フォローアップ応答をLLMに生成させる。
    """
    prompt_template = load_prompt_template("generate_followup")
    if prompt_template is None:
        print("Error: Generate followup prompt template could not be loaded.")
        return "システムエラー: フォローアップ応答プロンプトを読み込めませんでした。"

    # 会話履歴をテキスト形式に整形
    history_text = ""
    for message in conversation_history:
        # Gemini APIは parts を直接渡せるので、ここではプロンプト用に整形
        # 実際のAPIコールでは、message["role"]が "user" または "model" (Geminiの期待する形式) になっている必要がある
        # ここではプロンプトテンプレートへの埋め込み用に簡易的に整形
        role_display = "生徒" if message["role"] == "user" else "AI"
        history_text += f"{role_display}: {message['content']}\n"
    
    # 最後のユーザー入力は別途渡すので、履歴からは除く（プロンプトテンプレートの構成による）
    # もしプロンプトが履歴全体を期待し、最後の発言を別途強調するなら調整
    if conversation_history and conversation_history[-1]["role"] == "user" and conversation_history[-1]["content"] == user_latest_input:
         history_for_prompt = conversation_history[:-1] # 最後のユーザー発言は除く
    else:
         history_for_prompt = conversation_history

    history_text_for_prompt = ""
    for message in history_for_prompt:
        role_display = "生徒" if message["role"] == "user" else "AI"
        history_text_for_prompt += f"{role_display}: {message['content']}\n"


    try:
        prompt = prompt_template.format(
            conversation_history=history_text_for_prompt.strip(), # 整形した会話履歴
            user_latest_input=user_latest_input
        )
    except KeyError as e:
        print(f"Error formatting followup prompt. Missing key: {e}")
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    # Gemini APIの `generate_content` は直接 `contents` (roleとpartsのリスト) を渡せる
    # ここではプロンプトテンプレートを使うアプローチを採用しているが、
    # 別の方法として、整形済みの会話履歴リストを直接 `model.generate_content(history_contents)` のように渡すことも可能。
    # その場合、最後のユーザー入力もリストに含める。

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False # 通常のテキスト応答を期待
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
    prompt_template = load_prompt_template("generate_summary")
    if prompt_template is None:
        print("Error: Generate summary prompt template could not be loaded.")
        return "システムエラー: 要約生成プロンプトを読み込めませんでした。"

    history_text = ""
    for message in conversation_history:
        # システムメッセージは要約に含めない方が良い場合がある
        if message["role"] != "system":
            role_display = "生徒" if message["role"] == "user" else "AI"
            history_text += f"{role_display}: {message['content']}\n"
    
    # トークン数制限を考慮し、履歴が長すぎる場合は末尾N件などに絞る処理が必要な場合も
    # (今回は簡易的に全量渡す)

    try:
        prompt = prompt_template.format(conversation_history=history_text.strip())
    except KeyError as e:
        print(f"Error formatting summary prompt. Missing key: {e}")
        return f"システムエラー: プロンプトのフォーマットに失敗しました (キー不足: {e})。"

    response_text = call_gemini_api(
        prompt,
        model_name=TEXT_MODEL_NAME,
        is_json_output=False
    )

    if response_text is None or isinstance(response_text, dict):
         return "AIが要約を生成できませんでした。"
    
    return response_text.strip()