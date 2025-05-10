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

# --- 今後追加するLLM呼び出し関数 ---
# def generate_clarification_question_llm(...)
# def analyze_user_clarification_llm(...)
# def generate_explanation_llm(...)
# def generate_followup_response_llm(...)
# def generate_summary_llm(...)