# services/gemini_service.py
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional, Union
from PIL import Image # 画像のMIMEタイプ判定や前処理に使う可能性
import io # バイトデータを扱うため

# Geminiモデルの設定 (後で設定ファイルなどから読み込めるようにしても良い)
TEXT_MODEL_NAME = "gemini-1.5-flash-latest" # または "gemini-pro"
VISION_MODEL_NAME = "gemini-1.5-flash-latest" # Vision APIが使えるモデル (例: "gemini-pro-vision")
                                         # Gemini 1.5 FlashもVisionをサポート

def load_prompt_template(template_name: str) -> str:
    """
    promptsディレクトリからプロンプトテンプレートを読み込む (仮実装、後で詳細化)
    将来的には prompts/{template_name}.md のようなファイルを読み込む
    """
    # TODO: prompts ディレクトリから実際にファイルを読み込む処理を実装する
    if template_name == "initial_analysis":
        # これは仮のプロンプトです。実際にはファイルから読み込みます。
        # また、プロンプトエンジニアリングで改善が必要です。
        return """
        ユーザーから提供された画像 (存在する場合) とテキストクエリを分析してください。

        ユーザーのテキストクエリ: 「{query_text}」
        {image_instructions}

        以下の情報を抽出し、指定されたJSON形式で出力してください。
        1.  **ocr_text**: 画像が存在する場合、画像から抽出した主要なテキスト。存在しない場合は null。
        2.  **request_category**: ユーザーの要求が以下のどのカテゴリに最も当てはまるか1つ選択（文法, 語彙, 長文読解, 英作文, 英文和訳, 構文解釈, 和文英訳, 添削, その他）。
        3.  **topic**: 分類された主題領域や具体的な内容。
        4.  **summary**: 特定された中核的な質問やタスクの簡潔な要約。
        5.  **ambiguity**: このまま直接詳細に解説できるほど要求が明確か ("clear")、それとも診断的な質問を通じた明確化が必要か ("ambiguous")。
        6.  **reason_for_ambiguity**: "ambiguous" と判断した場合、その理由を簡潔に。明確な場合は null。

        出力形式 (JSONのみを返すこと):
        ```json
        {{
            "ocr_text": "...",
            "request_category": "...",
            "topic": "...",
            "summary": "...",
            "ambiguity": "clear" | "ambiguous",
            "reason_for_ambiguity": "..."
        }}
        ```
        """
    # 他のプロンプトテンプレートも同様に追加
    return ""


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
        image_data: Optional[Dict[str, Any]] = None # mime_type, data
    ) -> Optional[Dict[str, Any]]:
    """
    ユーザーの初期入力（テキスト＋画像）を分析し、曖昧さなどを判定する。
    """
    prompt_template = load_prompt_template("initial_analysis")
    if not prompt_template:
        print("Error: Initial analysis prompt template not found.")
        return None

    image_instructions_part = ""
    if image_data:
        image_instructions_part = "提供された画像をOCRし、内容を理解してください。"
        # Visionモデルを使用
        model_to_use = VISION_MODEL_NAME
    else:
        # テキストのみの場合はVisionモデルは不要だが、Gemini 1.5 Flashは両対応なのでそのままでも良い
        model_to_use = TEXT_MODEL_NAME # または VISION_MODEL_NAME のまま

    prompt = prompt_template.format(
        query_text=query_text,
        image_instructions=image_instructions_part
    )

    # generation_config でJSONモードを指定できる場合がある (モデルによる)
    # config = genai.types.GenerationConfig(response_mime_type="application/json")
    # ない場合はプロンプトでJSON出力を強く指示し、レスポンスをパースする

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