import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List # List を追加
import streamlit as st  # エラー表示用 (任意)

# Prompt Template Keys (これはこのままでOK、gemini_service.py等から参照想定)
PROMPT_KEY_SYSTEM: str = "system_prompt"
# PROMPT_KEY_EXTRACT_TEXT_FROM_IMAGE: str = "extract_text_from_image" # 古いキーはコメントアウトまたは削除
PROMPT_KEY_EXTRACT_TEXT_AND_TYPE_FROM_IMAGE: str = "extract_text_and_type_from_image" # ★新規追加★
PROMPT_KEY_ANALYZE_QUERY_WITH_OCR: str = "analyze_query_with_ocr"
PROMPT_KEY_ANALYZE_CLARIFICATION_RESPONSE: str = "analyze_clarification_response"
PROMPT_KEY_CLARIFICATION_QUESTION: str = "clarification_question"
PROMPT_KEY_GENERATE_EXPLANATION: str = "generate_explanation"
PROMPT_KEY_GENERATE_FOLLOWUP: str = "generate_followup"
PROMPT_KEY_GENERATE_SUMMARY: str = "generate_summary"
PROMPT_KEY_ANALYZE_STUDENT_PERFORMANCE: str = "analyze_student_performance"
PROMPT_KEY_PLAN_GUIDANCE = "plan_guidance"  # 指導計画プロンプト用のキー

CONFIG_FILE_PATH = Path(__file__).parent.parent / "config.yaml"
CONFIG_PATH = CONFIG_FILE_PATH  # 互換性のため

_config_cache: Optional[Dict[str, Any]] = None

def get_config() -> Dict[str, Any]: # 型ヒントを Dict[str, Any] に統一
    """
    設定ファイルを読み込み、キャッシュする。
    エラー時はstreamlitで表示し、空辞書を返す。
    """
    global _config_cache
    if _config_cache is None:
        try:
            if not CONFIG_PATH.is_file(): # ファイル存在チェックを先に行う
                print(f"CRITICAL ERROR: config.yaml not found at {CONFIG_PATH}")
                if 'st' in globals() and hasattr(st, 'error'):
                    st.error(f"システム設定エラー: config.yamlが見つかりません ({CONFIG_PATH})。")
                _config_cache = {}
                return _config_cache # ここでリターン

            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f)
            
            if not isinstance(_config_cache, dict):
                print(f"CRITICAL ERROR: config.yaml is not a valid YAML dictionary.")
                if 'st' in globals() and hasattr(st, 'error'):
                    st.error("システム設定エラー: config.yamlの形式が正しくありません。")
                _config_cache = {}
        except yaml.YAMLError as e:
            print(f"CRITICAL ERROR: Error parsing config.yaml: {e}")
            if 'st' in globals() and hasattr(st, 'error'):
                st.error(f"システム設定エラー: config.yamlの解析に失敗しました: {e}")
            _config_cache = {}
        except Exception as e: # FileNotFoundError もここでキャッチされる
            print(f"CRITICAL ERROR: An unexpected error occurred while loading config.yaml: {e}")
            if 'st' in globals() and hasattr(st, 'error'):
                st.error(f"システム設定エラー: config.yamlの読み込み中に予期せぬエラーが発生しました: {e}")
            _config_cache = {}
    return _config_cache if _config_cache is not None else {} # Noneチェックを追加

# --- 画像処理関連の設定取得 ---
# app.py 側で get_config() を直接呼び出し、必要なセクションを取得する方式に移行するため、
# この get_image_processing_config() 関数は廃止または簡略化を推奨。
# もし残す場合は、app.py のキー名との整合性を取る必要がある。
# 以下は、app.py が config.yaml の構造を直接解釈することを前提とし、この関数は使用しない方向での提案。
# どうしてもこの関数経由で取得したい場合は、app.py の固定パラメータキーに合わせて値を返すように修正する。

# 例: もし get_image_processing_config を残す場合の修正案 (app.pyのキー名に合わせる)
# def get_image_processing_defaults_for_app() -> Dict[str, Any]:
#     ...（関数本体を削除）...

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    ネストされたキーパス (例: "llm_models.text_model_name") を使って設定値を取得する。
    キーが存在しない場合はデフォルト値を返す。
    """
    config = get_config()
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                # print(f"Warning: Key path '{key_path}' not fully found in config. Intermediate key '{key}' does not lead to a dictionary.")
                return default
        return value
    except KeyError:
        # print(f"Warning: Key '{key_path}' not found in config. Returning default value: {default}")
        return default
    except TypeError:
        # print(f"Warning: Cannot access key in a non-dictionary for '{key_path}'. Returning default value: {default}")
        return default

# --- 特定の設定値を取得するためのヘルパー関数 (オプションだが便利) ---
def get_text_model_name(default_value: str = "gemini-1.5-flash-latest") -> str:
    return get_config_value("llm_models.text_model_name", default_value)

def get_vision_model_name(default_value: str = "gemini-1.5-flash-latest") -> str:
    return get_config_value("llm_models.vision_model_name", default_value)

def get_prompt_template_name(template_key: str) -> Optional[str]:
    # プロンプト名が設定されていない場合、Noneを返すことで呼び出し元でエラーハンドリングしやすくする
    return get_config_value(f"prompt_templates.{template_key}") # defaultはNoneのまま

# 以下は app.py から直接 config を読むことを推奨するため、コメントアウトまたは削除
# def get_pillow_max_image_pixels(default_value: int = 225000000) -> int:
#     return get_config_value("image_processing.pillow_max_image_pixels", default_value)

# def get_default_max_pixels_for_resizing(default_value: int = 4000000) -> int:
#     return get_config_value("image_processing.default_max_pixels_for_resizing", default_value)

# def get_default_output_format(default_value: str = "JPEG") -> str:
#     return get_config_value("image_processing.default_output_format", default_value)

# def get_default_jpeg_quality(default_value: int = 85) -> int:
#     return get_config_value("image_processing.default_jpeg_quality", default_value)

# def get_apply_grayscale(default_value: bool = True) -> bool:
#     return get_config_value("image_processing.apply_grayscale", default_value)

# def get_supported_mime_types(default_value: List[str] = ["image/png", "image/jpeg", "image/webp"]) -> List[str]:
#     return get_config_value("image_processing.supported_mime_types", default_value)

# def get_convertable_mime_types(default_value: Dict[str, str] = {"image/gif": "image/png", "image/bmp": "image/png"}) -> Dict[str, str]:
#     return get_config_value("image_processing.convertable_mime_types", default_value)


if __name__ == '__main__':
    # テスト用
    print("--- Testing get_config ---")
    loaded_config = get_config()
    if loaded_config:
        print("Config loaded successfully.")
        # print(f"Full config: {loaded_config}")
    else:
        print("Config loading failed or returned empty.")

    print("\n--- Testing get_config_value ---")
    print(f"Text Model (direct): {get_config_value('llm_models.text_model_name', 'default-text-model')}")
    print(f"Vision Model (direct): {get_config_value('llm_models.vision_model_name', 'default-vision-model')}")
    print(f"A.B.C (non-existent): {get_config_value('a.b.c', 'default_val_for_abc')}")
    
    print("\n--- Testing specific getters ---")
    print(f"Text Model (getter): {get_text_model_name()}")
    print(f"Vision Model (getter): {get_vision_model_name()}")
    clarification_prompt_name = get_prompt_template_name(PROMPT_KEY_CLARIFICATION_QUESTION)
    if clarification_prompt_name:
        print(f"Clarification Question Prompt File: {clarification_prompt_name}.md")
    else:
        print(f"Clarification Question Prompt File: Not configured (template_key: {PROMPT_KEY_CLARIFICATION_QUESTION})")

    # ★ 新しいプロンプトキーのテストを追加 ★
    extract_type_prompt_name = get_prompt_template_name(PROMPT_KEY_EXTRACT_TEXT_AND_TYPE_FROM_IMAGE)
    if extract_type_prompt_name:
        print(f"Extract Text and Type Prompt File: {extract_type_prompt_name}.md")
    else:
        print(f"Extract Text and Type Prompt File: Not configured (template_key: {PROMPT_KEY_EXTRACT_TEXT_AND_TYPE_FROM_IMAGE})")

    analyze_ocr_prompt_name = get_prompt_template_name(PROMPT_KEY_ANALYZE_QUERY_WITH_OCR)
    if analyze_ocr_prompt_name:
        print(f"Analyze Query with OCR Prompt File: {analyze_ocr_prompt_name}.md")
    else:
        print(f"Analyze Query with OCR Prompt File: Not configured (template_key: {PROMPT_KEY_ANALYZE_QUERY_WITH_OCR})")

    print("\n--- Testing direct access to image processing config from get_config() ---")
    main_config = get_config()
    img_proc_section = main_config.get("image_processing", {})
    print(f"Pillow Max Pixels (direct from config): {img_proc_section.get('pillow_max_image_pixels', 'Not Set')}")
    opencv_trim_section = img_proc_section.get("opencv_trimming", {})
    print(f"OpenCV Apply (direct from config): {opencv_trim_section.get('apply', 'Not Set')}")
    print(f"OpenCV Padding (direct from config): {opencv_trim_section.get('padding', 'Not Set')}")