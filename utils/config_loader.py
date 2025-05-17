import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st  # エラー表示用 (任意)

# Prompt Template Keys (これをgemini_service.py等から参照する)
PROMPT_KEY_SYSTEM: str = "system_prompt"
PROMPT_KEY_EXTRACT_TEXT_FROM_IMAGE: str = "extract_text_from_image"
PROMPT_KEY_ANALYZE_QUERY_WITH_OCR: str = "analyze_query_with_ocr"
PROMPT_KEY_ANALYZE_CLARIFICATION_RESPONSE: str = "analyze_clarification_response"
PROMPT_KEY_CLARIFICATION_QUESTION: str = "clarification_question"
PROMPT_KEY_GENERATE_EXPLANATION: str = "generate_explanation"
PROMPT_KEY_GENERATE_FOLLOWUP: str = "generate_followup"
PROMPT_KEY_GENERATE_SUMMARY: str = "generate_summary"
PROMPT_KEY_ANALYZE_STUDENT_PERFORMANCE: str = "analyze_student_performance"

CONFIG_FILE_PATH = Path(__file__).parent.parent / "config.yaml" # プロジェクトルートのconfig.yamlを指す
CONFIG_PATH = CONFIG_FILE_PATH  # 互換性のため

_config_cache: Optional[Dict[str, Any]] = None

def get_config() -> dict:
    """
    設定ファイルを読み込み、キャッシュする。
    エラー時はstreamlitで表示し、空辞書を返す。
    """
    global _config_cache
    if _config_cache is None:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f)
            if not isinstance(_config_cache, dict):
                print(f"CRITICAL ERROR: config.yaml is not a valid YAML dictionary.")
                st.error("システム設定エラー: config.yamlの形式が正しくありません。")
                _config_cache = {}  # エラー時は空辞書を返す
        except FileNotFoundError:
            print(f"CRITICAL ERROR: config.yaml not found at {CONFIG_PATH}")
            st.error(f"システム設定エラー: config.yamlが見つかりません ({CONFIG_PATH})。")
            _config_cache = {}
        except yaml.YAMLError as e:
            print(f"CRITICAL ERROR: Error parsing config.yaml: {e}")
            st.error(f"システム設定エラー: config.yamlの解析に失敗しました: {e}")
            _config_cache = {}
        except Exception as e:
            print(f"CRITICAL ERROR: An unexpected error occurred while loading config.yaml: {e}")
            st.error(f"システム設定エラー: config.yamlの読み込み中に予期せぬエラーが発生しました: {e}")
            _config_cache = {}
    return _config_cache

# --- 画像処理関連の設定取得 ---
def get_image_processing_config() -> dict:
    """
    画像処理関連の設定を取得する。
    Returns:
        dict: 画像処理設定の辞書。キーが存在しない場合はデフォルト値を返す。
    """
    config = get_config()
    img_proc_conf = config.get("image_processing", {})
    if not isinstance(img_proc_conf, dict):
        print(f"Warning: 'image_processing' section in config.yaml is not a valid dictionary. Using defaults.")
        return {
            "pillow_max_image_pixels": 225000000,
            "default_max_pixels_for_resizing": 4000000,
            "default_output_format": "JPEG",
            "default_jpeg_quality": 85,
            "supported_mime_types": ["image/png", "image/jpeg", "image/webp"],
            "convertable_mime_types": {"image/gif": "image/png", "image/bmp": "image/png"}
        }
    return {
        "pillow_max_image_pixels": img_proc_conf.get("pillow_max_image_pixels", 225000000),
        "default_max_pixels_for_resizing": img_proc_conf.get("default_max_pixels_for_resizing", 4000000),
        "default_output_format": img_proc_conf.get("default_output_format", "JPEG"),
        "default_jpeg_quality": img_proc_conf.get("default_jpeg_quality", 85),
        "supported_mime_types": img_proc_conf.get("supported_mime_types", ["image/png", "image/jpeg", "image/webp"]),
        "convertable_mime_types": img_proc_conf.get("convertable_mime_types", {"image/gif": "image/png", "image/bmp": "image/png"})
    }

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
            else: # キーの途中で辞書でなくなった場合
                print(f"Warning: Key path '{key_path}' not fully found in config. Intermediate key '{key}' does not lead to a dictionary.")
                return default
        return value
    except KeyError:
        print(f"Warning: Key '{key_path}' not found in config. Returning default value: {default}")
        return default
    except TypeError: # valueがNoneなどでインデックスアクセスできない場合
        print(f"Warning: Cannot access key in a non-dictionary for '{key_path}'. Returning default value: {default}")
        return default

# --- 特定の設定値を取得するためのヘルパー関数 (オプションだが便利) ---
def get_text_model_name() -> str:
    return get_config_value("llm_models.text_model_name", "gemini-1.5-flash-latest")

def get_vision_model_name() -> str:
    return get_config_value("llm_models.vision_model_name", "gemini-1.5-flash-latest")

def get_prompt_template_name(template_key: str) -> Optional[str]:
    return get_config_value(f"prompt_templates.{template_key}")

if __name__ == '__main__':
    # テスト用
    print(f"Config loaded: {get_config()}")
    print(f"Text Model: {get_text_model_name()}")
    print(f"Vision Model: {get_vision_model_name()}")
    print(f"Initial Analysis Prompt File: {get_prompt_template_name('initial_analysis')}.md")
    print(f"Non Existent Key: {get_config_value('a.b.c', 'default_val')}")