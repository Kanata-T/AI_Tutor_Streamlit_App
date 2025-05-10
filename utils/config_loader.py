import yaml
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_FILE_PATH = Path(__file__).parent.parent / "config.yaml" # プロジェクトルートのconfig.yamlを指す

_config_cache: Optional[Dict[str, Any]] = None

def load_config() -> Dict[str, Any]:
    """
    config.yaml ファイルを読み込み、内容を辞書として返す。
    一度読み込んだ内容はキャッシュする。
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_FILE_PATH}")

    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            raise ValueError("Configuration file is not a valid YAML dictionary.")
        _config_cache = config_data
        return config_data
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading configuration: {e}")

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    ネストされたキーパス (例: "llm_models.text_model_name") を使って設定値を取得する。
    キーが存在しない場合はデフォルト値を返す。
    """
    config = load_config()
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
    print(f"Config loaded: {load_config()}")
    print(f"Text Model: {get_text_model_name()}")
    print(f"Vision Model: {get_vision_model_name()}")
    print(f"Initial Analysis Prompt File: {get_prompt_template_name('initial_analysis')}.md")
    print(f"Non Existent Key: {get_config_value('a.b.c', 'default_val')}")