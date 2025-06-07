#!/usr/bin/env python3
"""
Deepnote環境での初期化スクリプト
システム依存関係のインストールとOpenCV設定を行います
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """コマンドを実行し、結果を表示"""
    print(f"🔧 {description}")
    print(f"実行中: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ 成功: {description}")
        if result.stdout.strip():
            print(f"出力: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {description}")
        print(f"エラー内容: {e.stderr}")
        return False

def install_system_dependencies():
    """システム依存関係のインストール"""
    print("🚀 Deepnote環境の初期化を開始します...")
    
    # システムパッケージの更新
    commands = [
        ("apt-get update", "パッケージリストの更新"),
        ("apt-get install -y libgl1-mesa-glx", "OpenGL ライブラリのインストール"),
        ("apt-get install -y libglib2.0-0", "GLib ライブラリのインストール"),
        ("apt-get install -y libsm6", "Session Management ライブラリのインストール"),
        ("apt-get install -y libxext6", "X11 Extension ライブラリのインストール"),
        ("apt-get install -y libxrender-dev", "X Render Extension ライブラリのインストール"),
        ("apt-get install -y libgomp1", "OpenMP ライブラリのインストール"),
        ("apt-get install -y tesseract-ocr", "Tesseract OCR のインストール"),
        ("apt-get install -y tesseract-ocr-jpn", "Tesseract 日本語パックのインストール"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(f"sudo {command}", description):
            success_count += 1
    
    print(f"\n📊 システム依存関係のインストール結果: {success_count}/{len(commands)} 成功")
    return success_count == len(commands)

def install_python_dependencies():
    """Python依存関係のインストール"""
    print("\n📦 Python依存関係のインストール...")
    
    # 既存のopencv-pythonをアンインストール
    run_command("pip uninstall -y opencv-python opencv-contrib-python", "既存OpenCVのアンインストール")
    
    # requirements.txtからインストール
    if run_command("pip install -r requirements.txt", "Python依存関係のインストール"):
        print("✅ Python依存関係のインストール完了")
        return True
    else:
        print("❌ Python依存関係のインストールに失敗")
        return False

def verify_installation():
    """インストールの確認"""
    print("\n🔍 インストールの確認...")
    
    try:
        import cv2
        print(f"✅ OpenCV バージョン: {cv2.__version__}")
        
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract バージョン: {version}")
        
        import streamlit
        print(f"✅ Streamlit バージョン: {streamlit.__version__}")
        
        import google.generativeai
        print("✅ Google Generative AI インポート成功")
        
        return True
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False

def create_env_file():
    """環境変数ファイルの作成"""
    env_path = ".env"
    if not os.path.exists(env_path):
        print(f"\n📝 {env_path} ファイルを作成します...")
        env_content = """# AI Tutor Streamlit App - 環境変数設定ファイル
# Deepnote環境用

# Gemini API Key (必須)
GEMINI_API_KEY="your_gemini_api_key_here"

# OpenCV設定
OPENCV_LOG_LEVEL=ERROR

# Tesseract設定
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
"""
        try:
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print(f"✅ {env_path} ファイルを作成しました")
            print("⚠️ GEMINI_API_KEY を実際のAPIキーに変更してください")
        except Exception as e:
            print(f"❌ {env_path} ファイルの作成に失敗: {e}")
    else:
        print(f"✅ {env_path} ファイルは既に存在します")

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("🚀 AI Tutor Streamlit App - Deepnote環境初期化")
    print("=" * 60)
    
    # システム依存関係のインストール
    if not install_system_dependencies():
        print("⚠️ 一部のシステム依存関係のインストールに失敗しましたが、続行します")
    
    # Python依存関係のインストール
    if not install_python_dependencies():
        print("❌ Python依存関係のインストールに失敗しました")
        sys.exit(1)
    
    # インストールの確認
    if verify_installation():
        print("\n🎉 すべてのインストールが成功しました！")
    else:
        print("\n⚠️ 一部のコンポーネントに問題があります")
    
    # 環境変数ファイルの作成
    create_env_file()
    
    print("\n" + "=" * 60)
    print("✅ 初期化完了！")
    print("📝 次のステップ:")
    print("   1. .env ファイルでGEMINI_API_KEYを設定")
    print("   2. streamlit run app_deepnote.py でアプリを起動")
    print("=" * 60)

if __name__ == "__main__":
    main() 