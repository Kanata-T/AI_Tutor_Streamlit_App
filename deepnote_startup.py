#!/usr/bin/env python3
"""
Deepnote起動時に最新のコードを自動取得するスクリプト
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """コマンドを実行し、結果を返す"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ 成功: {command}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {command}")
        print(f"エラー内容: {e.stderr}")
        return False

def setup_fresh_environment():
    """最新のコードで環境をセットアップ"""
    
    print("🚀 AI Tutor Streamlit App - 起動時セットアップ開始")
    print("=" * 50)
    
    # 作業ディレクトリの設定
    work_dir = "/work"
    repo_url = "https://github.com/Kanata-T/AI_Tutor_Streamlit_App.git"
    project_dir = f"{work_dir}/ai-tutor-app"
    
    # 既存のプロジェクトディレクトリを削除（完全にクリーンな状態にする）
    if os.path.exists(project_dir):
        print(f"📁 既存のプロジェクトディレクトリを削除: {project_dir}")
        run_command(f"rm -rf {project_dir}")
    
    # 最新のコードをクローン
    print(f"📥 最新のコードをクローン中...")
    if not run_command(f"git clone {repo_url} {project_dir}"):
        print("❌ クローンに失敗しました")
        return False
    
    # プロジェクトディレクトリに移動
    os.chdir(project_dir)
    print(f"📂 作業ディレクトリを変更: {project_dir}")
    
    # 依存関係のインストール
    print("📦 依存関係をインストール中...")
    if not run_command("pip install -r requirements.txt"):
        print("⚠️ 依存関係のインストールに失敗しました")
    
    # 環境変数の確認
    print("🔑 環境変数の確認...")
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print("✅ GEMINI_API_KEY が設定されています")
    else:
        print("⚠️ GEMINI_API_KEY が設定されていません")
        print("   Deepnoteの Settings > Environment variables で設定してください")
    
    print("=" * 50)
    print("🎉 セットアップ完了！")
    print(f"📍 プロジェクトディレクトリ: {project_dir}")
    print("🚀 Streamlitアプリを起動するには:")
    print("   streamlit run app.py")
    print("=" * 50)
    
    return True

def quick_update():
    """既存のプロジェクトを最新に更新（高速版）"""
    
    print("🔄 既存プロジェクトを最新に更新中...")
    
    # Gitの状態をリセット
    run_command("git reset --hard HEAD")
    run_command("git clean -fd")
    
    # 最新のコードを取得
    if run_command("git pull origin main"):
        print("✅ 最新のコードに更新されました")
        
        # 依存関係の更新確認
        if run_command("pip install -r requirements.txt --upgrade"):
            print("✅ 依存関係も更新されました")
        
        return True
    else:
        print("❌ 更新に失敗しました。完全セットアップを実行します...")
        return False

if __name__ == "__main__":
    # コマンドライン引数の処理
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # 既存プロジェクトの高速更新
        project_dir = "/work/ai-tutor-app"
        if os.path.exists(project_dir):
            os.chdir(project_dir)
            if not quick_update():
                setup_fresh_environment()
        else:
            setup_fresh_environment()
    else:
        # 完全な新規セットアップ
        setup_fresh_environment() 