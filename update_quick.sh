#!/bin/bash

# AI Tutor Streamlit App - 高速更新スクリプト

echo "🔄 AI Tutor Streamlit App - 高速更新"
echo "===================================="

PROJECT_DIR="/work/ai-tutor-app"

# プロジェクトディレクトリの存在確認
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ プロジェクトディレクトリが見つかりません: $PROJECT_DIR"
    echo "🚀 完全セットアップを実行してください: ./start_fresh.sh"
    exit 1
fi

# プロジェクトディレクトリに移動
cd "$PROJECT_DIR"
echo "📂 作業ディレクトリ: $(pwd)"

# Gitの状態をリセット（ローカルの変更を破棄）
echo "🔄 ローカルの変更をリセット中..."
git reset --hard HEAD
git clean -fd

# 最新のコードを取得
echo "📥 最新のコードを取得中..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ 最新のコードに更新されました"
    
    # 依存関係の更新
    echo "📦 依存関係を更新中..."
    pip install -r requirements.txt --upgrade
    
    echo "===================================="
    echo "🎉 高速更新完了！"
    echo "🚀 Streamlitアプリを起動するには:"
    echo "   streamlit run app.py"
    echo "===================================="
else
    echo "❌ 更新に失敗しました"
    echo "🚀 完全セットアップを実行してください: ./start_fresh.sh"
    exit 1
fi 