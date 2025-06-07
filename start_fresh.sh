#!/bin/bash

# AI Tutor Streamlit App - 最新コード取得・起動スクリプト

echo "🚀 AI Tutor Streamlit App - 最新版セットアップ"
echo "================================================"

# 作業ディレクトリの設定
WORK_DIR="/work"
REPO_URL="https://github.com/Kanata-T/AI_Tutor_Streamlit_App.git"
PROJECT_DIR="$WORK_DIR/ai-tutor-app"

# 既存のプロジェクトディレクトリを削除
if [ -d "$PROJECT_DIR" ]; then
    echo "📁 既存のプロジェクトディレクトリを削除中..."
    rm -rf "$PROJECT_DIR"
fi

# 最新のコードをクローン
echo "📥 最新のコードをクローン中..."
git clone "$REPO_URL" "$PROJECT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ クローン成功"
else
    echo "❌ クローンに失敗しました"
    exit 1
fi

# プロジェクトディレクトリに移動
cd "$PROJECT_DIR"
echo "📂 作業ディレクトリ: $(pwd)"

# 依存関係のインストール
echo "📦 依存関係をインストール中..."
pip install -r requirements.txt

# 環境変数の確認
echo "🔑 環境変数の確認..."
if [ -n "$GEMINI_API_KEY" ]; then
    echo "✅ GEMINI_API_KEY が設定されています"
else
    echo "⚠️ GEMINI_API_KEY が設定されていません"
    echo "   Deepnoteの Settings > Environment variables で設定してください"
fi

echo "================================================"
echo "🎉 セットアップ完了！"
echo "📍 プロジェクトディレクトリ: $PROJECT_DIR"
echo ""
echo "🚀 Streamlitアプリを起動するには:"
echo "   streamlit run app.py"
echo ""
echo "🔄 次回、高速更新するには:"
echo "   ./update_quick.sh"
echo "================================================" 