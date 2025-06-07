# 🚀 Deepnote ネイティブ Streamlit サポート使用ガイド

Deepnoteの**ネイティブ Streamlit サポート**を使用してAI学習チューターアプリを実行する方法です。

## 🎯 ネイティブサポートの利点

- ✅ **自動デプロイ**: `.py`ファイルをドロップするだけで自動デプロイ
- ✅ **リアルタイム更新**: コード変更が即座に反映
- ✅ **共有機能**: URLで簡単に共有可能
- ✅ **統合環境**: Deepnoteの他機能との連携
- ✅ **パフォーマンス**: 最適化された実行環境

## 📋 セットアップ手順

### 1. プロジェクトの準備

#### 方法A: GitHubからクローン（推奨）
```bash
# Deepnoteのターミナルで実行
git clone https://github.com/Kanata-T/AI_Tutor_Streamlit_App.git
cd AI_Tutor_Streamlit_App
```

#### 方法B: ファイル個別アップロード
必要なファイルをDeepnoteにアップロード：
- `app_deepnote.py` （メインアプリ）
- `requirements.txt`
- `config.yaml`
- `prompts/` フォルダ
- `core/`, `services/`, `ui/`, `utils/` フォルダ

### 2. 依存関係のインストール

```bash
# Deepnoteのターミナルで実行
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env` ファイルを作成：
```bash
# .env ファイルの内容
GEMINI_API_KEY="your_api_key_here"
```

### 4. Streamlitアプリのデプロイ

#### 方法A: 自動検出（推奨）
1. `app_deepnote.py` をプロジェクトのルートに配置
2. Deepnoteが自動的にStreamlitアプリを検出
3. 「Create Streamlit app」ボタンが表示される
4. ボタンをクリックしてデプロイ

#### 方法B: 手動デプロイ
1. ファイルマネージャーで `app_deepnote.py` を右クリック
2. 「Create Streamlit app」を選択
3. アプリが自動的にデプロイされる

## 🎮 使用方法

### アプリの起動
1. デプロイ後、「Open app」ボタンをクリック
2. 新しいタブでStreamlitアプリが開く
3. サイドバーでモードを選択
4. AIチューターまたは画像処理チューニングを使用

### アプリの状態
- **Live**: アプリが実行中（緑色）
- **Sleeping**: 非アクティブ状態（黄色）
- **Deploying**: 更新中（青色）

### コードの更新
- ファイルを編集すると自動的にアプリに反映
- リアルタイム更新を無効にする場合：Settings > Run on save をオフ

## 🔧 設定とカスタマイズ

### アプリ設定
Streamlitアプリの設定（ハンバーガーメニュー > Settings）：
- **Run on save**: 自動更新の有効/無効
- **Wide mode**: ワイドレイアウト
- **Theme**: ライト/ダークテーマ

### 共有設定
1. アプリ内で Settings > Copy link
2. プロジェクトの共有設定を確認：
   - Share > Link sharing を 'View' に設定
   - 外部ユーザーがアプリを閲覧可能

## 📁 ファイル構造

```
Deepnoteプロジェクト/
├── app_deepnote.py          # メインアプリ（Streamlit用）
├── .env                     # 環境変数
├── requirements.txt         # 依存関係
├── config.yaml             # 設定ファイル
├── prompts/                # プロンプトテンプレート
│   ├── analyze_request.md
│   ├── clarify_request.md
│   ├── plan_guidance.md
│   └── generate_explanation.md
├── core/                   # コアロジック
│   ├── tutor_logic.py
│   └── state_manager.py
├── services/               # サービス層
│   ├── gemini_service.py
│   └── image_processing_service.py
├── ui/                     # UIコンポーネント
│   ├── tutor_mode_ui.py
│   ├── tuning_mode_ui.py
│   └── display_helpers.py
└── utils/                  # ユーティリティ
    ├── config_loader.py
    └── image_utils.py
```

## 🔄 開発ワークフロー

### 1. ローカル開発
```bash
# ローカルでの開発・テスト
uv sync                    # 依存関係インストール
streamlit run app_deepnote.py  # ローカル実行
```

### 2. Deepnoteデプロイ
```bash
# 変更をGitHubにプッシュ
git add .
git commit -m "Update app"
git push origin main

# Deepnoteで最新版を取得
git pull origin main
```

### 3. 自動更新
- ファイル変更時に自動的にアプリが更新
- 即座に変更が反映される

## 🛠️ トラブルシューティング

### よくある問題

| 問題 | 解決方法 |
|------|----------|
| アプリが検出されない | ファイル名を確認、`.py`拡張子が必要 |
| 依存関係エラー | `pip install -r requirements.txt` を実行 |
| API キーエラー | `.env` ファイルの内容を確認 |
| アプリが起動しない | ログを確認、構文エラーをチェック |

### デバッグ方法
1. **ログの確認**: アプリ下部のログセクション
2. **ターミナル実行**: `streamlit run app_deepnote.py`
3. **段階的テスト**: 機能を一つずつ有効化

### パフォーマンス最適化
- **キャッシュ活用**: `@st.cache_data` デコレータ使用
- **セッション管理**: 不要なデータの削除
- **画像最適化**: サイズと品質の調整

## 🚀 高度な機能

### データ統合
```python
# Deepnoteの統合機能を使用
import deepnote_toolkit
deepnote_toolkit.set_integration_env()

# S3, Google Drive, データベースとの連携
```

### 自動化
- **スケジュール実行**: Notebookでデータ準備
- **データパイプライン**: 定期的なデータ更新
- **通知機能**: 処理完了の通知

### 監視
- **アクセス解析**: アプリの使用状況
- **エラー監視**: 自動エラー検出
- **パフォーマンス**: 応答時間の測定

---

## 📞 サポート

### 公式リソース
- [Deepnote Streamlit ドキュメント](https://docs.deepnote.com/features/streamlit-apps)
- [Streamlit 公式ドキュメント](https://docs.streamlit.io/)

### コミュニティ
- Deepnote Community
- Streamlit Community Forum

問題が発生した場合は、まずログを確認し、必要に応じてプロジェクトを再起動してください。 