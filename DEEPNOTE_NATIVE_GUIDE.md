# 🚀 Deepnote ネイティブ Streamlit サポート使用ガイド

Deepnoteの**ネイティブ Streamlit サポート**を使用してAI学習チューターアプリを実行する方法です。

## ⚠️ 重要：OpenCVエラーの解決

Deepnote環境では、OpenCVが必要とするシステムライブラリが不足している場合があります。以下の手順で解決してください。

## 📋 セットアップ手順（更新版）

### 1. プロジェクトの準備

```bash
# Deepnoteのターミナルで実行
git clone https://github.com/Kanata-T/AI_Tutor_Streamlit_App.git
cd AI_Tutor_Streamlit_App
```

### 2. 🔧 初期化スクリプトの実行（重要）

```bash
# システム依存関係とPython依存関係を自動インストール
python init_deepnote.py
```

このスクリプトが以下を自動実行します：
- ✅ システムライブラリのインストール（libGL.so.1など）
- ✅ OpenCV-headless版のインストール
- ✅ Tesseract OCRのインストール
- ✅ 環境変数ファイルの作成

### 3. 環境変数の設定

`.env` ファイルを編集：
```bash
# .env ファイルの内容
GEMINI_API_KEY="AIzaSyBNo7LYXjANwg2uEUafsCox8wmEcSEh5AI"
```

### 4. Streamlitアプリのデプロイ

1. `app_deepnote.py` をプロジェクトのルートに配置
2. Deepnoteが自動的にStreamlitアプリを検出
3. 「Create Streamlit app」ボタンをクリック
4. アプリが自動起動

## 🛠️ トラブルシューティング

### OpenCVエラーの解決

#### エラー例：
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

#### 解決方法：
```bash
# 1. 初期化スクリプトを実行
python init_deepnote.py

# 2. 手動でシステムライブラリをインストール（必要に応じて）
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# 3. OpenCV-headless版を再インストール
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless>=4.11.0.86
```

### よくある問題と解決方法

| 問題 | 解決方法 |
|------|----------|
| **libGL.so.1エラー** | `python init_deepnote.py` を実行 |
| **Tesseract not found** | `sudo apt-get install tesseract-ocr tesseract-ocr-jpn` |
| **API キーエラー** | `.env` ファイルの `GEMINI_API_KEY` を確認 |
| **モジュールインポートエラー** | 依存関係を再インストール |

### 段階的デバッグ

1. **環境確認**：
```bash
python -c "import cv2; print(cv2.__version__)"
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

2. **アプリ起動テスト**：
```bash
streamlit run app_deepnote.py
```

3. **ログ確認**：
   - Streamlitアプリ下部のログセクション
   - ターミナルでのエラーメッセージ

## 🎯 ネイティブサポートの利点

- ✅ **自動デプロイ**: `.py`ファイルをドロップするだけで自動デプロイ
- ✅ **リアルタイム更新**: コード変更が即座に反映
- ✅ **共有機能**: URLで簡単に共有可能
- ✅ **統合環境**: Deepnoteの他機能との連携
- ✅ **エラー処理**: 環境問題の自動検出と対処法表示

## 🎮 使用方法

### アプリの起動
1. デプロイ後、「Open app」ボタンをクリック
2. 新しいタブでStreamlitアプリが開く
3. 環境チェックが自動実行される
4. 問題があれば解決方法が表示される

### 環境状態の確認
アプリのサイドバーで以下を確認：
- ✅ Deepnote環境で実行中
- ✅ OpenCV 利用可能
- ❌ OpenCV 利用不可（要修復）

### モード選択
- **AIチューター**: 質問応答と学習支援
- **画像処理チューニング**: OpenCV必須（利用不可時はエラー表示）

## 📁 ファイル構造（更新版）

```
Deepnoteプロジェクト/
├── app_deepnote.py          # メインアプリ（エラー処理強化版）
├── init_deepnote.py         # 初期化スクリプト（新規）
├── .env                     # 環境変数
├── requirements.txt         # 依存関係（opencv-python-headless使用）
├── config.yaml             # 設定ファイル
├── prompts/                # プロンプトテンプレート
├── core/                   # コアロジック
├── services/               # サービス層
├── ui/                     # UIコンポーネント
└── utils/                  # ユーティリティ
```

## 🔄 開発ワークフロー（更新版）

### 1. 初回セットアップ
```bash
git clone https://github.com/Kanata-T/AI_Tutor_Streamlit_App.git
cd AI_Tutor_Streamlit_App
python init_deepnote.py  # 重要：初期化スクリプト実行
# .env ファイルでAPIキーを設定
```

### 2. アプリデプロイ
- `app_deepnote.py` を配置
- 「Create Streamlit app」をクリック
- 環境チェックが自動実行

### 3. 継続的開発
```bash
# コード変更時は自動更新
# 環境に問題があれば自動検出・対処法表示
```

## 🚀 高度な機能

### 自動環境修復
- OpenCVエラーの自動検出
- 解決方法の自動表示
- 段階的な問題解決ガイド

### 安全なモジュールインポート
- 依存関係の段階的確認
- エラー時の詳細情報表示
- 続行/停止の選択肢提供

### 環境適応型UI
- 利用可能機能の動的表示
- 環境制限時の代替案提示
- リアルタイム状態監視

## 📊 パフォーマンス

- **初期セットアップ**: 約3-5分（初期化スクリプト含む）
- **アプリ起動**: 約10-15秒
- **環境チェック**: 約2-3秒
- **エラー回復**: 約1-2分

---

## 📞 サポート

### 緊急時の対処

1. **完全リセット**：
```bash
rm -rf AI_Tutor_Streamlit_App
git clone https://github.com/Kanata-T/AI_Tutor_Streamlit_App.git
cd AI_Tutor_Streamlit_App
python init_deepnote.py
```

2. **手動修復**：
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
pip install --force-reinstall opencv-python-headless
```

### 公式リソース
- [Deepnote Streamlit ドキュメント](https://docs.deepnote.com/features/streamlit-apps)
- [OpenCV-Python ドキュメント](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

問題が解決しない場合は、プロジェクトを削除して完全に再セットアップしてください。 