from typing import List, Dict, Any, Optional, Literal, TypedDict
from enum import Enum

class ImageType(Enum):
    """
    画像種別を表す列挙型。
    PROBLEM: 問題文
    EXPLANATION: 解説
    ANSWER: 解答
    OTHER: その他（判別不能または該当なし）
    """
    PROBLEM = "問題文"
    EXPLANATION = "解説"
    ANSWER = "解答"
    OTHER = "その他"  # LLMが判別できなかった場合や、いずれにも当てはまらない場合

    def __str__(self):
        return self.value

class ChatMessage(TypedDict):
    """
    チャットメッセージの型定義。

    Attributes:
        role (Literal): メッセージの役割（user, assistant, system）
        content (str): メッセージ本文
    """
    role: Literal["user", "assistant", "system"]
    content: str

class UploadedFileData(TypedDict):
    """
    アップロードされたファイルの情報。

    Attributes:
        mime_type (str): MIMEタイプ
        data (bytes): ファイルデータ本体
        filename (str): 元のファイル名
    """
    mime_type: str
    data: bytes
    filename: str

class ProcessedImageInfo(TypedDict):
    """
    LLMによる画像種別判別後の情報。

    Attributes:
        original_filename (str): 元のファイル名
        image_type (ImageType): LLMが判別した画像種別
        mime_type (str): 処理後のMIMEタイプ
        input_bytes (bytes): Vision/OCRモデルに渡す画像データ
        ocr_text (Optional[str]): OCRで抽出されたテキスト
    """
    original_filename: str
    image_type: ImageType
    mime_type: str
    input_bytes: bytes # このバイトデータは前処理（トリミング等）後のもの
    ocr_text: Optional[str]

# LLMからの初期分析結果の型 (例)
class InitialAnalysisResult(TypedDict, total=False):
    """
    LLMからの初期分析結果の型。

    Attributes:
        request_category (str): リクエストカテゴリ
        topic (str): トピック
        summary (str): 要約
        ambiguity (Literal): 明確さ（clear/ambiguous）
        reason_for_ambiguity (Optional[str]): 曖昧な理由
    """
    request_category: str
    topic: str
    summary: str
    ambiguity: Literal["clear", "ambiguous"]
    reason_for_ambiguity: Optional[str]
    # ocr_text_from_extraction: Optional[str] # tutor_logicで追加されることがある
    # error: Optional[str] # エラー時に含まれることがある

# LLMからの明確化応答分析結果の型 (例)
class ClarificationAnalysisResult(TypedDict, total=False):
    """
    LLMからの明確化応答分析結果の型。

    Attributes:
        resolved (bool): 問題が解決したかどうか
        clarified_request (Optional[str]): 明確化されたリクエスト
        remaining_issue (Optional[str]): 残る課題
        ai_response_to_user (Optional[str]): ユーザーへのAI応答
    """
    resolved: bool
    clarified_request: Optional[str]
    remaining_issue: Optional[str]
    ai_response_to_user: Optional[str]
    # error: Optional[str]

# ★新規追加★
class ProblemContext(TypedDict):
    """
    現在のセッションで扱っている問題文のコンテキスト情報。

    Attributes:
        initial_query (str): ユーザーが最初に入力した質問テキスト。
        problem_images (List[ProcessedImageInfo]): 問題文と判断された画像のリスト。
                                                  各画像の情報はProcessedImageInfo型。
    """
    initial_query: str
    problem_images: List[ProcessedImageInfo]