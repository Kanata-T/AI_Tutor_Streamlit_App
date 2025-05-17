from typing import List, Dict, Any, Optional, Literal, TypedDict

class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

class UploadedFileData(TypedDict):
    mime_type: str
    data: bytes

# LLMからの初期分析結果の型 (例)
class InitialAnalysisResult(TypedDict, total=False):
    request_category: str
    topic: str
    summary: str
    ambiguity: Literal["clear", "ambiguous"]
    reason_for_ambiguity: Optional[str]
    # ocr_text_from_extraction: Optional[str] # tutor_logicで追加されることがある
    # error: Optional[str] # エラー時に含まれることがある

# LLMからの明確化応答分析結果の型 (例)
class ClarificationAnalysisResult(TypedDict, total=False):
    resolved: bool
    clarified_request: Optional[str]
    remaining_issue: Optional[str]
    ai_response_to_user: Optional[str]
    # error: Optional[str] 