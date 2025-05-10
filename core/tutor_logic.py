# core/tutor_logic.py
import streamlit as st # st.session_state にアクセスするために必要
from typing import Dict, Any, Optional
from services import gemini_service # services/__init__.py 経由でインポート
from . import state_manager # 同じディレクトリの state_manager をインポート

def perform_initial_analysis_logic() -> Optional[Dict[str, Any]]:
    """
    ユーザーの入力に基づいて初期分析を実行し、結果を返す。
    st.session_state から必要な情報を取得し、gemini_service を呼び出す。
    """
    query_text = st.session_state.get("user_query_text", "")
    image_data = st.session_state.get("uploaded_file_data", None) # mime_type, data を含む辞書
    # selected_topic = st.session_state.get("selected_topic", "") # LLMに渡す情報として含めても良い

    if not query_text and not image_data:
        # st.error("分析するテキストまたは画像がありません。") # UI側で表示するのでここでは不要かも
        print("Error in tutor_logic: No query text or image data for analysis.")
        return {"error": "入力がありません。"}


    # gemini_service の analyze_initial_input を呼び出す
    # この関数はJSONレスポンス(辞書)またはエラー情報(辞書)またはNoneを返す想定
    print(f"Tutor Logic: Calling Gemini for initial analysis. Query: '{query_text[:50]}...', Image: {'Yes' if image_data else 'No'}")
    analysis_result = gemini_service.analyze_initial_input(
        query_text=query_text,
        image_data=image_data
    )
    print(f"Tutor Logic: Received analysis result: {analysis_result}")


    if analysis_result is None:
        # API呼び出し自体が失敗した場合など
        return {"error": "AIによる分析に失敗しました。時間をおいて再度お試しください。"}
    
    if "error" in analysis_result:
        # APIは成功したが、レスポンスのパースエラーなど、サービス層でエラーがセットされた場合
        return analysis_result # エラー情報をそのまま返す

    # 成功した場合、結果を返す (この辞書が state_manager.store_initial_analysis_result に渡される)
    return analysis_result

# --- 今後追加するロジック関数 ---
# def generate_clarification_question_logic() -> Optional[str]:
#     # ...
#     pass

# def analyze_user_clarification_logic(user_response: str) -> Optional[Dict[str, Any]]:
#     # ...
#     pass

# def generate_explanation_logic() -> Optional[str]:
#     # ...
#     pass