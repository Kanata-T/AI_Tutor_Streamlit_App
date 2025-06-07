# deepnote_setup.py
# Deepnote用の環境設定ファイル

import os
import streamlit as st

def setup_deepnote_environment():
    """Deepnote環境での設定を行う"""
    
    # Streamlit設定
    if 'setup_done' not in st.session_state:
        # 初回のみ実行
        st.session_state.setup_done = True
        
        # 必要に応じて環境変数を設定
        # 注意: 実際のAPIキーは Deepnoteの環境変数で設定してください
        
        print("Deepnote environment setup completed")
        
    return True

def load_environment_variables():
    """環境変数をロード"""
    # Deepnoteの場合、環境変数は Project Settings > Environment variables で設定
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        st.error("⚠️ GEMINI_API_KEY が設定されていません。")
        st.info("Deepnoteの Project Settings > Environment variables で GEMINI_API_KEY を設定してください。")
        return False
        
    return True 