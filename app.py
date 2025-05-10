# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image # Pillowをインポート

# coreモジュールのインポート (まとめてインポートできるように __init__.py を利用)
from core import state_manager, tutor_logic # tutor_logic をインポート

# --- 初期設定 ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini APIキーが設定されていません。.envファイルを確認してください。")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Gemini APIの設定に失敗しました: {e}")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="AI学習チューター", layout="wide")
st.title("AI学習チューター プロトタイプ")

# セッション状態の初期化
state_manager.initialize_session_state()

current_step = state_manager.get_current_step()

if current_step == state_manager.STEP_INPUT_SUBMISSION:
    st.header("1. 質問を入力してください")
    with st.form("submission_form"):
        user_query_text_input = st.text_input(
            "質問内容を入力してください",
            value=st.session_state.user_query_text
        )
        uploaded_file_obj = st.file_uploader(
            "画像をアップロード (任意)",
            type=["png", "jpg", "jpeg"]
        )
        selected_topic_input = st.selectbox(
            "トピックを選択 (任意)",
            ["", "文法", "語彙", "長文読解", "英作文", "その他"],
            index=["", "文法", "語彙", "長文読解", "英作文", "その他"].index(st.session_state.selected_topic) if st.session_state.selected_topic else 0
        )
        submitted = st.form_submit_button("質問を送信")

        if submitted:
            if not user_query_text_input and not uploaded_file_obj:
                st.warning("質問内容のテキスト入力または画像のアップロードのいずれかを行ってください。")
            else:
                image_data_for_state = None
                if uploaded_file_obj is not None:
                    try:
                        image_data_for_state = {
                            "mime_type": uploaded_file_obj.type,
                            "data": uploaded_file_obj.getvalue()
                        }
                        # st.success("画像を読み込みました。") # 送信ボタン押下後のメッセージは集約
                    except Exception as e:
                        st.error(f"画像の読み込みに失敗しました: {e}")
                        image_data_for_state = None
                
                state_manager.store_user_input(
                    user_query_text_input,
                    image_data_for_state,
                    selected_topic_input
                )
                
                user_message_content = f"質問: {user_query_text_input}"
                if image_data_for_state:
                    user_message_content += " (画像あり)"
                state_manager.add_message("user", user_message_content)
                
                state_manager.set_current_step(state_manager.STEP_INITIAL_ANALYSIS)
                st.rerun()

elif current_step == state_manager.STEP_INITIAL_ANALYSIS:
    st.header("2. 初期分析中です...")
    # (デバッグ用) 保存されたユーザー情報を表示
    # st.write("ユーザーの質問 (セッションステートより):")
    # st.write(f"- テキスト: {st.session_state.user_query_text}")
    # st.write(f"- トピック: {st.session_state.selected_topic}")
    # if st.session_state.uploaded_file_data:
    #     st.write("- アップロードされた画像:")
    #     st.image(st.session_state.uploaded_file_data["data"])

    # 初期分析処理がまだ実行されていない場合のみ実行する (st.rerun対策)
    if st.session_state.initial_analysis_result is None and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True) # 処理中フラグを立てる
        with st.spinner("AIが分析中です... しばらくお待ちください。"):
            # tutor_logic の関数を呼び出す
            analysis_result = tutor_logic.perform_initial_analysis_logic()
        state_manager.set_processing_status(False) # 処理中フラグを下ろす

        if analysis_result:
            if "error" in analysis_result:
                st.error(f"分析エラー: {analysis_result['error']}")
                # エラー内容によっては前のステップに戻すなどの処理も検討
                # state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
                # st.session_state.error_message = analysis_result['error'] # state_managerに関数作っても良い
            else:
                state_manager.store_initial_analysis_result(analysis_result)
                # st.success("初期分析が完了しました！") # デバッグ用
                # 曖昧さに基づいて次のステップを決定
                if st.session_state.is_request_ambiguous:
                    state_manager.add_message("assistant", "ご質問について、もう少し詳しく教えていただけますか？ (初期分析の結果、曖昧な点がありました)") # 仮のメッセージ
                    state_manager.set_current_step(state_manager.STEP_CLARIFICATION_NEEDED)
                else:
                    state_manager.add_message("assistant", "ご質問内容を理解しました。解説の準備をします。 (初期分析完了)") # 仮のメッセージ
                    state_manager.set_current_step(state_manager.STEP_SELECT_STYLE) # 仮にスタイル選択へ
                st.rerun() # 分析結果を反映して再描画
        else: # analysis_result が None の場合 (API呼び出し以前の致命的エラーなど)
            st.error("分析処理中に予期せぬエラーが発生しました。")
            # state_manager.set_current_step(state_manager.STEP_INPUT_SUBMISSION)
            # st.rerun()

    # 分析結果の表示 (デバッグ用)
    if st.session_state.initial_analysis_result and "error" not in st.session_state.initial_analysis_result:
        st.subheader("初期分析結果:")
        st.json(st.session_state.initial_analysis_result)
    elif st.session_state.initial_analysis_result and "error" in st.session_state.initial_analysis_result:
        st.warning("分析結果にエラーが含まれています。上記のエラーメッセージを確認してください。")
    
    # 次のステップに進むためのボタンなどをここに配置しても良い (例: エラー時に再試行ボタン)

elif current_step == state_manager.STEP_CLARIFICATION_NEEDED:
    st.header("3. ご質問の明確化")

    # AIからの明確化質問がまだ表示されていなければ生成・表示
    needs_clarification_q_generation = True
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant":
            needs_clarification_q_generation = False

    if needs_clarification_q_generation and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True)
        with st.spinner("AIが確認のための質問を準備中です..."):
            question_from_ai = tutor_logic.generate_clarification_question_logic()
        state_manager.set_processing_status(False)

        if question_from_ai and "システムエラー" not in question_from_ai and "生成できませんでした" not in question_from_ai:
            state_manager.add_message("assistant", question_from_ai)
            st.rerun()
        else:
            error_msg = question_from_ai or "明確化のための質問を生成できませんでした。"
            state_manager.add_message("system", f"エラー: {error_msg}")
            st.error(error_msg)

elif current_step == state_manager.STEP_SELECT_STYLE:
    st.header("4. 解説スタイルを選択してください")

    # ユーザーに明確化されたリクエスト内容を表示 (確認のため)
    if st.session_state.clarified_request_text:
        st.info(f"明確化されたご要望: 「{st.session_state.clarified_request_text}」")
    elif st.session_state.initial_analysis_result and not st.session_state.is_request_ambiguous:
        # 初期分析で曖昧でなかった場合
        st.info(f"ご要望: 「{st.session_state.initial_analysis_result.get('summary', st.session_state.user_query_text)}」")
    else:
        st.warning("解説対象のリクエストが不明です。最初からやり直してください。")
        if st.button("最初に戻る"):
            state_manager.reset_for_new_session()
            st.rerun()
        st.stop()

    style_options = {
        "detailed": "詳しく解説 (Detailed)",
        "hint": "ヒントのみ (Hint Only)",
        "socratic": "ソクラテス的対話 (Socratic Questioning)"
    }
    
    # 現在選択されているスタイルを取得
    current_selected_style_key = st.session_state.get("selected_explanation_style", "detailed")
    # style_optionsのキーのリストを取得し、現在のスタイルのインデックスを見つける
    options_keys = list(style_options.keys())
    current_index = options_keys.index(current_selected_style_key) if current_selected_style_key in options_keys else 0

    selected_style_key = st.radio(
        "希望する解説のスタイルを選んでください:",
        options=options_keys, # 辞書のキーをオプションとして渡す
        format_func=lambda key: style_options[key], # 表示は辞書のバリューを使う
        index=current_index, # デフォルトで選択されている項目
        horizontal=True,
    )

    if st.button("このスタイルで解説を生成する"):
        state_manager.set_explanation_style(selected_style_key)
        state_manager.add_message("user", f"解説スタイルとして「{style_options[selected_style_key]}」を選択しました。")
        state_manager.set_current_step(state_manager.STEP_GENERATE_EXPLANATION)
        st.rerun()

elif current_step == state_manager.STEP_GENERATE_EXPLANATION:
    st.header("5. 解説生成中...")

    # このステップに入ったら一度だけ解説を生成する (st.rerun対策)
    # current_explanationがNoneで、かつ処理中でない場合に実行
    if st.session_state.current_explanation is None and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True)
        with st.spinner("AIが解説を準備中です... しばらくお待ちください。"):
            explanation = tutor_logic.generate_explanation_logic()
        state_manager.set_processing_status(False)

        if explanation and "システムエラー" not in explanation and "生成できませんでした" not in explanation:
            state_manager.store_generated_explanation(explanation)
            # 解説表示は会話履歴を通じて行われるので、ここでは次のステップへの遷移のみ
            state_manager.set_current_step(state_manager.STEP_FOLLOW_UP_LOOP) # 次はフォローアップループへ
            st.rerun() # 解説を会話履歴に表示するために再描画
        else:
            error_msg = explanation or "解説の生成に失敗しました。もう一度スタイル選択からお試しください。"
            st.error(error_msg)
            state_manager.add_message("system", f"エラー: {error_msg}")
            # エラー時はスタイル選択に戻すなど
            if st.button("スタイル選択に戻る"):
                state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                st.session_state.current_explanation = None # 生成試行をリセット
                st.rerun()
    
    # 解説が生成されたら、それは会話履歴に表示されるので、このステップでは特別な表示は不要
    # フォローアップ入力のために、このステップはすぐにFOLLOW_UP_LOOPに遷移する

elif current_step == state_manager.STEP_FOLLOW_UP_LOOP:
    # st.header("6. 解説の確認と追加の質問") # 必要に応じて
    
    # 「理解しました」ボタンをチャット入力欄の上に表示
    # このボタンは、会話履歴が空でなく、かつAIの最後の発言があった後に表示するのが自然
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button("✅ 理解しました / セッションを終了して要約へ", key="understood_button"):
            state_manager.add_message("user", "（理解しました）") # ユーザーの意思表示を履歴に
            state_manager.set_current_step(state_manager.STEP_CONFIRM_UNDERSTANDING) # 理解確認ステップへ
            st.rerun()
    # (ここに、さらに深掘りする質問を促すメッセージなどを表示しても良い)

elif current_step == state_manager.STEP_CONFIRM_UNDERSTANDING:
    st.header("7. 理解の確認")
    # このステップでは、すぐに要約生成に進むか、
    # LLMに「素晴らしいです！何か他にありますか？」のような最終確認をさせることもできる。
    # 今回はシンプルに、すぐに要約ステップに進むことにする。
    # (将来的には、ここで簡単なフィードバック収集UIを挟んでも良い)
    
    # state_manager.add_message("assistant", "理解されたとのこと、素晴らしいです！このセッションの内容を要約しますね。")
    # st.session_state.messages に直接追加するのではなく、次のステップで生成させる
    
    # すぐに要約ステップへ
    state_manager.set_current_step(state_manager.STEP_SUMMARIZE)
    st.rerun() # 要約ステップの処理を起動

elif current_step == state_manager.STEP_SUMMARIZE:
    st.header("8. 要約と持ち帰りメッセージ")
    if st.session_state.session_summary is None and not st.session_state.get("processing", False):
        state_manager.set_processing_status(True)
        with st.spinner("AIがセッションの要約を準備中です..."):
            summary_text = tutor_logic.generate_summary_logic()
        state_manager.set_processing_status(False)

        if summary_text and "システムエラー" not in summary_text and "生成できませんでした" not in summary_text:
            st.session_state.session_summary = summary_text
            state_manager.add_message("assistant", f"【今回の学習のまとめ】\n\n{summary_text}")
            state_manager.set_current_step(state_manager.STEP_SESSION_END)
            st.rerun()
        else:
            error_msg = summary_text or "要約の生成に失敗しました。"
            st.error(error_msg)
            state_manager.add_message("system", f"エラー: {error_msg}")
            # エラー時はフォローアップに戻るか、終了ボタンだけ表示など
            if st.button("フォローアップに戻る"):
                state_manager.set_current_step(state_manager.STEP_FOLLOW_UP_LOOP)
                st.session_state.session_summary = None # 生成試行をリセット
                st.rerun()
    
    # 要約は会話履歴に表示されるので、ここでは特別な表示はしない。
    # 次のセッションへの導線を表示する。

elif current_step == state_manager.STEP_SESSION_END:
    # st.header("セッション終了") # 会話履歴の最後に要約が表示されているはず
    st.success("学習セッションが終了しました。お疲れ様でした！")
    if st.button("新しい質問を始める", key="new_session_button"):
        state_manager.reset_for_new_session()
        st.rerun()

# --- 会話履歴の表示 (変更なし) ---
if st.session_state.messages:
    # st.subheader("会話履歴") # app.pyのメイン部分でst.titleを使っているので、サブヘッダーは不要かも
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Markdownで表示

# --- チャット入力 (表示条件を調整) ---
# STEP_FOLLOW_UP_LOOP のみでアクティブにするか、他の状態も考慮するか
# 要約表示後は入力不可にするなど
if current_step in [
    state_manager.STEP_CLARIFICATION_NEEDED,
    state_manager.STEP_FOLLOW_UP_LOOP
    # 他のステップでは非表示にするか、disabledにする
]:
    if user_input := st.chat_input(
        "AIへの返答や追加の質問を入力してください...",
        disabled=(st.session_state.get("processing", False) or current_step not in [state_manager.STEP_CLARIFICATION_NEEDED, state_manager.STEP_FOLLOW_UP_LOOP])
        # 処理中や対象ステップ以外では入力を無効化
    ):
        state_manager.add_message("user", user_input)
        
        if current_step == state_manager.STEP_CLARIFICATION_NEEDED:
            # ... (既存の明確化応答分析ロジック - 変更なし) ...
            state_manager.set_processing_status(True)
            with st.spinner("AIが応答を分析中です..."):
                clarification_analysis_result = tutor_logic.analyze_user_clarification_logic(user_input)
            state_manager.set_processing_status(False)
            if clarification_analysis_result: # 以下略、既存コード
                # ...
                if "error" not in clarification_analysis_result and not st.session_state.is_request_ambiguous:
                    state_manager.set_current_step(state_manager.STEP_SELECT_STYLE)
                # ...
            st.rerun()

        elif current_step == state_manager.STEP_FOLLOW_UP_LOOP:
            # ユーザーが解説に対してフォローアップ質問をした
            state_manager.set_processing_status(True)
            with st.spinner("AIがフォローアップの応答を準備中です..."):
                # tutor_logic のフォローアップ応答生成ロジックを呼び出す
                followup_response = tutor_logic.generate_followup_response_logic(user_input)
            state_manager.set_processing_status(False)
            
            if followup_response and "システムエラー" not in followup_response and "生成できませんでした" not in followup_response :
                state_manager.add_message("assistant", followup_response)
            else:
                error_msg = followup_response or "AIが応答を生成できませんでした。"
                state_manager.add_message("system", f"エラー: {error_msg}")
                st.error(error_msg) # エラーを画面にも表示
            st.rerun() # 新しいメッセージを表示するために再描画

# --- デバッグ情報 (変更なし) ---
with st.sidebar:
    st.header("デバッグ情報 (Session State)")
    st.json(st.session_state)