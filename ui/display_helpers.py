# ui/display_helpers.py
import streamlit as st
from typing import Dict, Any, List
from PIL import Image
from io import BytesIO

def display_analysis_result(analysis_data: Dict[str, Any], title: str = "分析結果"):
    """
    分析結果をStreamlit UIに表示する関数。
    st.expander内にJSON形式で表示し、特定のキー(ocr_text_from_extraction)が存在すればテキストエリアも表示。
    """
    with st.expander(title, expanded=False):
        if isinstance(analysis_data, dict):
            if "error" in analysis_data:
                st.error(f"分析エラー: {analysis_data['error']}")
            else:
                # 簡略化のため、ここでは単純にst.jsonを使用
                # 元コードでは特定のキーを特別扱いしていたが、共通コンポーネントとしては汎用的に
                st.json(analysis_data)
            
            # 初期分析時のOCRテキスト表示ロジック (必要に応じて維持)
            if "ocr_text_from_extraction" in analysis_data:
                st.text_area(
                    "初期分析時OCR:", 
                    analysis_data.get("ocr_text_from_extraction"), 
                    height=100, 
                    key=f"exp_ocr_{title.replace(' ','_').lower()}_disp_full_common" # キーの衝突を避けるために変更
                )
        else:
            st.write(analysis_data)

def display_debug_images_app(debug_images_list: List[Dict[str, Any]], title_prefix: str = ""):
    """
    デバッグ画像リストをStreamlit UIに表示する関数。
    st.selectboxを使用して表示する画像を選択し、選択された画像も表示用にリサイズする。
    """
    if not debug_images_list:
        if st.session_state.get("app_mode") == "画像処理チューニング":
             st.info("パラメータを調整して「選択した設定で再処理」ボタンを押すか、新しい画像をアップロードしてください。")
        return

    st.markdown("---")
    st.subheader(f"🎨 {title_prefix}画像処理ステップ")

    options_for_selectbox = [f"{i}: {item.get('label', f'Step {i+1}')}" for i, item in enumerate(debug_images_list)]

    if not options_for_selectbox:
        st.info("表示するデバッグ画像がありません。")
        return

    selectbox_key_suffix = "_app" if st.session_state.get("app_mode") == "AIチューター" else "_tuning"
    selected_debug_image_label = st.selectbox(
        label="表示する処理ステップを選択:",
        options=options_for_selectbox,
        index=0,
        key=f"debug_image_selector{selectbox_key_suffix}"
    )

    if selected_debug_image_label:
        try:
            selected_index = int(selected_debug_image_label.split(":")[0])
            if 0 <= selected_index < len(debug_images_list):
                item_to_display = debug_images_list[selected_index]
                img_data_bytes_or_pil = item_to_display.get("data") # これはbytesかPIL.Image
                original_caption_label = item_to_display.get("label", f"Step {selected_index+1}")
                
                if img_data_bytes_or_pil is not None:
                    img_pil_to_show = None
                    if isinstance(img_data_bytes_or_pil, Image.Image):
                        img_pil_to_show = img_data_bytes_or_pil.copy()
                    elif isinstance(img_data_bytes_or_pil, bytes):
                        try:
                            img_pil_to_show = Image.open(BytesIO(img_data_bytes_or_pil))
                        except Exception as e_open:
                            st.warning(f"デバッグ画像のPillow読み込み失敗 ({original_caption_label}): {e_open}")
                            st.image(img_data_bytes_or_pil, caption=f"Label: {original_caption_label} (Pillow Load Error)", use_container_width=True)
                            return
                    else:
                        st.warning(f"不明な画像データ形式 ({original_caption_label}): {type(img_data_bytes_or_pil)}")
                        return

                    if img_pil_to_show:
                        display_max_width = 800
                        if img_pil_to_show.width > display_max_width:
                            aspect_ratio = img_pil_to_show.height / img_pil_to_show.width
                            new_height = int(display_max_width * aspect_ratio)
                            img_display_resized = img_pil_to_show.resize((display_max_width, new_height), Image.Resampling.LANCZOS)
                        else:
                            img_display_resized = img_pil_to_show.copy()
                        st.image(
                            img_display_resized,
                            caption=f"Label: {original_caption_label}, Mode: {item_to_display.get('mode','N/A')}, Original Size: {img_pil_to_show.width}x{img_pil_to_show.height}, Displayed Size: {img_display_resized.width}x{img_display_resized.height}",
                            use_container_width=False
                        )
                else:
                    st.warning(f"画像データなし ({original_caption_label})")
            else:
                st.error("選択されたインデックスが無効です。")
        except (ValueError, IndexError):
            st.error("選択されたデバッグ画像の処理中にエラーが発生しました。")
            if debug_images_list:
                item_to_display = debug_images_list[0]
                img_data = item_to_display.get("data")
                original_caption_label = item_to_display.get("label", "Step 1")
                if img_data is not None:
                    try:
                        img_pil_fallback = None
                        if isinstance(img_data, Image.Image): img_pil_fallback = img_data
                        elif isinstance(img_data, bytes): img_pil_fallback = Image.open(BytesIO(img_data))
                        if img_pil_fallback:
                            display_max_width = 800
                            if img_pil_fallback.width > display_max_width:
                                aspect_ratio = img_pil_fallback.height / img_pil_fallback.width
                                new_height = int(display_max_width * aspect_ratio)
                                img_pil_fallback_resized = img_pil_fallback.resize((display_max_width, new_height), Image.Resampling.LANCZOS)
                            else:
                                img_pil_fallback_resized = img_pil_fallback.copy()
                            st.image(img_pil_fallback_resized, caption=f"Label: {original_caption_label} (Fallback, Resized)", use_container_width=True)
                        else:
                            st.image(img_data, caption=f"Label: {original_caption_label} (Fallback)", use_container_width=True)
                    except:
                        st.image(img_data, caption=f"Label: {original_caption_label} (Fallback, Resize Error)", use_container_width=True)

    # (オプション) 前後の画像に移動するボタンを追加
    if len(debug_images_list) > 1 and selected_debug_image_label:
        current_idx = options_for_selectbox.index(selected_debug_image_label)
        cols = st.columns(2)
        with cols[0]:
            if st.button("◀ 前のステップ", disabled=(current_idx == 0), key=f"prev_debug_img_btn{selectbox_key_suffix}", use_container_width=True):
                st.info("「前のステップ」ボタン機能は現在開発中です。selectboxから選択してください。")
        with cols[1]:
            if st.button("次のステップ ▶", disabled=(current_idx == len(options_for_selectbox) - 1), key=f"next_debug_img_btn{selectbox_key_suffix}", use_container_width=True):
                st.info("「次のステップ」ボタン機能は現在開発中です。selectboxから選択してください。")