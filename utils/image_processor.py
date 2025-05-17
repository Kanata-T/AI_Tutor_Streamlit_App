# utils/image_utils.py

from PIL import Image, ExifTags
from io import BytesIO
from typing import Optional, Dict, Literal
from utils import config_loader  # config_loader をインポート

# --- 設定値を config_loader から取得 ---
IMG_PROC_CONFIG = config_loader.get_image_processing_config()

# Pillowの最大ピクセル数設定 (configから取得、なければPillowデフォルト)
if IMG_PROC_CONFIG.get("pillow_max_image_pixels"):
    try:
        Image.MAX_IMAGE_PIXELS = int(IMG_PROC_CONFIG["pillow_max_image_pixels"])
    except (ValueError, TypeError):
        print(f"[image_processor] Warning: Invalid 'pillow_max_image_pixels' in config. Using Pillow default.")
        Image.MAX_IMAGE_PIXELS = None  # Pillowデフォルトに任せる
else:
    Image.MAX_IMAGE_PIXELS = None  # 明示的にNone (Pillowデフォルト)

SUPPORTED_MIME_TYPES = IMG_PROC_CONFIG.get("supported_mime_types", ["image/png", "image/jpeg", "image/webp"])
CONVERTABLE_MIME_TYPES = IMG_PROC_CONFIG.get("convertable_mime_types", {"image/gif": "image/png", "image/bmp": "image/png"})
DEFAULT_MAX_PIXELS = IMG_PROC_CONFIG.get("default_max_pixels_for_resizing", 4000000)
DEFAULT_OUTPUT_FORMAT = IMG_PROC_CONFIG.get("default_output_format", "JPEG").upper()  # 大文字に正規化
DEFAULT_JPEG_QUALITY = IMG_PROC_CONFIG.get("default_jpeg_quality", 85)

def get_image_orientation(img: Image.Image) -> Optional[int]:
    """
    画像のEXIF情報から向きタグを取得する。

    Args:
        img: PIL Imageオブジェクト。

    Returns:
        EXIFの向きタグの値 (int)、または見つからない/エラーの場合はNone。
    """
    try:
        exif = img.getexif() # アンダースコアなしの公開APIを使用
        if not exif: # exifが空の辞書の場合
            return None

        for k, v in exif.items():
            if k in ExifTags.TAGS and ExifTags.TAGS[k] == 'Orientation':
                return v
    except Exception as e:
        print(f"[image_processor] Warning: Could not get EXIF data: {e}")
        return None
    return None

def correct_image_orientation_from_exif(img: Image.Image) -> Image.Image:
    """
    画像のEXIF情報に基づいて向きを補正する。

    Args:
        img: PIL Imageオブジェクト。

    Returns:
        向きが補正されたPIL Imageオブジェクト。補正不要/失敗時は元画像を返す。
    """
    orientation = get_image_orientation(img)

    if orientation == 2:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        img = img.rotate(180)
    elif orientation == 4:
        img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        img = img.rotate(-90, expand=True)
    elif orientation == 7:
        img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        img = img.rotate(90, expand=True)
    # 他のorientation値は無視、またはエラーログを出すなどしても良い
    
    return img

def preprocess_uploaded_image(
    uploaded_file_data: bytes,
    mime_type: str,
    max_pixels: int = DEFAULT_MAX_PIXELS,  # configからの値をデフォルトに
    output_format: Literal["JPEG", "PNG"] = DEFAULT_OUTPUT_FORMAT,  # configからの値をデフォルトに
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,  # configからの値をデフォルトに
) -> Optional[Dict[str, any]]:
    if not uploaded_file_data:
        print("[image_processor] Error: No image data provided for preprocessing.")
        return None

    original_mime_type = mime_type.lower()
    target_output_format = output_format.upper()  # 大文字に統一
    processed_mime_type = ""

    if original_mime_type in CONVERTABLE_MIME_TYPES:
        processed_mime_type = CONVERTABLE_MIME_TYPES[original_mime_type]
        if processed_mime_type == "image/png":
            target_output_format = "PNG"
        elif processed_mime_type == "image/jpeg":
            target_output_format = "JPEG"
        print(f"[image_processor] Info: Converting {original_mime_type} to {processed_mime_type} (output as {target_output_format}).")
    elif original_mime_type in SUPPORTED_MIME_TYPES:
        if target_output_format == "JPEG":
            processed_mime_type = "image/jpeg"
        elif target_output_format == "PNG":
            processed_mime_type = "image/png"
        else:
            print(f"[image_processor] Warning: Unsupported target_output_format '{target_output_format}' for {original_mime_type}. Defaulting to {DEFAULT_OUTPUT_FORMAT}.")
            target_output_format = DEFAULT_OUTPUT_FORMAT  # configのデフォルトを使用
            processed_mime_type = "image/jpeg" if target_output_format == "JPEG" else "image/png"
    else:
        print(f"[image_processor] Error: Unsupported original MIME type '{original_mime_type}' and not in convertable list.")
        return {"error": f"サポートされていない画像形式です: {original_mime_type}。"}

    try:
        img = Image.open(BytesIO(uploaded_file_data))
        img = correct_image_orientation_from_exif(img)
        print(f"[image_processor] Debug: Orientation corrected. Mode: {img.mode}, Size: {img.size}")

        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            print(f"[image_processor] Debug: Alpha channel detected (mode: {img.mode}). Converting to RGB with white background.")
            try:
                img_rgb = Image.new("RGB", img.size, (255, 255, 255))
                img_rgb.paste(img, mask=img.split()[-1])
                img = img_rgb
            except Exception:
                img = img.convert('RGB')
        elif img.mode != 'RGB':
            print(f"[image_processor] Debug: Mode is {img.mode}. Converting to RGB.")
            img = img.convert('RGB')
        
        print(f"[image_processor] Debug: Converted to RGB. Mode: {img.mode}, Size: {img.size}")

        current_pixels = img.width * img.height
        if max_pixels > 0 and current_pixels > max_pixels:
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            if new_width >= 1 and new_height >= 1:
                print(f"[image_processor] Debug: Resizing from {img.size} to ({new_width}, {new_height}).")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                print(f"[image_processor] Warning: Resize resulted in zero dimension for {original_mime_type}. Skipping resize.")
        
        print(f"[image_processor] Debug: Resized (if needed). Final size: {img.size}")

        output_bytes_io = BytesIO()
        save_kwargs = {}
        if target_output_format == "JPEG":
            save_kwargs['quality'] = jpeg_quality
        img.save(output_bytes_io, format=target_output_format, **save_kwargs)
        processed_image_data = output_bytes_io.getvalue()
        
        print(f"[image_processor] Debug: Saved to bytes. Format: {target_output_format}, Size: {len(processed_image_data) / 1024:.1f} KB")

        return {
            "mime_type": processed_mime_type,
            "data": processed_image_data
        }

    except Image.DecompressionBombError as e_bomb:
        print(f"[image_processor] Error: Image is too large (DecompressionBombError): {e_bomb}")
        return {"error": "画像が大きすぎるか、非対応の形式の可能性があります。"}
    except ValueError as e_val:
        if "Pixel count exceeds limit" in str(e_val):
            print(f"[image_processor] Error: Image pixel count exceeds Pillow's limit: {e_val}")
            return {"error": "画像のピクセル数が大きすぎます。"}
        else:
            print(f"[image_processor] Error during image preprocessing (ValueError): {e_val}")
            return {"error": "画像処理中にエラーが発生しました。"}
    except Exception as e:
        print(f"[image_processor] Error during image preprocessing: {e}")
        return {"error": "画像処理中に予期せぬエラーが発生しました。"}