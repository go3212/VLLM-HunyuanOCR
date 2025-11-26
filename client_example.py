"""
Example client for HunyuanOCR vLLM server.

Usage:
    python client_example.py path/to/your/image.jpg
"""

import base64
import sys
from pathlib import Path

import requests


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the media type based on file extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def ocr_image(
    image_path: str,
    prompt: str = "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
    server_url: str = "http://localhost:8000",
) -> str:
    """
    Send an image to the HunyuanOCR server and get the OCR result.

    Args:
        image_path: Path to the image file
        prompt: The instruction prompt for OCR
        server_url: The vLLM server URL

    Returns:
        The OCR result text
    """
    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    # Prepare the request payload
    payload = {
        "model": "tencent/HunyuanOCR",
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 16384,
        "temperature": 0,
    }

    # Send request to the server
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


# Application-oriented prompts
PROMPTS = {
    # Text Spotting
    "spotting_en": "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
    "spotting_zh": "检测并识别图片中的文字，将文本坐标格式化输出。",
    # Parsing
    "formula": "Identify the formula in the image and represent it using LaTeX format.",
    "table": "Parse the table in the image into HTML.",
    "chart": "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.",
    "document": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.",
    # Information Extraction
    "subtitles": "Extract the subtitles from the image.",
    # Translation
    "translate_to_english": "First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format.",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python client_example.py <image_path> [prompt_key]")
        print("\nAvailable prompt keys:")
        for key in PROMPTS:
            print(f"  - {key}")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt_key = sys.argv[2] if len(sys.argv) > 2 else "spotting_en"

    if prompt_key in PROMPTS:
        prompt = PROMPTS[prompt_key]
    else:
        prompt = prompt_key  # Use as custom prompt

    print(f"Processing image: {image_path}")
    print(f"Using prompt: {prompt}\n")

    try:
        result = ocr_image(image_path, prompt)
        print("=" * 50)
        print("OCR Result:")
        print("=" * 50)
        print(result)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the container is running.")
        print("Start the server with: docker compose up -d")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

