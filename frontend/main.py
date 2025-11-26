"""
HunyuanOCR Web Frontend - FastAPI Backend
"""

import asyncio
import base64
import io
import os
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pdf2image import convert_from_bytes
from PIL import Image

app = FastAPI(title="HunyuanOCR Frontend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR server URL (internal docker network)
OCR_SERVER_URL = os.getenv("OCR_SERVER_URL", "http://hunyuan-ocr:8000")

# Predefined prompts
PROMPTS = {
    "spotting": "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
    "document": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.",
    "table": "Parse the table in the image into HTML.",
    "formula": "Identify the formula in the image and represent it using LaTeX format.",
    "chart": "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.",
    "subtitles": "Extract the subtitles from the image.",
    "translate_english": "First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format.",
    "translate_chinese": "先提取文字，再将文字内容翻译为中文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。",
}


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def process_image_with_ocr(
    image_base64: str, prompt: str, timeout: float = 300.0, max_retries: int = 3
) -> str:
    """Send image to OCR server and get result with retry logic."""
    payload = {
        "model": "tencent/HunyuanOCR",
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 16384,
        "temperature": 0,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{OCR_SERVER_URL}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            continue
        except httpx.HTTPStatusError as e:
            # Don't retry on HTTP errors (4xx, 5xx) - they're likely not transient
            raise e
    
    raise last_error or httpx.ConnectError("OCR server connection failed after retries")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/backend-status")
async def backend_status():
    """Check if the OCR backend is ready."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OCR_SERVER_URL}/health")
            if response.status_code == 200:
                return {"status": "ready", "message": "OCR server is ready"}
    except Exception:
        pass
    return {"status": "loading", "message": "OCR server is still loading the model. Please wait..."}


@app.get("/api/prompts")
async def get_prompts():
    """Get available prompt templates."""
    return PROMPTS


@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    task: str = Form("document"),
    custom_prompt: Optional[str] = Form(None),
    pages: Optional[str] = Form(None),  # e.g., "1,2,3" or "1-5" or "all"
):
    """
    Process a PDF or image file with OCR.
    
    Args:
        file: The uploaded file (PDF or image)
        task: The task type (one of the predefined prompts)
        custom_prompt: Optional custom prompt (overrides task)
        pages: Which pages to process (for PDFs)
    """
    # Determine the prompt to use
    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()
    elif task in PROMPTS:
        prompt = PROMPTS[task]
    else:
        prompt = PROMPTS["document"]

    # Read file content
    content = await file.read()
    filename = file.filename.lower() if file.filename else ""

    results = []

    try:
        if filename.endswith(".pdf"):
            # Convert PDF to images at maximum DPI for best detail detection
            # (underlines, bold text, fine formatting)
            images = convert_from_bytes(content, dpi=600)
            
            # Parse pages parameter
            if pages and pages.strip() and pages.strip().lower() != "all":
                page_indices = parse_page_range(pages, len(images))
            else:
                page_indices = list(range(len(images)))

            # Process each page
            for i, page_idx in enumerate(page_indices):
                if page_idx < 0 or page_idx >= len(images):
                    continue
                    
                image = images[page_idx]
                image_b64 = image_to_base64(image)
                
                try:
                    text = await process_image_with_ocr(image_b64, prompt)
                    results.append({
                        "page": page_idx + 1,
                        "success": True,
                        "content": text,
                    })
                except Exception as e:
                    results.append({
                        "page": page_idx + 1,
                        "success": False,
                        "error": str(e),
                    })

        else:
            # Process as single image
            image = Image.open(io.BytesIO(content))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            image_b64 = image_to_base64(image)
            text = await process_image_with_ocr(image_b64, prompt)
            
            results.append({
                "page": 1,
                "success": True,
                "content": text,
            })

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="OCR server is not available. Please ensure the HunyuanOCR service is running.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename": file.filename,
        "total_pages": len(results),
        "results": results,
    }


def parse_page_range(pages_str: str, total_pages: int) -> list[int]:
    """Parse page range string like '1,2,3' or '1-5' or '1,3-5,7'."""
    indices = []
    parts = pages_str.replace(" ", "").split(",")
    
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start) - 1  # Convert to 0-indexed
            end = int(end)  # End is inclusive, but range is exclusive
            indices.extend(range(max(0, start), min(end, total_pages)))
        else:
            idx = int(part) - 1  # Convert to 0-indexed
            if 0 <= idx < total_pages:
                indices.append(idx)
    
    return sorted(set(indices))


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

