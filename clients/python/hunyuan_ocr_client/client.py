"""HunyuanOCR client for API communication."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypedDict

import httpx
from PIL import Image

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


class OCRPromptType(str, Enum):
    """Pre-defined OCR prompts for different tasks."""
    
    # Text spotting
    SPOTTING_EN = "spotting_en"
    SPOTTING_ZH = "spotting_zh"
    # Parsing
    FORMULA = "formula"
    TABLE = "table"
    CHART = "chart"
    DOCUMENT = "document"
    # Information extraction
    SUBTITLES = "subtitles"
    # Translation
    TRANSLATE_EN = "translate_to_english"


OCR_PROMPTS: dict[OCRPromptType, str] = {
    OCRPromptType.SPOTTING_EN: "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
    OCRPromptType.SPOTTING_ZH: "检测并识别图片中的文字，将文本坐标格式化输出。",
    OCRPromptType.FORMULA: "Identify the formula in the image and represent it using LaTeX format.",
    OCRPromptType.TABLE: "Parse the table in the image into HTML.",
    OCRPromptType.CHART: "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.",
    OCRPromptType.DOCUMENT: (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in HTML "
        "format, formulas in the document should be represented using LaTeX format, and the "
        "parsing should be organized according to the reading order."
    ),
    OCRPromptType.SUBTITLES: "Extract the subtitles from the image.",
    OCRPromptType.TRANSLATE_EN: (
        "First extract the text, then translate the text content into English. If it is a "
        "document, ignore the header and footer. Formulas should be represented in LaTeX "
        "format, and tables should be represented in HTML format."
    ),
}


class OCRResult(TypedDict):
    """Result from OCR operation."""
    
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ServerStatus(TypedDict):
    """Server health status."""
    
    healthy: bool
    model_loaded: bool
    error: str | None


@dataclass
class HunyuanOCRConfig:
    """Configuration for HunyuanOCR client."""
    
    server_url: str = "http://localhost:8000"
    model: str = "tencent/HunyuanOCR"
    api_key: str = "EMPTY"
    
    # Generation settings
    max_tokens: int = 16384
    temperature: float = 0.0
    
    # Timeouts (in seconds)
    connect_timeout: float = 10.0
    read_timeout: float = 120.0
    
    # Connection pool settings (for concurrent/threaded requests)
    max_connections: int = 10
    max_keepalive_connections: int = 5
    
    # Concurrency settings
    max_workers: int = 4  # Default thread pool size
    
    # Docker settings for VRAM management
    docker_compose_path: Path | str | None = None
    container_name: str = "hunyuan-ocr"
    
    # Health check settings
    health_check_interval: float = 2.0
    health_check_timeout: float = 300.0  # 5 minutes for model loading


def _image_to_base64(image: Image.Image | Path | str) -> tuple[str, str]:
    """Convert image to base64 string and determine media type."""
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        ext = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(ext, "image/png")
        
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, media_type
    
    # PIL Image
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return b64, "image/png"


class HunyuanOCRClient:
    """Async client for HunyuanOCR server.
    
    Example:
        async with HunyuanOCRClient() as client:
            result = await client.ocr_image("document.png")
            print(result["text"])
    """
    
    def __init__(self, config: HunyuanOCRConfig | None = None):
        self.config = config or HunyuanOCRConfig()
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "HunyuanOCRClient":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def connect(self) -> None:
        """Initialize the HTTP client."""
        if self._client is None:
            timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
                write=30.0,
                pool=10.0,
            )
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
            )
            self._client = httpx.AsyncClient(
                base_url=self.config.server_url,
                timeout=timeout,
                limits=limits,
                headers={"Content-Type": "application/json"},
            )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> ServerStatus:
        """Check if the server is healthy and model is loaded."""
        await self.connect()
        assert self._client is not None
        
        try:
            response = await self._client.get("/health")
            if response.status_code == 200:
                return ServerStatus(healthy=True, model_loaded=True, error=None)
            return ServerStatus(
                healthy=False,
                model_loaded=False,
                error=f"Health check returned {response.status_code}",
            )
        except httpx.ConnectError as e:
            return ServerStatus(healthy=False, model_loaded=False, error=f"Connection error: {e}")
        except Exception as e:
            return ServerStatus(healthy=False, model_loaded=False, error=str(e))
    
    async def wait_for_ready(
        self,
        timeout: float | None = None,
        interval: float | None = None,
    ) -> bool:
        """Wait for the server to be ready."""
        timeout = timeout or self.config.health_check_timeout
        interval = interval or self.config.health_check_interval
        
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            status = await self.health_check()
            if status["healthy"]:
                return True
            await asyncio.sleep(interval)
        
        return False
    
    async def ocr_image(
        self,
        image: Image.Image | Path | str,
        prompt: str | OCRPromptType = OCRPromptType.SPOTTING_EN,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> OCRResult:
        """Perform OCR on an image."""
        await self.connect()
        assert self._client is not None
        
        if isinstance(prompt, OCRPromptType):
            prompt_text = OCR_PROMPTS[prompt]
        else:
            prompt_text = prompt
        
        image_b64, media_type = _image_to_base64(image)
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        usage = data.get("usage", {})
        
        return OCRResult(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
    
    async def ocr_batch(
        self,
        images: list[Image.Image | Path | str],
        prompt: str | OCRPromptType = OCRPromptType.SPOTTING_EN,
        *,
        max_concurrency: int = 4,
    ) -> list[OCRResult]:
        """Perform OCR on multiple images concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_one(img: Image.Image | Path | str) -> OCRResult:
            async with semaphore:
                return await self.ocr_image(img, prompt)
        
        tasks = [process_one(img) for img in images]
        return await asyncio.gather(*tasks)


class HunyuanOCRClientSync:
    """Synchronous, thread-safe client for HunyuanOCR server.
    
    Example - Single threaded:
        with HunyuanOCRClientSync() as client:
            result = client.ocr_image("document.png")
    
    Example - Batch with threads:
        with HunyuanOCRClientSync() as client:
            results = client.ocr_batch(images, max_workers=8)
    """
    
    def __init__(self, config: HunyuanOCRConfig | None = None):
        self.config = config or HunyuanOCRConfig()
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
    
    def __enter__(self) -> "HunyuanOCRClientSync":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def connect(self) -> None:
        """Initialize the HTTP client (thread-safe)."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    timeout = httpx.Timeout(
                        connect=self.config.connect_timeout,
                        read=self.config.read_timeout,
                        write=30.0,
                        pool=10.0,
                    )
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive_connections,
                    )
                    self._client = httpx.Client(
                        base_url=self.config.server_url,
                        timeout=timeout,
                        limits=limits,
                        headers={"Content-Type": "application/json"},
                    )
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def health_check(self) -> ServerStatus:
        """Check if the server is healthy."""
        self.connect()
        assert self._client is not None
        
        try:
            response = self._client.get("/health")
            if response.status_code == 200:
                return ServerStatus(healthy=True, model_loaded=True, error=None)
            return ServerStatus(
                healthy=False,
                model_loaded=False,
                error=f"Health check returned {response.status_code}",
            )
        except httpx.ConnectError as e:
            return ServerStatus(healthy=False, model_loaded=False, error=f"Connection error: {e}")
        except Exception as e:
            return ServerStatus(healthy=False, model_loaded=False, error=str(e))
    
    def wait_for_ready(
        self,
        timeout: float | None = None,
        interval: float | None = None,
    ) -> bool:
        """Wait for the server to be ready."""
        timeout = timeout or self.config.health_check_timeout
        interval = interval or self.config.health_check_interval
        
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            status = self.health_check()
            if status["healthy"]:
                return True
            time.sleep(interval)
        
        return False
    
    def ocr_image(
        self,
        image: Image.Image | Path | str,
        prompt: str | OCRPromptType = OCRPromptType.SPOTTING_EN,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> OCRResult:
        """Perform OCR on an image."""
        self.connect()
        assert self._client is not None
        
        if isinstance(prompt, OCRPromptType):
            prompt_text = OCR_PROMPTS[prompt]
        else:
            prompt_text = prompt
        
        image_b64, media_type = _image_to_base64(image)
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        
        response = self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        usage = data.get("usage", {})
        
        return OCRResult(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
    
    def ocr_batch(
        self,
        images: list[Image.Image | Path | str],
        prompt: str | OCRPromptType = OCRPromptType.SPOTTING_EN,
        *,
        max_workers: int | None = None,
        preserve_order: bool = True,
    ) -> list[OCRResult]:
        """Perform OCR on multiple images using a thread pool."""
        self.connect()
        max_workers = max_workers or self.config.max_workers
        
        if preserve_order:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(
                    lambda img: self.ocr_image(img, prompt),
                    images,
                ))
            return results
        else:
            results: list[OCRResult] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.ocr_image, img, prompt): i
                    for i, img in enumerate(images)
                }
                for future in as_completed(futures):
                    results.append(future.result())
            return results
    
    def ocr_batch_with_callback(
        self,
        images: list[Image.Image | Path | str],
        prompt: str | OCRPromptType = OCRPromptType.SPOTTING_EN,
        *,
        callback: Callable[[int, OCRResult], None] | None = None,
        error_callback: Callable[[int, Exception], None] | None = None,
        max_workers: int | None = None,
    ) -> list[OCRResult | None]:
        """Perform batch OCR with callbacks for progress tracking."""
        self.connect()
        max_workers = max_workers or self.config.max_workers
        results: list[OCRResult | None] = [None] * len(images)
        
        def process_image(idx: int, img: Image.Image | Path | str) -> tuple[int, OCRResult | None, Exception | None]:
            try:
                result = self.ocr_image(img, prompt)
                return (idx, result, None)
            except Exception as e:
                return (idx, None, e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image, i, img)
                for i, img in enumerate(images)
            ]
            
            for future in as_completed(futures):
                idx, result, error = future.result()
                results[idx] = result
                
                if error is not None:
                    if error_callback:
                        error_callback(idx, error)
                elif callback:
                    callback(idx, result)  # type: ignore
        
        return results

