"""HunyuanOCR client library with VRAM management.

This package provides a client for communicating with the VLLM-HunyuanOCR server
and tools for managing GPU memory.

Example usage:
    from hunyuan_ocr_client import HunyuanOCRClient
    from hunyuan_ocr import HunyuanOCRManager
    
    # Simple OCR (server must be running)
    with HunyuanOCRClientSync() as client:
        result = client.ocr_image("document.png")
    
    # With VRAM management - stops server when done
    with HunyuanOCRManager() as manager:
        result = manager.client.ocr_image("page.png")
    # Server stopped, GPU memory freed
"""

from hunyuan_ocr_client import (
    HunyuanOCRClient,
    HunyuanOCRClientSync,
    HunyuanOCRConfig,
    OCRPromptType,
    OCR_PROMPTS,
    OCRResult,
    ServerStatus,
)
from hunyuan_ocr.manager import (
    HunyuanOCRManager,
    IdleWatchdog,
    hunyuan_ocr_session,
    hunyuan_ocr_session_async,
)

__version__ = "0.1.0"

__all__ = [
    # Client
    "HunyuanOCRClient",
    "HunyuanOCRClientSync",
    "HunyuanOCRConfig",
    "OCRPromptType",
    "OCR_PROMPTS",
    "OCRResult",
    "ServerStatus",
    # Manager
    "HunyuanOCRManager",
    "IdleWatchdog",
    "hunyuan_ocr_session",
    "hunyuan_ocr_session_async",
]

