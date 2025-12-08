#!/usr/bin/env python3
"""Smart proxy for vLLM server with auto-start and idle shutdown.

This proxy:
1. Receives all requests intended for the vLLM server
2. Automatically starts the server if it's stopped
3. Waits for it to become healthy before forwarding
4. Tracks activity and shuts down after idle timeout to free VRAM

Environment variables:
    VLLM_URL: Backend vLLM server URL (default: http://hunyuan-ocr:8000)
    PROXY_PORT: Port for this proxy to listen on (default: 8000)
    IDLE_TIMEOUT: Seconds of inactivity before shutdown (default: 300 = 5 min)
    CHECK_INTERVAL: Seconds between idle checks (default: 30)
    STARTUP_TIMEOUT: Max seconds to wait for server startup (default: 600 = 10 min)
    CONTAINER_NAME: Docker container name to start/stop (default: hunyuan-ocr)
"""

import asyncio
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [proxy] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(name, "")
    if value.strip():
        try:
            return int(value)
        except ValueError:
            log.warning(f"Invalid {name}={value}, using default {default}")
    return default


# Configuration
VLLM_URL = os.environ.get("VLLM_URL", "http://hunyuan-ocr:8000")
PROXY_PORT = get_env_int("PROXY_PORT", 8000)
IDLE_TIMEOUT = get_env_int("IDLE_TIMEOUT", 300)
CHECK_INTERVAL = get_env_int("CHECK_INTERVAL", 30)
STARTUP_TIMEOUT = get_env_int("STARTUP_TIMEOUT", 600)  # 10 min for model loading
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "hunyuan-ocr")


class ServerManager:
    """Manages the backend vLLM server lifecycle."""

    def __init__(self):
        self.last_activity_time = time.monotonic()
        self._lock = asyncio.Lock()
        self._starting = False
        self._healthy = False

    def record_activity(self):
        """Record that a request was made."""
        self.last_activity_time = time.monotonic()

    def get_idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return time.monotonic() - self.last_activity_time

    async def check_health(self, client: httpx.AsyncClient) -> bool:
        """Check if the backend server is healthy."""
        try:
            response = await client.get(f"{VLLM_URL}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    async def is_container_running(self) -> bool:
        """Check if the Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip().lower() == "true"
        except Exception as e:
            log.error(f"Failed to check container status: {e}")
            return False

    async def start_container(self) -> bool:
        """Start the Docker container."""
        log.info(f"🚀 Starting container: {CONTAINER_NAME}")
        try:
            result = subprocess.run(
                ["docker", "start", CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                log.info(f"✓ Container start command successful")
                return True
            else:
                log.error(f"Failed to start container: {result.stderr}")
                return False
        except Exception as e:
            log.error(f"Failed to start container: {e}")
            return False

    async def stop_container(self) -> bool:
        """Stop the Docker container to free VRAM."""
        log.info(f"🛑 Stopping container to free VRAM: {CONTAINER_NAME}")
        try:
            result = subprocess.run(
                ["docker", "stop", CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                log.info(f"✓ Container stopped successfully")
                self._healthy = False
                return True
            else:
                log.error(f"Failed to stop container: {result.stderr}")
                return False
        except Exception as e:
            log.error(f"Failed to stop container: {e}")
            return False

    async def ensure_server_ready(self, client: httpx.AsyncClient) -> bool:
        """Ensure the server is running and healthy, starting it if needed."""
        async with self._lock:
            # Quick check if already healthy
            if self._healthy and await self.check_health(client):
                return True

            self._healthy = False
            self._starting = True

            try:
                # Check if container is running
                if not await self.is_container_running():
                    log.info("Backend server not running, starting it...")
                    if not await self.start_container():
                        return False

                # Wait for health check to pass
                log.info("⏳ Waiting for server to become healthy...")
                start_time = time.monotonic()
                check_count = 0

                while time.monotonic() - start_time < STARTUP_TIMEOUT:
                    check_count += 1
                    if await self.check_health(client):
                        elapsed = time.monotonic() - start_time
                        log.info(f"✓ Server healthy after {elapsed:.1f}s ({check_count} checks)")
                        self._healthy = True
                        self.record_activity()
                        return True

                    # Progressive backoff: 2s for first 30s, then 5s
                    wait_time = 2 if (time.monotonic() - start_time) < 30 else 5
                    await asyncio.sleep(wait_time)

                log.error(f"Server failed to become healthy within {STARTUP_TIMEOUT}s")
                return False

            finally:
                self._starting = False

    @property
    def is_starting(self) -> bool:
        """Check if server is currently starting up."""
        return self._starting


# Global server manager
server_manager = ServerManager()

# Shared HTTP client for backend communication
http_client: httpx.AsyncClient = None


async def idle_monitor():
    """Background task to monitor idle time and shutdown if needed."""
    global http_client

    log.info(f"Idle monitor started (timeout: {IDLE_TIMEOUT}s, check: {CHECK_INTERVAL}s)")

    while True:
        await asyncio.sleep(CHECK_INTERVAL)

        # Don't check during startup
        if server_manager.is_starting:
            continue

        # Only check if server is supposed to be running
        if not server_manager._healthy:
            continue

        idle_seconds = server_manager.get_idle_seconds()
        remaining = IDLE_TIMEOUT - idle_seconds

        if remaining <= 0:
            log.warning(f"Server idle for {idle_seconds:.0f}s (timeout: {IDLE_TIMEOUT}s)")
            await server_manager.stop_container()
        else:
            # Log every check but only at debug level unless getting close
            if remaining <= 60:
                log.info(f"⏰ Idle: {idle_seconds:.0f}s / {IDLE_TIMEOUT}s (shutdown in {remaining:.0f}s)")
            else:
                log.debug(f"Idle: {idle_seconds:.0f}s / {IDLE_TIMEOUT}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global http_client

    log.info("=" * 60)
    log.info("vLLM Smart Proxy Server")
    log.info("=" * 60)
    log.info(f"  Backend URL: {VLLM_URL}")
    log.info(f"  Container: {CONTAINER_NAME}")
    log.info(f"  Idle timeout: {IDLE_TIMEOUT}s ({IDLE_TIMEOUT // 60}m)")
    log.info(f"  Startup timeout: {STARTUP_TIMEOUT}s ({STARTUP_TIMEOUT // 60}m)")
    log.info("=" * 60)

    # Create shared HTTP client with long timeouts for model inference
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=600, write=60, pool=30),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    # Start idle monitor background task
    monitor_task = asyncio.create_task(idle_monitor())

    yield

    # Cleanup
    monitor_task.cancel()
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/proxy/health")
async def proxy_health():
    """Health check for the proxy itself (always healthy if running)."""
    return {"status": "healthy", "service": "proxy"}


@app.get("/proxy/status")
async def proxy_status():
    """Get detailed status of the proxy and backend."""
    is_running = await server_manager.is_container_running()
    is_healthy = await server_manager.check_health(http_client) if is_running else False

    return {
        "proxy": "healthy",
        "backend": {
            "container": CONTAINER_NAME,
            "running": is_running,
            "healthy": is_healthy,
            "starting": server_manager.is_starting,
        },
        "idle": {
            "seconds": server_manager.get_idle_seconds(),
            "timeout": IDLE_TIMEOUT,
            "remaining": max(0, IDLE_TIMEOUT - server_manager.get_idle_seconds()),
        },
    }


async def stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    """Stream response content from backend."""
    async for chunk in response.aiter_bytes():
        yield chunk


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    """Proxy all requests to the backend vLLM server."""
    global http_client

    # Record activity immediately
    server_manager.record_activity()

    # Ensure server is running and healthy
    if not await server_manager.ensure_server_ready(http_client):
        return Response(
            content='{"error": "Backend server failed to start"}',
            status_code=503,
            media_type="application/json",
        )

    # Build the backend URL
    url = f"{VLLM_URL}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    # Get request body
    body = await request.body()

    # Forward headers (filter out hop-by-hop headers)
    headers = dict(request.headers)
    for h in ["host", "connection", "keep-alive", "transfer-encoding", "upgrade"]:
        headers.pop(h, None)

    try:
        # Make the request to backend
        backend_request = http_client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        response = await http_client.send(backend_request, stream=True)

        # Check if this is a streaming response
        content_type = response.headers.get("content-type", "")
        is_streaming = (
            "text/event-stream" in content_type
            or "chunked" in response.headers.get("transfer-encoding", "")
        )

        # Build response headers (filter hop-by-hop)
        response_headers = dict(response.headers)
        for h in ["connection", "keep-alive", "transfer-encoding"]:
            response_headers.pop(h, None)

        if is_streaming:
            # Stream the response
            return StreamingResponse(
                stream_response(response),
                status_code=response.status_code,
                headers=response_headers,
                media_type=content_type,
            )
        else:
            # Read full response
            content = await response.aread()
            await response.aclose()
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
            )

    except httpx.TimeoutException:
        log.error(f"Backend request timeout: {request.method} {path}")
        return Response(
            content='{"error": "Backend request timeout"}',
            status_code=504,
            media_type="application/json",
        )
    except Exception as e:
        log.error(f"Backend request failed: {e}")
        # Server might have crashed, mark as unhealthy
        server_manager._healthy = False
        return Response(
            content=f'{{"error": "Backend request failed: {str(e)}"}}',
            status_code=502,
            media_type="application/json",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")

