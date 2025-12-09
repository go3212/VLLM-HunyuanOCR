"""VRAM management for HunyuanOCR server."""

from __future__ import annotations

import atexit
import logging
import subprocess
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

from hunyuan_ocr_client import (
    HunyuanOCRClient,
    HunyuanOCRClientSync,
    HunyuanOCRConfig,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_log = logging.getLogger(__name__)

# Get the package directory (where docker-compose.yml is located)
_PACKAGE_DIR = Path(__file__).parent.parent


class IdleWatchdog:
    """Background thread that monitors activity and stops server after inactivity.
    
    This is a client-side watchdog - it runs in your Python process and stops
    the server container when there's been no OCR activity for `idle_timeout` seconds.
    
    Example:
        manager = HunyuanOCRManager(idle_timeout=300)  # 5 minute timeout
        manager.start()
        
        # Do OCR work...
        result = manager.client.ocr_image("page.png")
        
        # If no more requests for 5 minutes, server auto-stops
        # You can also manually stop:
        manager.stop()
    """
    
    def __init__(
        self,
        idle_timeout: float,
        check_interval: float,
        on_idle: "Callable[[], None]",
    ):
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.on_idle = on_idle
        
        self._last_activity = time.monotonic()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
    
    def record_activity(self) -> None:
        """Record that activity occurred (call this on each OCR request)."""
        with self._lock:
            self._last_activity = time.monotonic()
    
    def start(self) -> None:
        """Start the watchdog thread."""
        if self._running:
            return
        
        self._running = True
        self._last_activity = time.monotonic()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        _log.debug(f"Idle watchdog started (timeout: {self.idle_timeout}s)")
    
    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            time.sleep(self.check_interval)
            
            if not self._running:
                break
            
            with self._lock:
                idle_time = time.monotonic() - self._last_activity
            
            if idle_time >= self.idle_timeout:
                _log.info(f"Server idle for {idle_time:.0f}s, triggering auto-shutdown")
                self.on_idle()
                self._running = False
                break


class HunyuanOCRManager:
    """Manages HunyuanOCR server lifecycle for VRAM management.
    
    This class controls when the HunyuanOCR Docker container is running,
    allowing you to free GPU memory when running other GPU-intensive tasks.
    
    VRAM Management Strategies:
        1. Manual: Call `stop()` to free VRAM, `start()` to load model
        2. Context manager: Auto-stop on exit
        3. Auto-shutdown: Set `idle_timeout` to auto-stop after inactivity
        
    Example - Manual control:
        manager = HunyuanOCRManager()
        manager.start()
        result = manager.client.ocr_image("page.png")
        manager.stop()  # Frees VRAM
    
    Example - Context manager:
        with HunyuanOCRManager() as manager:
            result = manager.client.ocr_image("page.png")
        # Server automatically stopped, VRAM freed
    
    Example - Auto-shutdown after 5 min idle:
        manager = HunyuanOCRManager(idle_timeout=300)
        manager.start()
        result = manager.client.ocr_image("page.png")
        # Server auto-stops after 5 min of no requests
    """
    
    def __init__(
        self,
        config: HunyuanOCRConfig | None = None,
        docker_compose_path: Path | str | None = None,
        auto_start: bool = True,
        wait_for_ready: bool = True,
        idle_timeout: float | None = None,
        idle_check_interval: float = 30.0,
    ):
        """Initialize the manager.
        
        Args:
            config: Client configuration.
            docker_compose_path: Path to directory containing docker-compose.yml.
                                 Defaults to the package directory.
            auto_start: If True, start server on context enter.
            wait_for_ready: If True, wait for server to be healthy after start.
            idle_timeout: If set, automatically stop server after this many
                         seconds of inactivity. None = no auto-shutdown.
            idle_check_interval: How often to check for idle (seconds).
        """
        self.config = config or HunyuanOCRConfig()
        
        # Determine docker-compose path
        if docker_compose_path:
            self._compose_path = Path(docker_compose_path)
        elif self.config.docker_compose_path:
            self._compose_path = Path(self.config.docker_compose_path)
        else:
            # Default to package directory
            self._compose_path = _PACKAGE_DIR
        
        self.auto_start = auto_start
        self.wait_for_ready = wait_for_ready
        self.idle_timeout = idle_timeout
        
        self._client: HunyuanOCRClientSync | None = None
        self._client_wrapper: _ActivityTrackingClient | None = None
        self._started_by_us = False
        self._watchdog: IdleWatchdog | None = None
        
        # Setup idle watchdog if timeout specified
        if idle_timeout is not None and idle_timeout > 0:
            self._watchdog = IdleWatchdog(
                idle_timeout=idle_timeout,
                check_interval=idle_check_interval,
                on_idle=self._on_idle_shutdown,
            )
    
    def _on_idle_shutdown(self) -> None:
        """Called by watchdog when idle timeout reached."""
        _log.info("Auto-shutdown triggered due to inactivity")
        self.stop()
    
    def __enter__(self) -> "HunyuanOCRManager":
        if self.auto_start:
            self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._watchdog:
            self._watchdog.stop()
        if self._started_by_us:
            self.stop()
        if self._client:
            self._client.close()
    
    @property
    def client(self) -> "HunyuanOCRClientSync | _ActivityTrackingClient":
        """Get the sync OCR client.
        
        If idle_timeout is set, returns a wrapper that tracks activity.
        """
        if self._client is None:
            self._client = HunyuanOCRClientSync(self.config)
            self._client.connect()
            
            # Wrap with activity tracking if watchdog is enabled
            if self._watchdog:
                self._client_wrapper = _ActivityTrackingClient(
                    self._client,
                    self._watchdog,
                )
        
        if self._client_wrapper:
            return self._client_wrapper
        return self._client
    
    def _run_docker_compose(self, *args: str) -> subprocess.CompletedProcess:
        """Run docker compose command."""
        cmd = ["docker", "compose", *args]
        _log.debug(f"Running: {' '.join(cmd)} in {self._compose_path}")
        
        return subprocess.run(
            cmd,
            cwd=self._compose_path,
            capture_output=True,
            text=True,
        )
    
    def is_running(self) -> bool:
        """Check if the HunyuanOCR container is running."""
        try:
            result = subprocess.run(
                ["docker", "container", "inspect", "-f", "{{.State.Running}}", self.config.container_name],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:
            return False
    
    def get_gpu_memory_usage(self) -> dict[str, int] | None:
        """Get current GPU memory usage (requires nvidia-smi).
        
        Returns:
            Dict with 'used' and 'total' in MB, or None if unavailable.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(", ")
                return {"used": int(used), "total": int(total)}
        except Exception:
            pass
        return None
    
    def start(self, wait: bool | None = None) -> bool:
        """Start the HunyuanOCR server.
        
        This will load the model into GPU memory (takes 2-5 minutes).
        
        Args:
            wait: Override wait_for_ready setting.
            
        Returns:
            True if server is running and healthy.
        """
        wait = wait if wait is not None else self.wait_for_ready
        
        # Check if already running
        if self.is_running():
            _log.info("HunyuanOCR server already running")
            if wait:
                return self.client.wait_for_ready()
            return True
        
        _log.info("Starting HunyuanOCR server...")
        gpu_before = self.get_gpu_memory_usage()
        if gpu_before:
            _log.info(f"GPU memory before start: {gpu_before['used']}MB / {gpu_before['total']}MB")
        
        result = self._run_docker_compose("up", "-d", "hunyuan-ocr")
        
        if result.returncode != 0:
            _log.error(f"Failed to start server: {result.stderr}")
            return False
        
        self._started_by_us = True
        
        if wait:
            _log.info("Waiting for server to be ready (model loading, may take 2-5 minutes)...")
            ready = self.client.wait_for_ready()
            if ready:
                gpu_after = self.get_gpu_memory_usage()
                if gpu_after:
                    _log.info(f"GPU memory after start: {gpu_after['used']}MB / {gpu_after['total']}MB")
                _log.info("HunyuanOCR server is ready")
                
                # Start idle watchdog if configured
                if self._watchdog:
                    self._watchdog.start()
                    _log.info(f"Idle watchdog started (timeout: {self.idle_timeout}s)")
            else:
                _log.warning("Server did not become ready within timeout")
            return ready
        
        # Start watchdog even if not waiting (assume server is ready)
        if self._watchdog:
            self._watchdog.start()
        
        return True
    
    def stop(self) -> bool:
        """Stop the HunyuanOCR server to free VRAM.
        
        This fully releases GPU memory used by the model.
        
        Returns:
            True if server was stopped successfully.
        """
        _log.info("Stopping HunyuanOCR server to free VRAM...")
        gpu_before = self.get_gpu_memory_usage()
        
        result = self._run_docker_compose("stop", "hunyuan-ocr")
        
        if result.returncode != 0:
            _log.error(f"Failed to stop server: {result.stderr}")
            return False
        
        self._started_by_us = False
        
        # Wait a moment for GPU memory to be released
        time.sleep(2)
        
        gpu_after = self.get_gpu_memory_usage()
        if gpu_before and gpu_after:
            freed = gpu_before['used'] - gpu_after['used']
            _log.info(f"VRAM freed: {freed}MB (was {gpu_before['used']}MB, now {gpu_after['used']}MB)")
        else:
            _log.info("HunyuanOCR server stopped, VRAM freed")
        
        return True
    
    def restart(self, wait: bool | None = None) -> bool:
        """Restart the HunyuanOCR server.
        
        Args:
            wait: Override wait_for_ready setting.
            
        Returns:
            True if server restarted and is healthy.
        """
        self.stop()
        return self.start(wait=wait)
    
    def start_with_autoshutdown(
        self,
        idle_timeout: float = 300,
        check_interval: float = 30,
    ) -> bool:
        """Start server with auto-shutdown enabled.
        
        Convenience method to start with idle timeout if not set in constructor.
        
        Args:
            idle_timeout: Seconds of inactivity before shutdown.
            check_interval: How often to check for idle.
            
        Returns:
            True if server started successfully.
        """
        if self._watchdog is None:
            self._watchdog = IdleWatchdog(
                idle_timeout=idle_timeout,
                check_interval=check_interval,
                on_idle=self._on_idle_shutdown,
            )
            self.idle_timeout = idle_timeout
        return self.start()


class _ActivityTrackingClient:
    """Wrapper that tracks activity for idle watchdog."""
    
    def __init__(self, client: HunyuanOCRClientSync, watchdog: IdleWatchdog):
        self._client = client
        self._watchdog = watchdog
    
    def __getattr__(self, name: str):
        """Forward all attribute access to wrapped client."""
        return getattr(self._client, name)
    
    def ocr_image(self, *args, **kwargs):
        """OCR with activity tracking."""
        self._watchdog.record_activity()
        result = self._client.ocr_image(*args, **kwargs)
        self._watchdog.record_activity()
        return result
    
    def ocr_batch(self, *args, **kwargs):
        """Batch OCR with activity tracking."""
        self._watchdog.record_activity()
        result = self._client.ocr_batch(*args, **kwargs)
        self._watchdog.record_activity()
        return result
    
    def ocr_batch_with_callback(self, *args, **kwargs):
        """Batch OCR with callbacks and activity tracking."""
        self._watchdog.record_activity()
        result = self._client.ocr_batch_with_callback(*args, **kwargs)
        self._watchdog.record_activity()
        return result


@contextmanager
def hunyuan_ocr_session(
    docker_compose_path: Path | str | None = None,
    config: HunyuanOCRConfig | None = None,
    stop_on_exit: bool = True,
    idle_timeout: float | None = None,
) -> "Iterator[Union[HunyuanOCRClientSync, _ActivityTrackingClient]]":
    """Context manager that manages OCR server lifecycle.
    
    Starts the server on entry, yields a client, and optionally stops
    the server on exit to free VRAM.
    
    Args:
        docker_compose_path: Path to docker-compose.yml directory.
        config: Client configuration.
        stop_on_exit: If True, stop server when exiting context.
        idle_timeout: If set, auto-stop after this many seconds of inactivity.
        
    Yields:
        HunyuanOCRClientSync instance (or activity-tracking wrapper if idle_timeout set).
        
    Example:
        with hunyuan_ocr_session() as client:
            result = client.ocr_image("page.png")
        # Server stopped, VRAM freed
        
    Example - Auto-shutdown after 5 min idle:
        with hunyuan_ocr_session(idle_timeout=300, stop_on_exit=False) as client:
            result = client.ocr_image("page.png")
        # Server will auto-stop after 5 min of no requests
    """
    manager = HunyuanOCRManager(
        config=config,
        docker_compose_path=docker_compose_path,
        auto_start=True,
        wait_for_ready=True,
        idle_timeout=idle_timeout,
    )
    
    with manager:
        yield manager.client
        
        if not stop_on_exit:
            manager._started_by_us = False


@asynccontextmanager
async def hunyuan_ocr_session_async(
    docker_compose_path: Path | str | None = None,
    config: HunyuanOCRConfig | None = None,
    stop_on_exit: bool = True,
) -> "AsyncIterator[HunyuanOCRClient]":
    """Async context manager for OCR server lifecycle.
    
    Args:
        docker_compose_path: Path to docker-compose.yml directory.
        config: Client configuration.
        stop_on_exit: If True, stop server when exiting context.
        
    Yields:
        HunyuanOCRClient instance (async).
    """
    manager = HunyuanOCRManager(
        config=config,
        docker_compose_path=docker_compose_path,
        auto_start=True,
        wait_for_ready=True,
    )
    
    manager.start()
    
    config = config or HunyuanOCRConfig()
    async with HunyuanOCRClient(config) as client:
        yield client
    
    if stop_on_exit:
        manager.stop()

