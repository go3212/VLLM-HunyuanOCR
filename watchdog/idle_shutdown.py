#!/usr/bin/env python3
"""Idle shutdown watchdog for vLLM server.

Monitors the vLLM server for inactivity and triggers shutdown after a timeout.
This frees GPU VRAM when the server isn't being used.

The watchdog checks:
1. /metrics endpoint for request counts
2. Falls back to tracking time since last successful health check

Environment variables:
    VLLM_URL: vLLM server URL (default: http://localhost:8000)
    IDLE_TIMEOUT: Seconds of inactivity before shutdown (default: 300 = 5 min)
    CHECK_INTERVAL: Seconds between activity checks (default: 30)
    SHUTDOWN_COMMAND: Command to run on shutdown (default: kill vLLM process)
"""

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import NamedTuple

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class ServerMetrics(NamedTuple):
    """Parsed metrics from vLLM."""
    
    requests_total: int
    requests_running: int
    is_healthy: bool


def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(name, "")
    if value.strip():
        try:
            return int(value)
        except ValueError:
            log.warning(f"Invalid {name}={value}, using default {default}")
    return default


# Configuration from environment
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
IDLE_TIMEOUT = get_env_int("IDLE_TIMEOUT", 300)  # 5 minutes default
CHECK_INTERVAL = get_env_int("CHECK_INTERVAL", 30)  # 30 seconds default
SHUTDOWN_COMMAND = os.environ.get("SHUTDOWN_COMMAND", "")


def parse_prometheus_metric(text: str, metric_name: str) -> float | None:
    """Parse a single metric value from Prometheus format."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # Format: metric_name{labels} value or metric_name value
        if line.startswith(metric_name):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Handle metric_name{...} value format
                    value_str = parts[-1]
                    return float(value_str)
                except (ValueError, IndexError):
                    continue
    return None


def get_server_metrics(client: httpx.Client) -> ServerMetrics | None:
    """Fetch and parse vLLM server metrics."""
    try:
        # Try metrics endpoint first (Prometheus format)
        response = client.get("/metrics", timeout=10)
        if response.status_code == 200:
            text = response.text
            
            # Look for request-related metrics
            # vLLM exposes: vllm:num_requests_running, vllm:num_requests_waiting
            # and counters like vllm:request_success_total
            requests_running = parse_prometheus_metric(text, "vllm:num_requests_running")
            requests_total = parse_prometheus_metric(text, "vllm:request_success_total")
            
            if requests_total is not None:
                return ServerMetrics(
                    requests_total=int(requests_total),
                    requests_running=int(requests_running or 0),
                    is_healthy=True,
                )
        
        # Fallback: just check health
        health_response = client.get("/health", timeout=5)
        if health_response.status_code == 200:
            return ServerMetrics(
                requests_total=-1,  # Unknown
                requests_running=0,
                is_healthy=True,
            )
        
        return None
        
    except Exception as e:
        log.debug(f"Failed to get metrics: {e}")
        return None


def shutdown_server():
    """Shutdown the vLLM server to free VRAM."""
    log.info("🛑 Initiating server shutdown to free VRAM...")
    
    if SHUTDOWN_COMMAND:
        log.info(f"Running shutdown command: {SHUTDOWN_COMMAND}")
        try:
            subprocess.run(SHUTDOWN_COMMAND, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            log.error(f"Shutdown command failed: {e}")
    else:
        # Default: send SIGTERM to PID 1 (main process in container)
        # This gracefully stops vLLM
        log.info("Sending SIGTERM to vLLM process...")
        try:
            # Find vLLM process
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for pid in result.stdout.strip().split("\n"):
                    if pid:
                        log.info(f"Killing vLLM process {pid}")
                        os.kill(int(pid), signal.SIGTERM)
            else:
                # Fallback: signal ourselves to stop (if running as PID 1)
                log.info("No vLLM process found, signaling container stop")
                sys.exit(0)
        except Exception as e:
            log.error(f"Failed to kill vLLM: {e}")
            sys.exit(1)


def main():
    """Main watchdog loop."""
    log.info("=" * 60)
    log.info("vLLM Idle Shutdown Watchdog")
    log.info("=" * 60)
    log.info(f"  Server URL: {VLLM_URL}")
    log.info(f"  Idle timeout: {IDLE_TIMEOUT}s ({IDLE_TIMEOUT // 60}m)")
    log.info(f"  Check interval: {CHECK_INTERVAL}s")
    log.info("=" * 60)
    
    client = httpx.Client(base_url=VLLM_URL)
    
    last_activity_time = time.monotonic()
    last_request_count = -1
    
    # Wait for server to be ready
    log.info("Waiting for vLLM server to be ready...")
    while True:
        metrics = get_server_metrics(client)
        if metrics and metrics.is_healthy:
            log.info("✓ Server is ready, starting idle monitoring")
            last_request_count = metrics.requests_total
            last_activity_time = time.monotonic()
            break
        time.sleep(5)
    
    # Main monitoring loop
    while True:
        time.sleep(CHECK_INTERVAL)
        
        metrics = get_server_metrics(client)
        
        if metrics is None:
            log.warning("Server not responding, may have crashed")
            continue
        
        # Check if there's been activity
        current_time = time.monotonic()
        
        if metrics.requests_running > 0:
            # Active requests
            log.debug(f"Active: {metrics.requests_running} requests running")
            last_activity_time = current_time
            last_request_count = metrics.requests_total
            
        elif metrics.requests_total > last_request_count:
            # New requests completed since last check
            new_requests = metrics.requests_total - last_request_count
            log.debug(f"Activity: {new_requests} new requests completed")
            last_activity_time = current_time
            last_request_count = metrics.requests_total
            
        else:
            # No activity
            idle_seconds = current_time - last_activity_time
            idle_remaining = IDLE_TIMEOUT - idle_seconds
            
            if idle_remaining <= 0:
                log.warning(f"Server idle for {idle_seconds:.0f}s (timeout: {IDLE_TIMEOUT}s)")
                shutdown_server()
                break
            else:
                log.info(f"Idle: {idle_seconds:.0f}s / {IDLE_TIMEOUT}s (shutdown in {idle_remaining:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Watchdog interrupted")
        sys.exit(0)

