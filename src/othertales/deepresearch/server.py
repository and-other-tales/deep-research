"""
Server module for the open_deep_research application.

This module provides:
1. A FastAPI server for exposing the research capabilities as an API
2. Dynamic port allocation to handle port conflicts
3. Integration with LangGraph for running research workflows
"""

import os
import sys
import logging
import signal
import threading
import time
import json
import asyncio
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, Body, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

from othertales.deepresearch.configuration import Configuration
from othertales.deepresearch.port_utils import (
    get_port_from_env, 
    is_port_available,
    find_available_port
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.environ.get('LOG_FILE', '/var/log/open_deep_research.log'), 
            mode='a'
        )
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Open Deep Research API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get configuration with default values
config = Configuration()

# Dynamically allocate ports using the port_utils module
SERVER_PORT = get_port_from_env(
    "SERVER_PORT", 
    config.server_port, 
    fallback_min=8000, 
    fallback_max=9000
)

LANGGRAPH_PORT = get_port_from_env(
    "LANGGRAPH_PORT", 
    config.langgraph_port, 
    fallback_min=8001, 
    fallback_max=9001
)

# Use the host from configuration
SERVER_HOST = config.server_host
INTERNAL_HOST = config.internal_host

# Build the URLs
LANGGRAPH_URL = f"http://{INTERNAL_HOST}:{LANGGRAPH_PORT}"

# Redirect URL for root path
DEFAULT_LANGSMITH_URL = "https://smith.langchain.com/studio/"
LANGSMITH_URL = os.environ.get("LANGSMITH_URL", DEFAULT_LANGSMITH_URL)
BASE_URL = os.environ.get("BASE_URL", f"http://{SERVER_HOST}:{SERVER_PORT}")
ORG_ID = os.environ.get("LANGSMITH_ORG_ID", "")
REDIRECT_URL = os.environ.get(
    "REDIRECT_URL", 
    f"{LANGSMITH_URL}?baseUrl={BASE_URL}" + (f"&organizationId={ORG_ID}" if ORG_ID else "")
)

# LangGraph process
langgraph_process: Optional[asyncio.subprocess.Process] = None

def start_langgraph_server():
    """Start LangGraph server in a separate process."""
    global LANGGRAPH_PORT
    
    # Get configuration from environment variables with fallbacks
    langgraph_executable = os.environ.get("LANGGRAPH_EXECUTABLE", "langgraph")
    langgraph_command = os.environ.get("LANGGRAPH_COMMAND", "dev")
    langgraph_log_file = os.environ.get("LANGGRAPH_LOG_FILE", "/var/log/langgraph.log")
    
    # Build command with parameters
    cmd = [
        langgraph_executable, 
        langgraph_command, 
        "--port", str(LANGGRAPH_PORT), 
        "--host", INTERNAL_HOST
    ]
    
    # Add any additional arguments from environment
    extra_args = os.environ.get("LANGGRAPH_EXTRA_ARGS", "")
    if extra_args:
        cmd.extend(extra_args.split())
    
    logger.info(f"Starting LangGraph server: {' '.join(cmd)}")
    
    try:
        # Use subprocess.Popen to start the server in the background
        import subprocess
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )
        
        # Start a thread to monitor and log LangGraph output
        def log_output(process):
            for line in iter(process.stdout.readline, ""):
                logger.info(f"[LangGraph] {line.strip()}")
                # Also write to log file if configured
                if langgraph_log_file:
                    try:
                        with open(langgraph_log_file, "a") as f:
                            f.write(f"{line}\n")
                    except Exception as e:
                        logger.error(f"Error writing to log file: {e}")
        
        log_thread = threading.Thread(target=log_output, args=(process,), daemon=True)
        log_thread.start()
        
        # Wait for LangGraph server to start up
        max_retries = 10
        for i in range(max_retries):
            logger.info(f"Waiting for LangGraph server to start (attempt {i+1}/{max_retries})...")
            time.sleep(2)  # Wait for a while
            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(f"{LANGGRAPH_URL}/health")
                    if response.status_code == 200:
                        logger.info("LangGraph server started successfully.")
                        break
            except Exception:
                # Continue to next retry
                pass
            
            # If this is the last retry and we still can't connect, the port might be in use
            # but not responding to our health check
            if i == max_retries - 1:
                logger.warning("LangGraph server health check failed after maximum retries.")
                
        # Return the process so it can be terminated later if needed
        return process
    except Exception as e:
        logger.error(f"Error starting LangGraph server: {e}")
        return None

@app.get("/", response_class=RedirectResponse, status_code=301)
async def root():
    """Redirect root path to LangSmith dashboard."""
    return REDIRECT_URL

@app.post("/api/research")
async def process_research(
    topic: str = Body(..., embed=True),
    instruction: Optional[str] = Body(None, embed=True),
    mode: Optional[str] = Body("general", embed=True),
):
    """Process a research topic and start the research workflow.
    
    This endpoint provides a direct way to submit a topic for research
    without going through the LangGraph API.
    
    Args:
        topic: The research topic
        instruction: Optional specific instruction for the research
        mode: Mode of operation - "legal", "tax", or "general" (default)
        
    Returns:
        JSON response with process ID and status
    """
    # Validate mode
    if mode not in ["legal", "tax", "general"]:
        return JSONResponse(
            content={
                "error": "Invalid mode",
                "detail": "Mode must be one of: legal, tax, general"
            },
            status_code=400
        )
    
    async with httpx.AsyncClient() as client:
        # Prepare the payload for LangGraph
        payload = {
            "input": {
                "topic": topic,
                "mode": mode,
                "instruction": instruction
            }
        }
        
        # Forward to LangGraph server
        try:
            response = await client.post(
                f"{LANGGRAPH_URL}/open_deep_research/process",
                json=payload,
                timeout=10.0
            )
            
            # Return the response
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.RequestError as e:
            logger.error(f"Error connecting to LangGraph server: {e}")
            return JSONResponse(
                content={
                    "error": "Failed to connect to LangGraph server",
                    "detail": str(e)
                },
                status_code=503
            )

@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check if LangGraph server is reachable
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LANGGRAPH_URL}/health", timeout=2.0)
            if response.status_code == 200:
                return {"status": "ok", "langgraph": "ok"}
            else:
                return {"status": "ok", "langgraph": "unhealthy", "langgraph_status": response.status_code}
    except Exception as e:
        return {"status": "ok", "langgraph": "unreachable", "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize LangGraph server on startup."""
    global langgraph_process
    
    # Start LangGraph server
    langgraph_process = start_langgraph_server()
    
    if langgraph_process:
        logger.info("LangGraph server started.")
    else:
        logger.error("Failed to start LangGraph server.")

@app.on_event("shutdown")
async def shutdown_event():
    """Shut down LangGraph server on shutdown."""
    global langgraph_process
    if langgraph_process:
        langgraph_process.terminate()
        try:
            langgraph_process.wait(timeout=5)
            logger.info("LangGraph server terminated.")
        except:
            # Force kill if not terminated within timeout
            langgraph_process.kill()
            logger.info("LangGraph server forcefully killed.")

# Forward requests to LangGraph server using a middleware
@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    """Forward requests to LangGraph server, except for root path and API endpoints."""
    # Skip root path, health check, and API endpoints as they're handled directly
    if request.url.path == "/" or request.url.path == "/health" or request.url.path.startswith("/api/"):
        return await call_next(request)
    
    # Forward request to LangGraph server
    async with httpx.AsyncClient(timeout=120.0) as client:  # Set a longer client-level timeout
        url = f"{LANGGRAPH_URL}{request.url.path}"
        
        if request.query_params:
            url = f"{url}?{request.query_params}"
        
        try:
            # Forward request body and headers
            body = await request.body()
            headers = dict(request.headers)
            # Remove host header as it will be set by httpx
            headers.pop("host", None)
            
            logger.debug(f"Proxying request to {url} with method {request.method}")
            
            # Forward the request using the same HTTP method
            response = await client.request(
                request.method,
                url,
                headers=headers,
                content=body,
                timeout=120.0,
                follow_redirects=True
            )
            
            logger.debug(f"Received response from {url} with status {response.status_code}")
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type", "application/json")
            )
        except httpx.ConnectError as e:
            error_msg = f"Connection error proxying request to {url}: {str(e)}"
            logger.error(error_msg)
            return Response(
                content=json.dumps({
                    "error": error_msg,
                    "detail": "The LangGraph server may still be starting or is unreachable"
                }),
                status_code=502,  # Bad Gateway
                media_type="application/json"
            )
        except httpx.TimeoutException as e:
            error_msg = f"Timeout error proxying request to {url}: {str(e)}"
            logger.error(error_msg)
            return Response(
                content=json.dumps({
                    "error": error_msg,
                    "detail": "The request to LangGraph server timed out"
                }),
                status_code=504,  # Gateway Timeout
                media_type="application/json"
            )
        except Exception as e:
            error_msg = f"Error proxying request to {url}: {str(e)}"
            logger.error(error_msg)
            return Response(
                content=json.dumps({"error": error_msg}),
                status_code=502,
                media_type="application/json"
            )

def start():
    """Start the FastAPI server."""
    global SERVER_PORT, SERVER_HOST
    
    # Log the ports being used
    logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"LangGraph server on {INTERNAL_HOST}:{LANGGRAPH_PORT}")
    
    # Start the server
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    start()