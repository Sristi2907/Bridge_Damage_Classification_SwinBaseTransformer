import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
import uvicorn

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.error_handler import GlobalExceptionMiddleware
from app.api.routes import router as api_router
from app.services.db_service import db_service
from app.core.device_manager import device_manager

# Initialize logging config
setup_logging()
logger = logging.getLogger("infrascan.startup")

frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
    "dist"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    try:
        db_service.initialize_database()
    except Exception as e:
        logger.critical(f"Database failed to initialize: {e}", exc_info=True)
        
    logger.info("Initializing AI device manager...")
    try:
        device_manager.initialize()
        logger.info("Model singleton will lazy-load on first inference request.")
    except Exception as e:
        logger.critical(f"Device manager failed to initialize: {e}", exc_info=True)
        
    yield
    # Shutdown logic if any
    logger.info("Application shutting down. Cleaning memory resources...")
    device_manager.cleanup()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Exception middleware goes first to catch everything
app.add_middleware(GlobalExceptionMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount local storage folder under /static and /data to serve uploaded/processed images
if os.path.exists(settings.LOCAL_STORAGE_DIR):
    app.mount("/static", StaticFiles(directory=settings.LOCAL_STORAGE_DIR), name="static")
    app.mount("/data", StaticFiles(directory=settings.LOCAL_STORAGE_DIR), name="data")
else:
    logger.warning(f"LOCAL_STORAGE_DIR {settings.LOCAL_STORAGE_DIR} does not exist yet. Cannot mount static folders.")

# Mount React static assets if directory exists
assets_dir = os.path.join(frontend_dir, "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
else:
    logger.warning(f"Frontend assets directory {assets_dir} not found. React assets will not be served.")

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Fallback to local app/index.html
        local_index = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(local_index):
            with open(local_index, "r", encoding="utf-8") as f:
                return f.read()
        return HTMLResponse("InfraScan AI is running, but UI assets were not found.", status_code=200)

# Custom 404 handler for SPA routing
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    if (request.url.path.startswith("/api") or 
        request.url.path.startswith("/static") or 
        request.url.path.startswith("/data") or 
        request.url.path.startswith("/assets")):
        return Response(content="Not Found", status_code=404)
        
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
        
    local_index = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(local_index):
        return FileResponse(local_index)
        
    return Response(content="Not Found", status_code=404)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
