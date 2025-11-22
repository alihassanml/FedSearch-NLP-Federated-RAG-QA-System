from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.routes import router as api_router
from app.core.config import settings
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Federated RAG QA System for Enterprise Document Search",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates directory
templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["RAG API"])

@app.get("/")
async def root(request: Request):
    """Serve the main frontend interface"""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION
        })
    except Exception as e:
        logger.error(f"Error loading template: {str(e)}")
        # Fallback to JSON response if template not found
        from fastapi.responses import JSONResponse
        return JSONResponse({
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "message": "Frontend template not found. Please create templates/index.html",
            "docs": "/docs",
            "endpoints": {
                "health": "/api/health",
                "query": "/api/query",
                "index": "/api/index",
                "stats": "/api/documents/stats"
            }
        })

@app.get("/health-json")
async def health_json():
    """JSON health endpoint (for monitoring)"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Retriever Model: {settings.RETRIEVER_MODEL}")
    logger.info(f"Generator Model: {settings.GENERATOR_MODEL}")
    logger.info(f"Templates directory: {templates_dir}")
    logger.info("System ready to accept requests")
    logger.info(f"Frontend available at: http://{settings.HOST}:{settings.PORT}/")
    logger.info(f"API docs available at: http://{settings.HOST}:{settings.PORT}/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down gracefully...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )