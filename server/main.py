"""
FastAPI server for Data Smith - Dataset Generation Tool.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Literal
import logging
import traceback
import uvicorn

from agent import DatasetAgent
from errors import GenerationError
from formats import GenerateResponse
from model_factory import ModelFactory
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, str(settings.get("app", {}).get("log_level", "INFO")).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_smith")


# Initialize FastAPI app
app_cfg = settings.get("app", {})
app = FastAPI(
    title=app_cfg.get("name", "Data Smith API"),
    description="Generate fine-tuning datasets from text files using LangChain and Ollama",
    version=app_cfg.get("version", "1.0.0")
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("cors", {}).get("origins", []),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def catch_errors_middleware(request: Request, call_next):
    """Log unhandled exceptions and return a structured JSON error response."""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(
            "Unhandled error on %s %s: %s\n%s",
            request.method,
            request.url.path,
            str(exc),
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": "Internal server error"},
        )


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError):
    """Surface generation failures as a clean 502 with a friendly message."""
    logger.warning("Generation error: %s | detail=%s", exc.user_message, exc.detail)
    return JSONResponse(
        status_code=502,
        content={"success": False, "detail": exc.user_message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a flat, consistent validation error shape."""
    return JSONResponse(
        status_code=422,
        content={"success": False, "detail": "Validation failed", "errors": exc.errors()},
    )

# Initialize the dataset agent using the model configured in config.toml
agent = DatasetAgent(llm=ModelFactory().create())


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Data Smith API",
        "version": "1.0.0",
        "endpoints": {
            "/api/health": "Health check",
            "/api/generate": "Generate dataset from text file"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    llm_cfg = settings.get("llm", {})
    return {
        "status": "healthy",
        "provider": llm_cfg.get("provider"),
        "model": llm_cfg.get(llm_cfg.get("provider", ""), {}).get("model"),
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_dataset(
    file: UploadFile = File(...),
    format_type: Literal["alpaca", "chat", "completion"] = Form(...),
    num_samples: int = Form(default=5)
):
    """
    Generate a dataset from an uploaded text file.
    
    Args:
        file: Text file to process (.txt)
        format_type: Output format - 'alpaca', 'chat', or 'completion'
        num_samples: Number of samples to generate (default: 5)
        
    Returns:
        Generated dataset in the specified format
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    # Validate num_samples
    if num_samples < 1 or num_samples > 1000:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 1000"
        )
    
    try:
        # Read file content
        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be UTF-8 encoded text"
            )
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        logger.info(
            "Generating dataset: file=%s format=%s samples=%d chars=%d",
            file.filename, format_type, num_samples, len(text)
        )
        
        # Generate dataset
        data = await agent.generate(
            text=text,
            format_type=format_type,
            num_samples=num_samples
        )
        
        logger.info("Generated %d samples in %s format", len(data), format_type)
        
        return GenerateResponse(
            success=True,
            format_type=format_type,
            data=data,
            message=f"Generated {len(data)} samples in {format_type} format"
        )
        
    except HTTPException:
        raise
    except GenerationError:
        raise
    except Exception as e:
        logger.error("Dataset generation failed: %s\n%s", str(e), traceback.format_exc())
        raise GenerationError(f"Unexpected error: {e}") from e


@app.post("/api/generate-text")
async def generate_from_text(
    text: str = Form(...),
    format_type: Literal["alpaca", "chat", "completion"] = Form(...),
    num_samples: int = Form(default=5)
):
    """
    Generate a dataset from raw text input (no file upload).
    
    Args:
        text: Raw text content to process
        format_type: Output format - 'alpaca', 'chat', or 'completion'
        num_samples: Number of samples to generate (default: 5)
        
    Returns:
        Generated dataset in the specified format
    """
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text content is empty"
        )
    
    if num_samples < 1 or num_samples > 1000:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 1000"
        )
    
    try:
        logger.info(
            "Generating dataset from text: format=%s samples=%d chars=%d",
            format_type, num_samples, len(text)
        )
        data = await agent.generate(
            text=text,
            format_type=format_type,
            num_samples=num_samples
        )
        
        logger.info("Generated %d samples in %s format", len(data), format_type)
        
        return GenerateResponse(
            success=True,
            format_type=format_type,
            data=data,
            message=f"Generated {len(data)} samples in {format_type} format"
        )
        
    except HTTPException:
        raise
    except GenerationError:
        raise
    except Exception as e:
        logger.error("Dataset generation failed: %s\n%s", str(e), traceback.format_exc())
        raise GenerationError(f"Unexpected error: {e}") from e


if __name__ == "__main__":
    app_cfg = settings.get("app", {})
    reload_val = str(app_cfg.get("reload", "true")).lower() in ("1", "true", "yes")
    uvicorn.run(
        "main:app",
        host=str(app_cfg.get("host", "0.0.0.0")),
        port=int(app_cfg.get("port", 8000)),
        reload=reload_val,
    )
