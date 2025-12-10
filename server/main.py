"""
FastAPI server for Data Smith - Dataset Generation Tool.
Enhanced with multi-format document processing and multi-agent orchestration.
"""

import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Literal, Optional, List
from pydantic import BaseModel
import uvicorn

# Add server directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import DatasetAgent
from formats import GenerateResponse

# Import new services (with fallback for missing dependencies)
try:
    from services.document_parser import DocumentParser, parse_document, DocumentParserError
    from services.document_splitter import DocumentSplitter, SplitStrategy, SplitterConfig
    DOCUMENT_SERVICES_AVAILABLE = True
except ImportError as e:
    DOCUMENT_SERVICES_AVAILABLE = False
    print(f"Warning: Document services not available: {e}")

try:
    from agents.supervisor_agent import SupervisorAgent, TaskType
    from agents.worker_agents import WorkerAgentFactory
    AGENT_SERVICES_AVAILABLE = True
except ImportError as e:
    AGENT_SERVICES_AVAILABLE = False
    print(f"Warning: Agent services not available: {e}")


# ============================================
# Pydantic Models for New Endpoints
# ============================================

class DocumentProcessRequest(BaseModel):
    """Request for document processing."""
    format_type: Literal["alpaca", "chat", "completion"] = "alpaca"
    num_samples_per_chunk: int = 2
    split_strategy: Literal["character", "token", "semantic", "sliding_window", "paragraph", "sentence"] = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    task_type: Literal["qa_generation", "summarization", "entity_extraction", "classification", "key_points"] = "qa_generation"


class DocumentProcessResponse(BaseModel):
    """Response from document processing."""
    success: bool
    format_type: str
    data: List[dict]
    metadata: dict = {}
    message: str = ""


class ChunkPreviewResponse(BaseModel):
    """Response with chunk preview."""
    success: bool
    total_chunks: int
    preview_chunks: List[dict]
    estimated_samples: int
    message: str = ""


# ============================================
# Initialize FastAPI App
# ============================================

app = FastAPI(
    title="Data Smith API",
    description="Generate fine-tuning datasets from documents using multi-agent orchestration",
    version="2.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
agent = DatasetAgent(model_name="mistral:7b-instruct")
supervisor = SupervisorAgent(model_name="mistral:7b-instruct") if AGENT_SERVICES_AVAILABLE else None
document_parser = DocumentParser() if DOCUMENT_SERVICES_AVAILABLE else None


# ============================================
# Health & Info Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Data Smith API",
        "version": "2.0.0",
        "features": {
            "document_services": DOCUMENT_SERVICES_AVAILABLE,
            "agent_services": AGENT_SERVICES_AVAILABLE
        },
        "endpoints": {
            "/api/health": "Health check",
            "/api/generate": "Generate from text file (legacy)",
            "/api/generate-text": "Generate from text input (legacy)",
            "/api/v1/documents/process": "Process multi-format document",
            "/api/v1/documents/preview-chunks": "Preview document chunks",
            "/api/v1/documents/supported-formats": "List supported formats"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "mistral:7b-instruct",
        "features": {
            "document_processing": DOCUMENT_SERVICES_AVAILABLE,
            "multi_agent": AGENT_SERVICES_AVAILABLE
        }
    }


@app.get("/api/v1/documents/supported-formats")
async def get_supported_formats():
    """Get list of supported document formats."""
    return {
        "formats": [
            {"extension": ".txt", "name": "Plain Text", "available": True},
            {"extension": ".pdf", "name": "PDF Document", "available": DOCUMENT_SERVICES_AVAILABLE},
            {"extension": ".docx", "name": "Word Document", "available": DOCUMENT_SERVICES_AVAILABLE},
            {"extension": ".csv", "name": "CSV Spreadsheet", "available": DOCUMENT_SERVICES_AVAILABLE},
            {"extension": ".xlsx", "name": "Excel Spreadsheet", "available": DOCUMENT_SERVICES_AVAILABLE},
        ],
        "split_strategies": ["character", "token", "semantic", "sliding_window", "paragraph", "sentence"],
        "task_types": ["qa_generation", "summarization", "entity_extraction", "classification", "key_points"],
        "output_formats": ["alpaca", "chat", "completion"]
    }


# ============================================
# Legacy Endpoints (Backward Compatibility)
# ============================================

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_dataset(
    file: UploadFile = File(...),
    format_type: Literal["alpaca", "chat", "completion"] = Form(...),
    num_samples: int = Form(default=5)
):
    """
    Generate a dataset from an uploaded text file (Legacy endpoint).
    """
    # Validate file type
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported in legacy endpoint. Use /api/v1/documents/process for other formats."
        )
    
    if num_samples < 1 or num_samples > 50:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 50"
        )
    
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty")
        
        data = await agent.generate(
            text=text,
            format_type=format_type,
            num_samples=num_samples
        )
        
        return GenerateResponse(
            success=True,
            format_type=format_type,
            data=data,
            message=f"Generated {len(data)} samples in {format_type} format"
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        return GenerateResponse(
            success=False,
            format_type=format_type,
            data=[],
            message=f"Error generating dataset: {str(e)}"
        )


@app.post("/api/generate-text")
async def generate_from_text(
    text: str = Form(...),
    format_type: Literal["alpaca", "chat", "completion"] = Form(...),
    num_samples: int = Form(default=5)
):
    """Generate a dataset from raw text input (Legacy endpoint)."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content is empty")
    
    if num_samples < 1 or num_samples > 50:
        raise HTTPException(status_code=400, detail="num_samples must be between 1 and 50")
    
    try:
        data = await agent.generate(
            text=text,
            format_type=format_type,
            num_samples=num_samples
        )
        
        return GenerateResponse(
            success=True,
            format_type=format_type,
            data=data,
            message=f"Generated {len(data)} samples in {format_type} format"
        )
        
    except Exception as e:
        return GenerateResponse(
            success=False,
            format_type=format_type,
            data=[],
            message=f"Error generating dataset: {str(e)}"
        )


# ============================================
# New Document Processing Endpoints
# ============================================

@app.post("/api/v1/documents/preview-chunks", response_model=ChunkPreviewResponse)
async def preview_document_chunks(
    file: UploadFile = File(...),
    split_strategy: Literal["character", "token", "semantic", "sliding_window", "paragraph", "sentence"] = Form(default="semantic"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=100),
    samples_per_chunk: int = Form(default=2)
):
    """
    Preview how a document will be chunked without processing.
    
    Returns:
        Preview of chunks and estimated output samples
    """
    if not DOCUMENT_SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Document processing services not available. Install required dependencies."
        )
    
    try:
        # Read and parse document
        content = await file.read()
        doc = await document_parser.parse(
            file_content=content,
            filename=file.filename
        )
        
        # Split into chunks
        config = SplitterConfig(
            strategy=SplitStrategy(split_strategy),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splitter = DocumentSplitter(config)
        chunks = splitter.split(doc.text)
        
        # Create preview (first 5 chunks)
        preview = [
            {
                "chunk_id": c.chunk_id,
                "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "char_count": c.char_count,
                "word_count": c.word_count
            }
            for c in chunks[:5]
        ]
        
        return ChunkPreviewResponse(
            success=True,
            total_chunks=len(chunks),
            preview_chunks=preview,
            estimated_samples=len(chunks) * samples_per_chunk,
            message=f"Document will be split into {len(chunks)} chunks"
        )
        
    except DocumentParserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/api/v1/documents/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    format_type: Literal["alpaca", "chat", "completion"] = Form(default="alpaca"),
    num_samples_per_chunk: int = Form(default=2),
    split_strategy: Literal["character", "token", "semantic", "sliding_window", "paragraph", "sentence"] = Form(default="semantic"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=100),
    task_type: Literal["qa_generation", "summarization", "entity_extraction", "classification", "key_points"] = Form(default="qa_generation")
):
    """
    Process a document using multi-agent orchestration.
    
    Supports PDF, DOCX, CSV, XLSX, and TXT files.
    
    Args:
        file: Document file to process
        format_type: Output format (alpaca, chat, completion)
        num_samples_per_chunk: Samples to generate per chunk
        split_strategy: How to split the document
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        task_type: Type of dataset generation task
        
    Returns:
        Generated dataset with metadata
    """
    if not DOCUMENT_SERVICES_AVAILABLE or not AGENT_SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Document processing services not available. Install required dependencies."
        )
    
    # Validate parameters
    if num_samples_per_chunk < 1 or num_samples_per_chunk > 10:
        raise HTTPException(status_code=400, detail="num_samples_per_chunk must be between 1 and 10")
    
    if chunk_size < 100 or chunk_size > 5000:
        raise HTTPException(status_code=400, detail="chunk_size must be between 100 and 5000")
    
    try:
        # Parse document
        content = await file.read()
        doc = await document_parser.parse(
            file_content=content,
            filename=file.filename
        )
        
        # Split into chunks
        config = SplitterConfig(
            strategy=SplitStrategy(split_strategy),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splitter = DocumentSplitter(config)
        chunks = splitter.split(doc.text)
        
        if not chunks:
            return DocumentProcessResponse(
                success=False,
                format_type=format_type,
                data=[],
                message="No content could be extracted from the document"
            )
        
        # Process with supervisor agent
        task = TaskType(task_type)
        data = await supervisor.process_document(
            chunks=chunks,
            task_type=task,
            output_format=format_type,
            samples_per_chunk=num_samples_per_chunk
        )
        
        return DocumentProcessResponse(
            success=True,
            format_type=format_type,
            data=data,
            metadata={
                "filename": file.filename,
                "file_type": doc.metadata.file_type.value,
                "total_chunks": len(chunks),
                "split_strategy": split_strategy,
                "task_type": task_type,
                "word_count": doc.metadata.word_count,
                "char_count": doc.metadata.char_count
            },
            message=f"Generated {len(data)} samples from {len(chunks)} chunks"
        )
        
    except DocumentParserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return DocumentProcessResponse(
            success=False,
            format_type=format_type,
            data=[],
            message=f"Error processing document: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

