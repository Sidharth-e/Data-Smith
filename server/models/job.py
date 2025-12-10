"""
Job Configuration Models for Data Smith.
Defines job types, status, and queue management.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


class JobType(str, Enum):
    """Types of jobs that can be processed."""
    DOCUMENT_PROCESS = "document_process"
    BROWSER_SCRAPE = "browser_scrape"
    DATASET_EXPORT = "dataset_export"
    BATCH_PROCESS = "batch_process"


class JobStatus(str, Enum):
    """Status of a job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10


class JobCreate(BaseModel):
    """Request to create a new job."""
    job_type: JobType
    params: Dict[str, Any] = Field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    """A processing job."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    
    # Params and results
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Progress tracking
    progress: float = 0.0
    progress_message: str = ""
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at:
            end = self.completed_at or datetime.now()
            return (end - self.started_at).total_seconds()
        return None


class JobUpdate(BaseModel):
    """Update to a job's status."""
    status: Optional[JobStatus] = None
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """Response containing list of jobs."""
    jobs: List[Job]
    total: int
    page: int = 1
    page_size: int = 20
