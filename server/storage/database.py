"""
Database Models and Session Management for Data Smith.
Uses SQLite for MVP, easily upgradeable to PostgreSQL.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import json

# Database URL - default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storage/datasmith.db")

# For async support with SQLite
if DATABASE_URL.startswith("sqlite"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
else:
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

Base = declarative_base()


# ============================================
# Database Models
# ============================================

class JobModel(Base):
    """Database model for processing jobs."""
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True)
    job_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    priority = Column(Integer, default=5)
    
    # Params stored as JSON
    params = Column(JSON, default=dict)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    
    # Progress
    progress = Column(Float, default=0.0)
    progress_message = Column(String(255), default="")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    user_id = Column(String(36), nullable=True, index=True)


class DatasetModel(Base):
    """Database model for generated datasets."""
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Source info
    source_type = Column(String(50))  # document, browser, text
    source_file = Column(String(255), nullable=True)
    source_url = Column(String(1000), nullable=True)
    
    # Dataset info
    format_type = Column(String(20))  # alpaca, chat, completion
    item_count = Column(Integer, default=0)
    file_path = Column(String(500), nullable=True)  # Path in storage
    file_size = Column(Integer, default=0)
    
    # Generation settings
    settings = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    user_id = Column(String(36), nullable=True, index=True)
    job_id = Column(String(36), nullable=True, index=True)


class DocumentModel(Base):
    """Database model for uploaded documents."""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20))
    file_size = Column(Integer)
    file_path = Column(String(500))  # Path in storage
    
    # Parsed info
    word_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    chunk_count = Column(Integer, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    user_id = Column(String(36), nullable=True, index=True)


# ============================================
# Database Session Management
# ============================================

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        """Initialize database manager."""
        self.database_url = database_url
        
        # Create sync engine for migrations
        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)


# Repository classes for data access

class JobRepository:
    """Repository for job database operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, job_data: Dict[str, Any]) -> JobModel:
        """Create a new job record."""
        job = JobModel(**job_data)
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job
    
    def get(self, job_id: str) -> Optional[JobModel]:
        """Get a job by ID."""
        return self.session.query(JobModel).filter(JobModel.id == job_id).first()
    
    def list(
        self, 
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[JobModel]:
        """List jobs with filters."""
        query = self.session.query(JobModel)
        
        if status:
            query = query.filter(JobModel.status == status)
        if job_type:
            query = query.filter(JobModel.job_type == job_type)
        if user_id:
            query = query.filter(JobModel.user_id == user_id)
        
        return query.order_by(JobModel.created_at.desc()).offset(offset).limit(limit).all()
    
    def update(self, job_id: str, update_data: Dict[str, Any]) -> Optional[JobModel]:
        """Update a job."""
        job = self.get(job_id)
        if not job:
            return None
        
        for key, value in update_data.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        self.session.commit()
        self.session.refresh(job)
        return job
    
    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        job = self.get(job_id)
        if not job:
            return False
        
        self.session.delete(job)
        self.session.commit()
        return True


class DatasetRepository:
    """Repository for dataset database operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, dataset_data: Dict[str, Any]) -> DatasetModel:
        """Create a new dataset record."""
        dataset = DatasetModel(**dataset_data)
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset
    
    def get(self, dataset_id: str) -> Optional[DatasetModel]:
        """Get a dataset by ID."""
        return self.session.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    
    def list(
        self,
        user_id: Optional[str] = None,
        format_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[DatasetModel]:
        """List datasets with filters."""
        query = self.session.query(DatasetModel)
        
        if user_id:
            query = query.filter(DatasetModel.user_id == user_id)
        if format_type:
            query = query.filter(DatasetModel.format_type == format_type)
        
        return query.order_by(DatasetModel.created_at.desc()).offset(offset).limit(limit).all()
    
    def update(self, dataset_id: str, update_data: Dict[str, Any]) -> Optional[DatasetModel]:
        """Update a dataset."""
        dataset = self.get(dataset_id)
        if not dataset:
            return None
        
        for key, value in update_data.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        
        self.session.commit()
        self.session.refresh(dataset)
        return dataset
    
    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        dataset = self.get(dataset_id)
        if not dataset:
            return False
        
        self.session.delete(dataset)
        self.session.commit()
        return True


# Global database manager
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager


def get_db_session() -> Session:
    """Get a database session."""
    return get_db_manager().get_session()
