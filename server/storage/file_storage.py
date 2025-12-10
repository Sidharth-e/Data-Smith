"""
File Storage Service for Data Smith.
Abstract storage interface with local filesystem implementation.
"""

import os
import shutil
import aiofiles
import aiofiles.os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, BinaryIO
from datetime import datetime
from dataclasses import dataclass
import hashlib
import mimetypes


@dataclass
class FileInfo:
    """Information about a stored file."""
    path: str
    filename: str
    size: int
    mime_type: Optional[str]
    created_at: datetime
    modified_at: datetime
    checksum: Optional[str] = None


class FileStorageError(Exception):
    """Exception raised by file storage operations."""
    pass


class FileStorage(ABC):
    """
    Abstract file storage interface.
    
    Can be implemented with local filesystem, S3, MinIO, etc.
    """
    
    @abstractmethod
    async def upload(
        self, 
        content: bytes, 
        path: str, 
        filename: Optional[str] = None
    ) -> str:
        """
        Upload a file to storage.
        
        Args:
            content: File content as bytes
            path: Storage path/prefix
            filename: Optional filename (generated if not provided)
            
        Returns:
            Full path to the stored file
        """
        pass
    
    @abstractmethod
    async def upload_file(
        self, 
        file_path: str, 
        dest_path: str
    ) -> str:
        """
        Upload a file from disk to storage.
        
        Args:
            file_path: Local file path
            dest_path: Destination path in storage
            
        Returns:
            Full path to the stored file
        """
        pass
    
    @abstractmethod
    async def download(self, path: str) -> bytes:
        """
        Download a file from storage.
        
        Args:
            path: Path to the file
            
        Returns:
            File content as bytes
        """
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            path: Path to the file
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass
    
    @abstractmethod
    async def get_info(self, path: str) -> Optional[FileInfo]:
        """Get file information."""
        pass
    
    @abstractmethod
    async def list_files(
        self, 
        prefix: str = "", 
        recursive: bool = False
    ) -> List[FileInfo]:
        """List files with optional prefix filter."""
        pass


class LocalFileStorage(FileStorage):
    """
    Local filesystem storage implementation.
    
    Stores files in a configurable base directory.
    """
    
    def __init__(self, base_dir: str = "./storage"):
        """
        Initialize local file storage.
        
        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir).absolute()
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path for a storage path."""
        # Prevent path traversal
        clean_path = Path(path).as_posix().lstrip("/")
        full_path = self.base_dir / clean_path
        
        # Ensure path is within base directory
        if not str(full_path.absolute()).startswith(str(self.base_dir)):
            raise FileStorageError("Invalid path: path traversal detected")
        
        return full_path
    
    def _generate_filename(self, content: bytes, ext: str = "") -> str:
        """Generate a unique filename based on content hash."""
        hash_val = hashlib.sha256(content).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{hash_val}{ext}"
    
    async def upload(
        self, 
        content: bytes, 
        path: str, 
        filename: Optional[str] = None
    ) -> str:
        """Upload content to storage."""
        if not filename:
            filename = self._generate_filename(content)
        
        full_path = self._get_full_path(f"{path}/{filename}")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(content)
        
        return f"{path}/{filename}"
    
    async def upload_file(self, file_path: str, dest_path: str) -> str:
        """Upload a file from disk."""
        source = Path(file_path)
        if not source.exists():
            raise FileStorageError(f"Source file not found: {file_path}")
        
        dest = self._get_full_path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source, dest)
        return dest_path
    
    async def download(self, path: str) -> bytes:
        """Download file content."""
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileStorageError(f"File not found: {path}")
        
        async with aiofiles.open(full_path, "rb") as f:
            return await f.read()
    
    async def delete(self, path: str) -> bool:
        """Delete a file."""
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            return False
        
        if full_path.is_dir():
            shutil.rmtree(full_path)
        else:
            full_path.unlink()
        
        return True
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        full_path = self._get_full_path(path)
        return full_path.exists()
    
    async def get_info(self, path: str) -> Optional[FileInfo]:
        """Get file information."""
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            return None
        
        stat = full_path.stat()
        mime_type, _ = mimetypes.guess_type(str(full_path))
        
        return FileInfo(
            path=path,
            filename=full_path.name,
            size=stat.st_size,
            mime_type=mime_type,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime)
        )
    
    async def list_files(
        self, 
        prefix: str = "", 
        recursive: bool = False
    ) -> List[FileInfo]:
        """List files in storage."""
        base = self._get_full_path(prefix) if prefix else self.base_dir
        
        if not base.exists():
            return []
        
        files = []
        
        if recursive:
            for item in base.rglob("*"):
                if item.is_file():
                    rel_path = str(item.relative_to(self.base_dir))
                    info = await self.get_info(rel_path)
                    if info:
                        files.append(info)
        else:
            for item in base.iterdir():
                if item.is_file():
                    rel_path = str(item.relative_to(self.base_dir))
                    info = await self.get_info(rel_path)
                    if info:
                        files.append(info)
        
        return files
    
    async def get_checksum(self, path: str) -> Optional[str]:
        """Calculate file checksum."""
        try:
            content = await self.download(path)
            return hashlib.sha256(content).hexdigest()
        except FileStorageError:
            return None


# Global storage instance
_storage: Optional[FileStorage] = None


def get_storage(base_dir: str = "./storage") -> FileStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = LocalFileStorage(base_dir)
    return _storage
