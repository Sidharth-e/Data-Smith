"""
Job Manager Service for Data Smith.
Manages async background jobs with in-memory queue (upgradeable to Redis/Celery).
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Awaitable
from collections import defaultdict
import logging

from models.job import Job, JobType, JobStatus, JobCreate, JobUpdate, JobPriority

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages background processing jobs.
    
    Features:
    - In-memory job queue (can be upgraded to Redis)
    - Async job execution
    - Progress tracking
    - Job cancellation
    - Event callbacks for real-time updates
    """
    
    def __init__(self, max_concurrent_jobs: int = 5):
        """
        Initialize the job manager.
        
        Args:
            max_concurrent_jobs: Maximum jobs to run concurrently
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._handlers: Dict[JobType, Callable] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    def register_handler(
        self, 
        job_type: JobType, 
        handler: Callable[[Job], Awaitable[Dict[str, Any]]]
    ):
        """
        Register a handler for a job type.
        
        Args:
            job_type: Type of job this handler processes
            handler: Async function that processes the job
        """
        self._handlers[job_type] = handler
    
    def on_job_update(self, job_id: str, callback: Callable[[Job], None]):
        """
        Register a callback for job updates.
        
        Args:
            job_id: Job ID to watch
            callback: Function to call on updates
        """
        self._callbacks[job_id].append(callback)
    
    async def start(self):
        """Start the job worker."""
        if self._worker_task is None:
            self._shutdown = False
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Job manager started")
    
    async def stop(self):
        """Stop the job worker."""
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        
        # Cancel running jobs
        for task in self._running_jobs.values():
            task.cancel()
        
        logger.info("Job manager stopped")
    
    async def create_job(self, request: JobCreate) -> Job:
        """
        Create and queue a new job.
        
        Args:
            request: Job creation request
            
        Returns:
            Created job
        """
        job = Job(
            job_type=request.job_type,
            params=request.params,
            priority=request.priority,
            metadata=request.metadata,
            status=JobStatus.QUEUED
        )
        
        self._jobs[job.id] = job
        
        # Add to priority queue (negative priority for max-heap behavior)
        await self._queue.put((-job.priority.value, job.id))
        
        logger.info(f"Job {job.id} created and queued")
        await self._notify_update(job)
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self, 
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 50
    ) -> List[Job]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    async def update_job(self, job_id: str, update: JobUpdate) -> Optional[Job]:
        """
        Update a job's status.
        
        Args:
            job_id: Job ID
            update: Update data
            
        Returns:
            Updated job or None
        """
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        if update.status:
            job.status = update.status
            if update.status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now()
            elif update.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = datetime.now()
        
        if update.progress is not None:
            job.progress = update.progress
        if update.progress_message:
            job.progress_message = update.progress_message
        if update.result:
            job.result = update.result
        if update.error:
            job.error = update.error
        
        await self._notify_update(job)
        return job
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        
        # Cancel running task if exists
        if job_id in self._running_jobs:
            self._running_jobs[job_id].cancel()
            del self._running_jobs[job_id]
        
        await self.update_job(job_id, JobUpdate(status=JobStatus.CANCELLED))
        logger.info(f"Job {job_id} cancelled")
        return True
    
    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a completed job.
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if deleted successfully
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status == JobStatus.RUNNING:
            return False
        
        del self._jobs[job_id]
        if job_id in self._callbacks:
            del self._callbacks[job_id]
        
        return True
    
    async def _worker_loop(self):
        """Main worker loop that processes jobs from the queue."""
        while not self._shutdown:
            try:
                # Wait for available slot
                while len(self._running_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(0.1)
                
                # Get next job from queue
                try:
                    _, job_id = await asyncio.wait_for(
                        self._queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                job = self._jobs.get(job_id)
                if not job or job.status == JobStatus.CANCELLED:
                    continue
                
                # Start job execution
                task = asyncio.create_task(self._execute_job(job))
                self._running_jobs[job_id] = task
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_job(self, job: Job):
        """Execute a single job."""
        try:
            # Update status to running
            await self.update_job(job.id, JobUpdate(
                status=JobStatus.RUNNING,
                progress=0.0,
                progress_message="Starting..."
            ))
            
            # Get handler
            handler = self._handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job.job_type}")
            
            # Execute handler
            result = await handler(job)
            
            # Mark as completed
            await self.update_job(job.id, JobUpdate(
                status=JobStatus.COMPLETED,
                progress=1.0,
                progress_message="Completed",
                result=result
            ))
            
            logger.info(f"Job {job.id} completed successfully")
            
        except asyncio.CancelledError:
            await self.update_job(job.id, JobUpdate(
                status=JobStatus.CANCELLED,
                progress_message="Cancelled"
            ))
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            await self.update_job(job.id, JobUpdate(
                status=JobStatus.FAILED,
                error=str(e),
                progress_message=f"Failed: {str(e)}"
            ))
        finally:
            if job.id in self._running_jobs:
                del self._running_jobs[job.id]
    
    async def _notify_update(self, job: Job):
        """Notify callbacks of job update."""
        for callback in self._callbacks.get(job.id, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job)
                else:
                    callback(job)
            except Exception as e:
                logger.error(f"Callback error for job {job.id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get job manager statistics."""
        status_counts = defaultdict(int)
        for job in self._jobs.values():
            status_counts[job.status.value] += 1
        
        return {
            "total_jobs": len(self._jobs),
            "running_jobs": len(self._running_jobs),
            "queued_jobs": self._queue.qsize(),
            "max_concurrent": self.max_concurrent_jobs,
            "by_status": dict(status_counts)
        }


# Global job manager instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


async def init_job_manager():
    """Initialize and start the job manager."""
    manager = get_job_manager()
    await manager.start()
    return manager
