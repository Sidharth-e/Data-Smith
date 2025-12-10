"""
Supervisor Agent for Data Smith.
Orchestrates worker agents for parallel document processing.
"""

import asyncio
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.document_splitter import DocumentChunk


class TaskType(str, Enum):
    """Types of tasks that can be assigned to workers."""
    QA_GENERATION = "qa_generation"
    SUMMARIZATION = "summarization"
    ENTITY_EXTRACTION = "entity_extraction"
    CLASSIFICATION = "classification"
    KEY_POINTS = "key_points"
    CUSTOM = "custom"


@dataclass
class WorkerTask:
    """A task assigned to a worker agent."""
    task_id: str
    task_type: TaskType
    chunks: List[DocumentChunk]
    instructions: str
    output_format: Dict[str, Any]
    priority: int = 1


@dataclass
class WorkerResult:
    """Result from a worker agent."""
    task_id: str
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ProcessingPlan:
    """Plan created by supervisor for processing a document."""
    document_type: str
    recommended_tasks: List[TaskType]
    worker_count: int
    chunk_distribution: Dict[str, List[int]]  # worker_id -> chunk_ids
    estimated_time: float  # seconds
    notes: str


class SupervisorAgent:
    """
    Supervisor agent that orchestrates document processing.
    
    Responsibilities:
    - Analyze document structure and content
    - Create optimal processing plan
    - Distribute chunks to worker agents
    - Monitor worker progress
    - Aggregate and validate results
    """
    
    def __init__(
        self,
        model_name: str = "mistral:7b-instruct",
        max_workers: int = 5,
        temperature: float = 0.3
    ):
        """
        Initialize the supervisor agent.
        
        Args:
            model_name: Ollama model to use
            max_workers: Maximum number of concurrent workers
            temperature: LLM temperature for planning
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.parser = StrOutputParser()
        
        # Import worker agents lazily
        from agents.worker_agents import WorkerAgentFactory
        self.worker_factory = WorkerAgentFactory(model_name=model_name)
    
    async def analyze_document(
        self, 
        chunks: List[DocumentChunk],
        sample_size: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze document to determine optimal processing strategy.
        
        Args:
            chunks: Document chunks to analyze
            sample_size: Number of chunks to sample for analysis
            
        Returns:
            Analysis results including document type and recommendations
        """
        # Sample chunks for analysis
        sample_chunks = chunks[:sample_size] if len(chunks) >= sample_size else chunks
        sample_text = "\n\n---\n\n".join([c.content for c in sample_chunks])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document analysis expert. Analyze the provided text samples and determine:
1. Document type (e.g., technical, academic, narrative, instructional, data)
2. Main topics covered
3. Best processing approach for dataset generation

Return your analysis as JSON."""),
            ("human", """Analyze these document samples:

{sample_text}

Total chunks: {total_chunks}
Average chunk size: {avg_size} characters

Return JSON with:
- document_type: string
- topics: list of main topics
- dataset_types: list of recommended dataset types (qa, summary, entities, classification)
- notes: processing recommendations""")
        ])
        
        chain = prompt | self.llm | self.parser
        
        try:
            response = await chain.ainvoke({
                "sample_text": sample_text[:3000],
                "total_chunks": len(chunks),
                "avg_size": sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
            })
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            pass
        
        # Default analysis
        return {
            "document_type": "general",
            "topics": ["unknown"],
            "dataset_types": ["qa"],
            "notes": "Default processing"
        }
    
    async def create_processing_plan(
        self,
        chunks: List[DocumentChunk],
        requested_tasks: Optional[List[TaskType]] = None,
        analysis: Optional[Dict[str, Any]] = None
    ) -> ProcessingPlan:
        """
        Create a processing plan for the document.
        
        Args:
            chunks: Document chunks to process
            requested_tasks: Specific tasks requested (optional)
            analysis: Pre-computed document analysis (optional)
            
        Returns:
            ProcessingPlan with task distribution
        """
        if not analysis:
            analysis = await self.analyze_document(chunks)
        
        # Determine tasks
        if requested_tasks:
            tasks = requested_tasks
        else:
            # Map analysis recommendations to task types
            type_map = {
                "qa": TaskType.QA_GENERATION,
                "summary": TaskType.SUMMARIZATION,
                "entities": TaskType.ENTITY_EXTRACTION,
                "classification": TaskType.CLASSIFICATION,
                "key_points": TaskType.KEY_POINTS
            }
            tasks = [
                type_map.get(t, TaskType.QA_GENERATION) 
                for t in analysis.get("dataset_types", ["qa"])
            ]
        
        # Calculate optimal worker count
        chunk_count = len(chunks)
        worker_count = min(
            self.max_workers,
            max(1, chunk_count // 2)  # At least 2 chunks per worker
        )
        
        # Distribute chunks to workers
        distribution = {}
        chunks_per_worker = chunk_count // worker_count
        remainder = chunk_count % worker_count
        
        start = 0
        for i in range(worker_count):
            worker_id = f"worker_{i}"
            count = chunks_per_worker + (1 if i < remainder else 0)
            distribution[worker_id] = list(range(start, start + count))
            start += count
        
        # Estimate processing time (rough estimate: 2 seconds per chunk)
        estimated_time = (chunk_count / worker_count) * 2.0
        
        return ProcessingPlan(
            document_type=analysis.get("document_type", "general"),
            recommended_tasks=tasks,
            worker_count=worker_count,
            chunk_distribution=distribution,
            estimated_time=estimated_time,
            notes=analysis.get("notes", "")
        )
    
    async def process_document(
        self,
        chunks: List[DocumentChunk],
        task_type: TaskType = TaskType.QA_GENERATION,
        output_format: Literal["alpaca", "chat", "completion"] = "alpaca",
        samples_per_chunk: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Process document using worker agents and aggregate results.
        
        Args:
            chunks: Document chunks to process
            task_type: Type of processing task
            output_format: Output dataset format
            samples_per_chunk: Number of samples to generate per chunk
            
        Returns:
            List of generated dataset samples
        """
        if not chunks:
            return []
        
        # Create processing plan
        plan = await self.create_processing_plan(chunks, [task_type])
        
        # Create worker tasks
        worker_tasks = []
        for worker_id, chunk_ids in plan.chunk_distribution.items():
            worker_chunks = [chunks[i] for i in chunk_ids if i < len(chunks)]
            if worker_chunks:
                task = WorkerTask(
                    task_id=worker_id,
                    task_type=task_type,
                    chunks=worker_chunks,
                    instructions=self._get_task_instructions(task_type),
                    output_format=self._get_output_format(output_format)
                )
                worker_tasks.append(task)
        
        # Execute tasks in parallel
        results = await asyncio.gather(*[
            self._execute_worker_task(task, output_format, samples_per_chunk)
            for task in worker_tasks
        ])
        
        # Aggregate results
        all_samples = []
        for result in results:
            if result.success:
                all_samples.extend(result.data)
        
        return all_samples
    
    async def _execute_worker_task(
        self,
        task: WorkerTask,
        output_format: str,
        samples_per_chunk: int
    ) -> WorkerResult:
        """Execute a single worker task."""
        try:
            worker = self.worker_factory.create_worker(task.task_type)
            
            all_data = []
            for chunk in task.chunks:
                samples = await worker.process(
                    chunk.content,
                    output_format=output_format,
                    num_samples=samples_per_chunk
                )
                all_data.extend(samples)
            
            return WorkerResult(
                task_id=task.task_id,
                success=True,
                data=all_data,
                metadata={"chunks_processed": len(task.chunks)}
            )
            
        except Exception as e:
            return WorkerResult(
                task_id=task.task_id,
                success=False,
                data=[],
                error=str(e)
            )
    
    def _get_task_instructions(self, task_type: TaskType) -> str:
        """Get instructions for a specific task type."""
        instructions = {
            TaskType.QA_GENERATION: "Generate question-answer pairs from the content",
            TaskType.SUMMARIZATION: "Create concise summaries of the content",
            TaskType.ENTITY_EXTRACTION: "Extract named entities and their relationships",
            TaskType.CLASSIFICATION: "Classify the content into relevant categories",
            TaskType.KEY_POINTS: "Extract key points and important information",
            TaskType.CUSTOM: "Process the content as specified"
        }
        return instructions.get(task_type, "Process the content")
    
    def _get_output_format(self, format_name: str) -> Dict[str, Any]:
        """Get output format specification."""
        formats = {
            "alpaca": {
                "instruction": "string",
                "input": "string (optional context)",
                "output": "string"
            },
            "chat": {
                "messages": [
                    {"role": "system", "content": "string"},
                    {"role": "user", "content": "string"},
                    {"role": "assistant", "content": "string"}
                ]
            },
            "completion": {
                "text": "string"
            }
        }
        return formats.get(format_name, formats["alpaca"])
