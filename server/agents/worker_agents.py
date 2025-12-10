"""
Worker Agents for Data Smith.
Specialized agents for different dataset generation tasks.
"""

import json
import re
from typing import List, Dict, Any, Literal
from abc import ABC, abstractmethod

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.supervisor_agent import TaskType


class BaseWorkerAgent(ABC):
    """Base class for worker agents."""
    
    def __init__(self, model_name: str = "mistral:7b-instruct", temperature: float = 0.7):
        """Initialize worker agent."""
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.parser = StrOutputParser()
    
    @abstractmethod
    async def process(
        self,
        content: str,
        output_format: str = "alpaca",
        num_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """Process content and generate samples."""
        pass
    
    def _extract_json_array(self, text: str) -> List[dict]:
        """Extract JSON array from LLM response."""
        # Try to find JSON array
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try individual objects
        objects = []
        for match in re.finditer(r'\{[^{}]*\}', text):
            try:
                obj = json.loads(match.group())
                objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        return objects


class QAGeneratorAgent(BaseWorkerAgent):
    """Worker agent for generating Q&A pairs."""
    
    async def process(
        self,
        content: str,
        output_format: str = "alpaca",
        num_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate question-answer pairs from content."""
        
        if output_format == "alpaca":
            return await self._generate_alpaca(content, num_samples)
        elif output_format == "chat":
            return await self._generate_chat(content, num_samples)
        else:
            return await self._generate_completion(content, num_samples)
    
    async def _generate_alpaca(self, content: str, num_samples: int) -> List[dict]:
        """Generate Alpaca format Q&A pairs."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator. Create instruction-following training data.
IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Generate {num_samples} Q&A pairs from this content:

{content}

Return JSON array with objects containing:
- "instruction": A question about the content
- "input": "" (empty or brief context)
- "output": The answer based on the content

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        # Validate and format
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict):
                validated.append({
                    "instruction": str(item.get("instruction", "")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", ""))
                })
        
        return validated
    
    async def _generate_chat(self, content: str, num_samples: int) -> List[dict]:
        """Generate chat format conversations."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator. Create conversational training data.
IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Generate {num_samples} conversations from this content:

{content}

Return JSON array where each object has a "messages" array with:
- system message (role: "system")
- user question (role: "user")
- assistant answer (role: "assistant")

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict) and "messages" in item:
                messages = []
                for msg in item["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append({
                            "role": msg["role"],
                            "content": str(msg["content"])
                        })
                if messages:
                    validated.append({"messages": messages})
        
        return validated
    
    async def _generate_completion(self, content: str, num_samples: int) -> List[dict]:
        """Generate completion format text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator. Create text completion data.
IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Generate {num_samples} informative text completions from this content:

{content}

Return JSON array where each object has:
- "text": A complete, coherent passage based on the content

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict) and "text" in item:
                validated.append({"text": str(item["text"])})
        
        return validated


class SummaryAgent(BaseWorkerAgent):
    """Worker agent for generating summaries."""
    
    async def process(
        self,
        content: str,
        output_format: str = "alpaca",
        num_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate summary-focused training data."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a summarization expert. Create training data for summarization tasks.
IMPORTANT: Return ONLY a valid JSON array."""),
            ("human", """Create {num_samples} summarization examples from this content:

{content}

Return JSON array with objects containing:
- "instruction": "Summarize the following text" or similar
- "input": A portion of the content to summarize
- "output": A concise summary

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict):
                validated.append({
                    "instruction": str(item.get("instruction", "Summarize the following text")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", ""))
                })
        
        return validated


class EntityExtractionAgent(BaseWorkerAgent):
    """Worker agent for extracting entities."""
    
    async def process(
        self,
        content: str,
        output_format: str = "alpaca",
        num_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate entity extraction training data."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an entity extraction expert. Create training data for NER tasks.
IMPORTANT: Return ONLY a valid JSON array."""),
            ("human", """Create {num_samples} entity extraction examples from this content:

{content}

Return JSON array with objects containing:
- "instruction": "Extract [entity type] from the text" or similar
- "input": Text containing entities
- "output": Extracted entities as structured text

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict):
                validated.append({
                    "instruction": str(item.get("instruction", "Extract entities from the text")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", ""))
                })
        
        return validated


class ClassificationAgent(BaseWorkerAgent):
    """Worker agent for classification tasks."""
    
    async def process(
        self,
        content: str,
        output_format: str = "alpaca",
        num_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate classification training data."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a classification expert. Create training data for text classification.
IMPORTANT: Return ONLY a valid JSON array."""),
            ("human", """Create {num_samples} classification examples from this content:

{content}

Return JSON array with objects containing:
- "instruction": "Classify the following text" or specify categories
- "input": Text to classify
- "output": Classification label or category

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict):
                validated.append({
                    "instruction": str(item.get("instruction", "Classify the following text")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", ""))
                })
        
        return validated


class KeyPointsAgent(BaseWorkerAgent):
    """Worker agent for extracting key points."""
    
    async def process(
        self,
        content: str,
        output_format: str = "alpaca",
        num_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate key points extraction training data."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying key information. Create training data for key point extraction.
IMPORTANT: Return ONLY a valid JSON array."""),
            ("human", """Create {num_samples} key point extraction examples from this content:

{content}

Return JSON array with objects containing:
- "instruction": "What are the key points in this text?" or similar
- "input": Text to analyze
- "output": Bullet points or numbered list of key information

Return ONLY the JSON array:""")
        ])
        
        chain = prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "content": content[:2000],
            "num_samples": num_samples
        })
        
        results = self._extract_json_array(response)
        
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict):
                validated.append({
                    "instruction": str(item.get("instruction", "What are the key points?")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", ""))
                })
        
        return validated


class WorkerAgentFactory:
    """Factory for creating worker agents."""
    
    def __init__(self, model_name: str = "mistral:7b-instruct"):
        """Initialize factory with model configuration."""
        self.model_name = model_name
    
    def create_worker(self, task_type: TaskType) -> BaseWorkerAgent:
        """
        Create a worker agent for the specified task type.
        
        Args:
            task_type: Type of task the worker should handle
            
        Returns:
            Appropriate worker agent instance
        """
        worker_map = {
            TaskType.QA_GENERATION: QAGeneratorAgent,
            TaskType.SUMMARIZATION: SummaryAgent,
            TaskType.ENTITY_EXTRACTION: EntityExtractionAgent,
            TaskType.CLASSIFICATION: ClassificationAgent,
            TaskType.KEY_POINTS: KeyPointsAgent,
            TaskType.CUSTOM: QAGeneratorAgent,  # Default to QA
        }
        
        worker_class = worker_map.get(task_type, QAGeneratorAgent)
        return worker_class(model_name=self.model_name)
    
    def create_all_workers(self) -> Dict[TaskType, BaseWorkerAgent]:
        """Create one instance of each worker type."""
        return {
            task_type: self.create_worker(task_type)
            for task_type in TaskType
            if task_type != TaskType.CUSTOM
        }
