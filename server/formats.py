"""
Pydantic models for dataset output formats.
"""

from pydantic import BaseModel
from typing import List, Literal


# ============================================
# Alpaca Format
# ============================================
class AlpacaFormat(BaseModel):
    """
    Alpaca-style instruction format for fine-tuning.
    Example:
    {
        "instruction": "Convert temperature from Celsius to Fahrenheit.",
        "input": "25",
        "output": "77°F"
    }
    """
    instruction: str
    input: str
    output: str


# ============================================
# Chat Format (Conversational)
# ============================================
class ChatMessage(BaseModel):
    """Single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatFormat(BaseModel):
    """
    Conversational chat format for fine-tuning.
    Example:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ]
    }
    """
    messages: List[ChatMessage]


# ============================================
# Completion Format (Raw Text)
# ============================================
class CompletionFormat(BaseModel):
    """
    Raw text completion format for fine-tuning.
    Example:
    {"text": "The mitochondria is the powerhouse of the cell..."}
    """
    text: str


# ============================================
# API Request/Response Models
# ============================================
class GenerateRequest(BaseModel):
    """Request model for dataset generation."""
    format_type: Literal["alpaca", "chat", "completion"]
    num_samples: int = 5


class GenerateResponse(BaseModel):
    """Response model for dataset generation."""
    success: bool
    format_type: str
    data: List[dict]
    message: str = ""
