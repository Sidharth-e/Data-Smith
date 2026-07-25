"""
Pydantic models for dataset output formats.
"""

from pydantic import BaseModel, Field
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
# ChatML Format (Conversational)
# ============================================
class ChatMLMessage(BaseModel):
    """Single message in a ChatML conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatMLFormat(BaseModel):
    """
    Conversational chat format (ChatML) for fine-tuning.
    Example:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ]
    }
    """
    messages: List[ChatMLMessage]


# ============================================
# ShareGPT Format (Conversational)
# ============================================
class ShareGPTMessage(BaseModel):
    """Single message in a ShareGPT conversation."""
    from_: Literal["human", "gpt"] = Field(alias="from")
    value: str

class ShareGPTFormat(BaseModel):
    """
    Conversational format (ShareGPT) for fine-tuning.
    Example:
    {
        "conversations": [
            {"from": "human", "value": "What is X?"},
            {"from": "gpt", "value": "X is..."}
        ]
    }
    """
    conversations: List[ShareGPTMessage]


# ============================================
# DPO Format (Preference Alignment)
# ============================================
class DPOFormat(BaseModel):
    """
    Preference alignment format (DPO) for fine-tuning.
    Example:
    {
        "prompt": "Write a short poem about code.",
        "chosen": "Lines of code write the future bold.",
        "rejected": "Code is text that does stuff on computers."
    }
    """
    prompt: str
    chosen: str
    rejected: str


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
    format_type: Literal["alpaca", "chatml", "sharegpt", "dpo", "completion"]
    num_samples: int = 5


class GenerateResponse(BaseModel):
    """Response model for dataset generation."""
    success: bool
    format_type: str
    data: List[dict]
    message: str = ""
