"""
LangChain Agent for Dataset Generation using Ollama.
"""

import json
import logging
import random
import re
from typing import List, Literal, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from errors import GenerationError, map_llm_error
from formats import AlpacaFormat, ChatFormat, ChatMessage, CompletionFormat
from model_factory import ModelFactory

logger = logging.getLogger("data_smith")

# How many samples to ask the LLM for in a single call. Larger values tend to
# truncate or produce lower-quality output; this is a sane ceiling per batch.
BATCH_SIZE = 20

# Cap on the source text fed to the model per call (characters).
SOURCE_CHAR_LIMIT = 4000


def _extract_first_balanced_array(text: str) -> str | None:
    """Return the substring of the first balanced top-level '[...]' block.

    Scans for the first '[' and tracks bracket depth (respecting string literals)
    so a verbose LLM response containing multiple arrays only yields the first
    complete one, instead of spanning from the first '[' to the last ']' (greedy).
    Returns None if no balanced array is found.
    """
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_str = False
    quote = ""
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2  # skip escaped char
                continue
            if ch == quote:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1
    return None


class DatasetAgent:
    """
    LangChain agent that generates fine-tuning datasets from text input.
    The LLM is supplied by `ModelFactory` based on `config.toml`.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the agent with a chat model.

        Args:
            llm: Optional pre-built chat model. When None, one is created
                 from the project config via `ModelFactory`.
        """
        self.llm = llm or ModelFactory().create()
        self.parser = StrOutputParser()
    
    def _extract_json_array(self, text: str) -> List[dict]:
        """Extract a JSON array from LLM response text.

        Tries, in order:
          1. The contents of a ```json ... ``` fenced block.
          2. The first balanced top-level JSON array (non-greedy, brace-aware).
          3. Individual JSON objects as a last resort.
        """
        # 1. Fenced code block (LLMs often wrap output in ```json ... ```)
        fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if fence:
            try:
                parsed = json.loads(fence.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 2. First balanced top-level JSON array
        array = _extract_first_balanced_array(text)
        if array is not None:
            try:
                parsed = json.loads(array)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 3. Individual JSON objects
        objects = []
        for match in re.finditer(r'\{[^{}]*\}', text):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        return objects

    async def _run_chain(self, prompt, prompt_vars: dict) -> str:
        """Run a prompt|llm|parser chain, mapping LLM errors to GenerationError."""
        chain = prompt | self.llm | self.parser
        try:
            return await chain.ainvoke(prompt_vars)
        except GenerationError:
            raise
        except Exception as exc:
            raise map_llm_error(exc) from exc

    async def generate_alpaca(self, text: str, num_samples: int = 5) -> List[dict]:
        """
        Generate Alpaca-style instruction-input-output pairs.
        
        Args:
            text: Source text to generate dataset from
            num_samples: Number of samples to generate
            
        Returns:
            List of Alpaca format dictionaries
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create instruction-following training data in Alpaca format.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} training examples in Alpaca format.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have:
- "instruction": A clear task or question
- "input": Optional context or input data (can be empty string)
- "output": The expected response

Example format:
[
  {{"instruction": "Summarize the main topic", "input": "", "output": "The text discusses..."}},
  {{"instruction": "What is mentioned about X?", "input": "Context here", "output": "X is described as..."}}
]

Return ONLY the JSON array:""")
        ])
        
        response = await self._run_chain(prompt, {
            "text": text[:SOURCE_CHAR_LIMIT],  # Limit input size
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)

        # Validate and clean results
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict):
                validated.append({
                    "instruction": str(item.get("instruction", "")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", ""))
                })

        return validated
    
    async def generate_chat(self, text: str, num_samples: int = 5) -> List[dict]:
        """
        Generate conversational chat format data.
        
        Args:
            text: Source text to generate dataset from
            num_samples: Number of conversations to generate
            
        Returns:
            List of chat format dictionaries with messages array
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create conversational training data in chat format.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} conversations in chat format.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have a "messages" array containing:
- A "system" message defining the assistant's role
- A "user" message with a question or request
- An "assistant" message with the response

Example format:
[
  {{
    "messages": [
      {{"role": "system", "content": "You are a helpful expert."}},
      {{"role": "user", "content": "What is X?"}},
      {{"role": "assistant", "content": "X is..."}}
    ]
  }}
]

Return ONLY the JSON array:""")
        ])
        
        response = await self._run_chain(prompt, {
            "text": text[:SOURCE_CHAR_LIMIT],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)

        # Validate and clean results
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
    
    async def generate_completion(self, text: str, num_samples: int = 5) -> List[dict]:
        """
        Generate raw text completion format data.
        
        Args:
            text: Source text to generate dataset from
            num_samples: Number of completions to generate
            
        Returns:
            List of completion format dictionaries
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create text completion training data.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} text completions.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have:
- "text": A complete, coherent paragraph or passage derived from the source

Example format:
[
  {{"text": "The concept of X involves... It is important because..."}},
  {{"text": "When considering Y, one must understand that..."}}
]

Each text should be informative and self-contained.

Return ONLY the JSON array:""")
        ])
        
        response = await self._run_chain(prompt, {
            "text": text[:SOURCE_CHAR_LIMIT],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)

        # Validate and clean results
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict) and "text" in item:
                validated.append({"text": str(item["text"])})

        return validated
    
    @staticmethod
    def _dedupe_key(item: dict) -> str:
        """Return a canonical key used to detect duplicate samples.

        Comparison is intentionally coarse (stripped, lowercased) so that
        near-identical LLM output counts as a duplicate.
        """
        if "messages" in item and isinstance(item["messages"], list):
            return " ".join(
                str(m.get("content", "")).strip().lower()
                for m in item["messages"]
                if isinstance(m, dict)
            )
        if "instruction" in item:
            return "|".join(
                str(item.get(k, "")).strip().lower() for k in ("instruction", "input", "output")
            )
        if "text" in item:
            return str(item["text"]).strip().lower()
        return json.dumps(item, sort_keys=True).lower()

    def _window(self, text: str, index: int, total: int) -> str:
        """Return a slice of the source text for batch `index` of `total`.

        For docs longer than SOURCE_CHAR_LIMIT this slides a window across the
        text so each batch sees a different region; for short docs it just
        returns the whole text. A deterministic pseudo-random offset per batch
        keeps batches from all starting at the same point.
        """
        if len(text) <= SOURCE_CHAR_LIMIT:
            return text
        # Deterministic but varied offset per batch.
        rng = random.Random(index * 17 + 7)
        max_start = len(text) - SOURCE_CHAR_LIMIT
        start = rng.randint(0, max_start) if total > 1 else 0
        return text[start:start + SOURCE_CHAR_LIMIT]

    async def generate(
        self,
        text: str,
        format_type: Literal["alpaca", "chat", "completion"],
        num_samples: int = 5
    ) -> List[dict]:
        """
        Generate a dataset in the specified format.

        For large `num_samples`, generation is split into batches of
        `BATCH_SIZE` to keep LLM output manageable and high quality. Batches
        iterate until the requested count is reached (or a retry cap is hit),
        accumulating results and removing duplicates.

        Args:
            text: Source text to generate dataset from
            format_type: Output format type
            num_samples: Number of samples to generate

        Returns:
            List of formatted dictionaries
        """
        handler = {
            "alpaca": self.generate_alpaca,
            "chat": self.generate_chat,
            "completion": self.generate_completion,
        }.get(format_type)
        if handler is None:
            raise ValueError(f"Unknown format type: {format_type}")

        # Small request: single call, original behaviour.
        if num_samples <= BATCH_SIZE:
            return await handler(text, num_samples)

        # Large request: batched accumulation with dedup.
        results: List[dict] = []
        seen: set[str] = set()

        # Estimate number of batches; allow extra attempts to compensate for
        # duplicates / under-generation.
        num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
        max_attempts = num_batches + 3  # retry slack
        attempt = 0

        while len(results) < num_samples and attempt < max_attempts:
            remaining = num_samples - len(results)
            batch_n = min(BATCH_SIZE, remaining)

            # Pick a region of the source text for this batch.
            window = self._window(text, attempt, max_attempts)

            try:
                batch = await handler(window, batch_n)
            except GenerationError:
                raise
            except Exception as exc:
                # A single bad batch shouldn't kill the whole run.
                logger.warning("Batch %d failed: %s", attempt, exc)
                attempt += 1
                continue

            for item in batch:
                key = self._dedupe_key(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                results.append(item)

            attempt += 1

        # Trim to exactly the requested count.
        return results[:num_samples]
