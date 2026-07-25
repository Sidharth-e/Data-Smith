"""
LangChain Agent for Dataset Generation using Ollama.
"""

import asyncio
import json
import logging
import random
import re
from typing import Any, AsyncIterator, Dict, List, Literal, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from errors import GenerationError, map_llm_error
from formats import AlpacaFormat, ChatFormat, ChatMessage, CompletionFormat
from model_factory import ModelFactory

logger = logging.getLogger("data_smith")

# How many samples to ask the LLM for in a single call. Smaller batches
# produce output faster and more reliably; a local 7B model can stall or
# truncate when asked for a large JSON array in one shot.
BATCH_SIZE = 10

# Cap on the source text fed to the model per call (characters).
SOURCE_CHAR_LIMIT = 4000

# Maximum number of LLM calls to run concurrently. Local Ollama usually serves
# one request at a time; raise this for cloud / multi-instance backends.
MAX_CONCURRENCY = 1

# Per-LLM-call timeout in seconds. Must be >= the underlying client timeout
# (see model_factory ollama timeout) so our wait_for doesn't fire first and
# mask the real error. Local models can take minutes per batch.
LLM_TIMEOUT = 600.0


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
            return await asyncio.wait_for(
                chain.ainvoke(prompt_vars), timeout=LLM_TIMEOUT
            )
        except GenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise GenerationError(
                f"LLM call timed out after {LLM_TIMEOUT}s"
            ) from exc
        except Exception as exc:
            raise map_llm_error(exc) from exc

    async def _run_chain_stream(
        self, prompt, prompt_vars: dict
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run a prompt|llm chain, yielding token-level deltas as SSE events.

        Emits:
          - {"type": "thinking", "content": <chunk>} for reasoning tokens
            (only when the underlying model exposes them, e.g. Ollama "reasoning").
          - {"type": "token", "content": <chunk>} for content tokens.
          - {"type": "done"} when the stream completes.

        Falls back to a single "token" event with the full text if the model
        does not support streaming (so callers still get the raw text).
        """
        try:
            if hasattr(self.llm, "astream"):
                buffer: List[str] = []
                # (prompt | llm).astream yields AIMessageChunk objects. The
                # `content` attribute carries the text delta. Ollama thinking
                # models surface reasoning in additional_kwargs instead.
                async for chunk in (prompt | self.llm).astream(prompt_vars):
                    content = getattr(chunk, "content", None)
                    if content is None:
                        content = str(chunk) if chunk else ""
                    add_kw = getattr(chunk, "additional_kwargs", {}) or {}
                    thinking = (
                        add_kw.get("reasoning_content")
                        or add_kw.get("reasoning")
                        or add_kw.get("thinking")
                    )
                    if thinking:
                        yield {"type": "thinking", "content": str(thinking)}
                        continue
                    if content:
                        buffer.append(content)
                        yield {"type": "token", "content": content}
                yield {"type": "done", "text": "".join(buffer)}
            else:
                # No streaming support: do a regular invoke and emit once.
                raw = await asyncio.wait_for(
                    (prompt | self.llm).ainvoke(prompt_vars), timeout=LLM_TIMEOUT
                )
                text = getattr(raw, "content", str(raw)) or ""
                yield {"type": "token", "content": text}
                yield {"type": "done", "text": text}
        except GenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise GenerationError(
                f"LLM call timed out after {LLM_TIMEOUT}s"
            ) from exc
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
        prompt = self._prompt_alpaca()
        
        response = await self._run_chain(prompt, {
            "text": text[:SOURCE_CHAR_LIMIT],  # Limit input size
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_alpaca(results, num_samples)
    
    def _prompt_alpaca(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
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

    def _validate_alpaca(self, results: List[dict], num_samples: int) -> List[dict]:
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
        prompt = self._prompt_chat()
        
        response = await self._run_chain(prompt, {
            "text": text[:SOURCE_CHAR_LIMIT],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_chat(results, num_samples)
    
    def _prompt_chat(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
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
    
    def _validate_chat(self, results: List[dict], num_samples: int) -> List[dict]:
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
        prompt = self._prompt_completion()
        
        response = await self._run_chain(prompt, {
            "text": text[:SOURCE_CHAR_LIMIT],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_completion(results, num_samples)
    
    def _prompt_completion(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
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
    
    def _validate_completion(self, results: List[dict], num_samples: int) -> List[dict]:
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
        # Batches run concurrently (bounded by MAX_CONCURRENCY) so a large
        # request doesn't take num_batches * single_call_latency sequentially.
        num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
        # Extra attempts to compensate for duplicates / under-generation.
        total_attempts = num_batches + 3

        async def run_batch(attempt: int) -> tuple[int, List[dict]]:
            window = self._window(text, attempt, total_attempts)
            try:
                batch = await handler(window, BATCH_SIZE)
            except GenerationError:
                raise
            except Exception as exc:
                logger.warning("Batch %d failed: %s", attempt, exc)
                return attempt, []
            return attempt, batch

        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        async def guarded(attempt: int) -> tuple[int, List[dict]]:
            async with sem:
                return await run_batch(attempt)

        batch_results = await asyncio.gather(
            *(guarded(i) for i in range(total_attempts))
        )

        results: List[dict] = []
        seen: set[str] = set()
        for _, batch in batch_results:
            for item in batch:
                key = self._dedupe_key(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                results.append(item)
            if len(results) >= num_samples:
                break

        # Trim to exactly the requested count.
        return results[:num_samples]

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------
    def _validator_for(self, format_type: str):
        return {
            "alpaca": self._validate_alpaca,
            "chat": self._validate_chat,
            "completion": self._validate_completion,
        }[format_type]

    def _prompt_for(self, format_type: str) -> ChatPromptTemplate:
        return {
            "alpaca": self._prompt_alpaca,
            "chat": self._prompt_chat,
            "completion": self._prompt_completion,
        }[format_type]()

    async def generate_stream(
        self,
        text: str,
        format_type: Literal["alpaca", "chat", "completion"],
        num_samples: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a dataset generation run as a sequence of structured events.

        Event types (each yielded as a dict; the HTTP layer serializes them):
          - {"type": "start", "format_type", "num_samples", "num_batches"}
          - {"type": "batch_start", "index", "total", "batch_size"}
          - {"type": "thinking", "content", "index"} (optional, per batch)
          - {"type": "token", "content", "index"} (per batch)
          - {"type": "batch_done", "index", "samples": [...], "count"}
          - {"type": "progress", "done", "total", "samples_so_far"}
          - {"type": "complete", "data": [...]}
          - {"type": "error", "message"}
          - {"type": "warning", "message", "index"}

        For large requests, batches run sequentially (so the frontend can
        follow along in order). Each batch streams its own thinking + content
        tokens; once a batch finishes we parse the accumulated text and emit
        the validated samples.
        """
        prompt = self._prompt_for(format_type)
        validate = self._validator_for(format_type)

        # Single batch for small requests, otherwise batched.
        if num_samples <= BATCH_SIZE:
            num_batches = 1
            extra = 0
        else:
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            extra = 3  # extra attempts to cover dedup / under-generation
        total_attempts = num_batches + extra

        yield {
            "type": "start",
            "format_type": format_type,
            "num_samples": num_samples,
            "num_batches": total_attempts,
            "batch_size": BATCH_SIZE,
        }

        results: List[dict] = []
        seen: set[str] = set()
        completed_batches = 0

        for attempt in range(total_attempts):
            if len(results) >= num_samples:
                break

            window = self._window(text, attempt, total_attempts)
            batch_size = min(BATCH_SIZE, num_samples - len(results))
            yield {
                "type": "batch_start",
                "index": attempt,
                "total": total_attempts,
                "batch_size": batch_size,
            }

            accumulated: List[str] = []
            try:
                async for ev in self._run_chain_stream(prompt, {
                    "text": window[:SOURCE_CHAR_LIMIT],
                    "num_samples": batch_size,
                }):
                    if ev["type"] == "thinking":
                        yield {
                            "type": "thinking",
                            "content": ev["content"],
                            "index": attempt,
                        }
                    elif ev["type"] == "token":
                        accumulated.append(ev["content"])
                        yield {
                            "type": "token",
                            "content": ev["content"],
                            "index": attempt,
                        }
                    elif ev["type"] == "done":
                        raw_text = ev.get("text") or "".join(accumulated)
                        parsed = self._extract_json_array(raw_text)
                        batch_samples = validate(parsed, batch_size)
                        # Dedupe against the running set.
                        new_samples: List[dict] = []
                        for item in batch_samples:
                            key = self._dedupe_key(item)
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            new_samples.append(item)
                        results.extend(new_samples)
                        completed_batches += 1
                        yield {
                            "type": "batch_done",
                            "index": attempt,
                            "samples": new_samples,
                            "count": len(new_samples),
                        }
                        yield {
                            "type": "progress",
                            "done": completed_batches,
                            "total": total_attempts,
                            "samples_so_far": len(results),
                        }
            except GenerationError as exc:
                yield {
                    "type": "warning",
                    "message": str(exc.user_message),
                    "index": attempt,
                }
                logger.warning("Batch %d failed in stream: %s", attempt, exc)
                completed_batches += 1
                yield {
                    "type": "progress",
                    "done": completed_batches,
                    "total": total_attempts,
                    "samples_so_far": len(results),
                }
                continue
            except Exception as exc:
                logger.warning("Batch %d failed: %s", attempt, exc)
                yield {
                    "type": "warning",
                    "message": str(exc),
                    "index": attempt,
                }
                completed_batches += 1
                yield {
                    "type": "progress",
                    "done": completed_batches,
                    "total": total_attempts,
                    "samples_so_far": len(results),
                }
                continue

        trimmed = results[:num_samples]
        yield {"type": "complete", "data": trimmed, "count": len(trimmed)}
