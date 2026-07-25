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
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from config import load_config
from errors import GenerationError, map_llm_error
from formats import AlpacaFormat, ChatMLFormat, ChatMLMessage, CompletionFormat
from model_factory import ModelFactory

logger = logging.getLogger("data_smith")

# Defaults used when the [generation] section is absent from config.toml.
_DEFAULTS = {
    "batch_size": 10,
    "source_char_limit": 12000,
    "max_concurrency": 1,
    "llm_timeout": 600.0,
    "extra_attempts": 3,
    "max_revise_rounds": 2,
}


def _generation_settings(config: Optional[dict] = None) -> dict:
    """Load generation tunables from the [generation] config section.

    Falls back to `_DEFAULTS` for any missing key. Coerces values to the
    expected types (ints / float for timeout) so the rest of the code can
    treat them as numbers.
    """
    cfg = (config or load_config()).get("generation", {})
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if v is not None})
    out["batch_size"] = int(out["batch_size"])
    out["source_char_limit"] = int(out["source_char_limit"])
    out["max_concurrency"] = int(out["max_concurrency"])
    out["llm_timeout"] = float(out["llm_timeout"])
    out["extra_attempts"] = int(out["extra_attempts"])
    out["max_revise_rounds"] = int(out["max_revise_rounds"])
    return out


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
    
    def __init__(self, llm: Optional[BaseChatModel] = None, config: Optional[dict] = None):
        """
        Initialize the agent with a chat model.

        Args:
            llm: Optional pre-built chat model. When None, one is created
                 from the project config via `ModelFactory`.
            config: Optional pre-loaded config dict. When None, config.toml
                 is loaded via `load_config`. The `[generation]` section
                 supplies batch_size, source_char_limit, max_concurrency,
                 llm_timeout, extra_attempts, max_revise_rounds.
        """
        self.llm = llm or ModelFactory(config=config).create()
        self.parser = StrOutputParser()
        s = _generation_settings(config)
        self.batch_size = s["batch_size"]
        self.source_char_limit = s["source_char_limit"]
        self.max_concurrency = s["max_concurrency"]
        self.llm_timeout = s["llm_timeout"]
        self.extra_attempts = s["extra_attempts"]
    
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
                chain.ainvoke(prompt_vars), timeout=self.llm_timeout
            )
        except GenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise GenerationError(
                f"LLM call timed out after {self.llm_timeout}s"
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
                    (prompt | self.llm).ainvoke(prompt_vars), timeout=self.llm_timeout
                )
                text = getattr(raw, "content", str(raw)) or ""
                yield {"type": "token", "content": text}
                yield {"type": "done", "text": text}
        except GenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise GenerationError(
                f"LLM call timed out after {self.llm_timeout}s"
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
            "text": text[:self.source_char_limit],  # Limit input size
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_alpaca(results, num_samples)
    
    def _prompt_alpaca(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create instruction-following training data in Alpaca format.

The source text may be anything: a resume, an article, documentation, a report,
a product description, a biography, etc. Use ALL of the information present in
the source, regardless of its type. Do not silently drop or invent information;
the samples must be faithful to the source.

Diversify the samples across the full breadth of the source so each sample
covers a distinct fact or topic rather than paraphrasing the same point.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} training examples in Alpaca format.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have:
- "instruction": A clear task or question
- "input": Optional context or input data (can be empty string)
- "output": The expected response

Cover the full breadth of the source. Include samples about any identifying
information, facts, figures, or topics that appear in the source.

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
    
    async def generate_chatml(self, text: str, num_samples: int = 5) -> List[dict]:
        """
        Generate conversational chat format data.
        
        Args:
            text: Source text to generate dataset from
            num_samples: Number of conversations to generate
            
        Returns:
            List of chat format dictionaries with messages array
        """
        prompt = self._prompt_chatml()
        
        response = await self._run_chain(prompt, {
            "text": text[:self.source_char_limit],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_chatml(results, num_samples)
    
    def _prompt_chatml(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create conversational training data in ChatML format.

The source text may be anything: a resume, an article, documentation, a report,
a product description, a biography, etc. Use ALL of the information present in
the source, regardless of its type. Do not silently drop or invent information;
the samples must be faithful to the source.

Diversify the samples across the full breadth of the source so each sample
covers a distinct fact or topic rather than paraphrasing the same point.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} conversations in ChatML format.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have a "messages" array containing:
- A "system" message defining the assistant's role
- A "user" message with a question or request
- An "assistant" message with the response

Cover the full breadth of the source. Include samples about any identifying
information, facts, figures, or topics that appear in the source.

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
    
    def _validate_chatml(self, results: List[dict], num_samples: int) -> List[dict]:
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
    
    async def generate_sharegpt(self, text: str, num_samples: int = 5) -> List[dict]:
        """
        Generate conversational data in ShareGPT format.
        """
        prompt = self._prompt_sharegpt()
        
        response = await self._run_chain(prompt, {
            "text": text[:self.source_char_limit],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_sharegpt(results, num_samples)
    
    def _prompt_sharegpt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create conversational training data in ShareGPT format.

The source text may be anything: a resume, an article, documentation, a report,
a product description, a biography, etc. Use ALL of the information present in
the source, regardless of its type. Do not silently drop or invent information;
the samples must be faithful to the source.

Diversify the samples across the full breadth of the source so each sample
covers a distinct fact or topic rather than paraphrasing the same point.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} conversations in ShareGPT format.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have a "conversations" array containing objects with "from" and "value" keys. "from" should be either "human" or "gpt".

Cover the full breadth of the source. Include samples about any identifying
information, facts, figures, or topics that appear in the source.

Example format:
[
  {{
    "conversations": [
      {{"from": "human", "value": "What is X?"}},
      {{"from": "gpt", "value": "X is..."}}
    ]
  }}
]

Return ONLY the JSON array:""")
        ])
    
    def _validate_sharegpt(self, results: List[dict], num_samples: int) -> List[dict]:
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict) and "conversations" in item:
                conversations = []
                for msg in item["conversations"]:
                    if isinstance(msg, dict) and "from" in msg and "value" in msg:
                        conversations.append({
                            "from": msg["from"],
                            "value": str(msg["value"])
                        })
                if conversations:
                    validated.append({"conversations": conversations})
        return validated
    
    async def generate_dpo(self, text: str, num_samples: int = 5) -> List[dict]:
        """
        Generate preference alignment data in DPO format.
        """
        prompt = self._prompt_dpo()
        
        response = await self._run_chain(prompt, {
            "text": text[:self.source_char_limit],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_dpo(results, num_samples)
    
    def _prompt_dpo(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create preference alignment training data in DPO format.

The source text may be anything: a resume, an article, documentation, a report,
a product description, a biography, etc. Use ALL of the information present in
the source, regardless of its type. Do not silently drop or invent information;
the samples must be faithful to the source.

Diversify the prompts across the full breadth of the source so each prompt
covers a distinct fact or topic rather than paraphrasing the same point.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} preference pairs in DPO format.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have:
- "prompt": The instruction or question based on the text
- "chosen": A high-quality correct response grounded in the source
- "rejected": A poor-quality or incorrect response

Cover the full breadth of the source. For each prompt, the chosen response
should be accurate to the source and the rejected response should be vague or
incorrect.

Example format:
[
  {{
    "prompt": "Explain X",
    "chosen": "X is an important concept that...",
    "rejected": "X is just some stuff."
  }}
]

Return ONLY the JSON array:""")
        ])
    
    def _validate_dpo(self, results: List[dict], num_samples: int) -> List[dict]:
        validated = []
        for item in results[:num_samples]:
            if isinstance(item, dict) and "prompt" in item and "chosen" in item and "rejected" in item:
                validated.append({
                    "prompt": str(item["prompt"]),
                    "chosen": str(item["chosen"]),
                    "rejected": str(item["rejected"])
                })
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
            "text": text[:self.source_char_limit],
            "num_samples": num_samples
        })

        results = self._extract_json_array(response)
        return self._validate_completion(results, num_samples)
    
    def _prompt_completion(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator for fine-tuning language models.
Your task is to create text completion training data.

The source text may be anything: a resume, an article, documentation, a report,
a product description, a biography, etc. Use ALL of the information present in
the source, regardless of its type. Do not silently drop or invent information;
the samples must be faithful to the source.

Diversify the completions across the full breadth of the source so each
completion covers a distinct fact or topic rather than paraphrasing the same
point.

IMPORTANT: Return ONLY a valid JSON array, no other text."""),
            ("human", """Based on the following source text, generate {num_samples} text completions.

Source Text:
{text}

Generate a JSON array with exactly {num_samples} objects. Each object must have:
- "text": A complete, coherent paragraph or passage derived from the source

Cover the full breadth of the source. Include completions about any
identifying information, facts, figures, or topics that appear in the source.

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
        if "conversations" in item and isinstance(item["conversations"], list):
            return " ".join(
                str(m.get("value", "")).strip().lower()
                for m in item["conversations"]
                if isinstance(m, dict)
            )
        if "instruction" in item:
            return "|".join(
                str(item.get(k, "")).strip().lower() for k in ("instruction", "input", "output")
            )
        if "prompt" in item and "chosen" in item:
            return "|".join(
                str(item.get(k, "")).strip().lower() for k in ("prompt", "chosen", "rejected")
            )
        if "text" in item:
            return str(item["text"]).strip().lower()
        return json.dumps(item, sort_keys=True).lower()

    def _window(self, text: str, index: int, total: int) -> str:
        """Return a slice of the source text for batch `index` of `total`.

        For docs longer than `self.source_char_limit` this slides a window across the
        text so each batch sees a different region; for short docs it just
        returns the whole text. A deterministic pseudo-random offset per batch
        keeps batches from all starting at the same point.
        """
        if len(text) <= self.source_char_limit:
            return text
        # Deterministic but varied offset per batch.
        rng = random.Random(index * 17 + 7)
        max_start = len(text) - self.source_char_limit
        start = rng.randint(0, max_start) if total > 1 else 0
        return text[start:start + self.source_char_limit]

    async def generate(
        self,
        text: str,
        format_type: Literal["alpaca", "chatml", "sharegpt", "dpo", "completion"],
        num_samples: int = 5
    ) -> List[dict]:
        """
        Generate a dataset in the specified format.

        For large `num_samples`, generation is split into batches of
        `self.batch_size` to keep LLM output manageable and high quality. Batches
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
            "chatml": self.generate_chatml,
            "sharegpt": self.generate_sharegpt,
            "dpo": self.generate_dpo,
            "completion": self.generate_completion,
        }.get(format_type)
        if handler is None:
            raise ValueError(f"Unknown format type: {format_type}")

        # Small request: single call, original behaviour.
        if num_samples <= self.batch_size:
            return await handler(text, num_samples)

        # Large request: batched accumulation with dedup.
        # Batches run concurrently (bounded by `self.max_concurrency`) so a large
        # request doesn't take num_batches * single_call_latency sequentially.
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        # Extra attempts to compensate for duplicates / under-generation.
        total_attempts = num_batches + self.extra_attempts

        async def run_batch(attempt: int) -> tuple[int, List[dict]]:
            window = self._window(text, attempt, total_attempts)
            try:
                batch = await handler(window, self.batch_size)
            except GenerationError:
                raise
            except Exception as exc:
                logger.warning("Batch %d failed: %s", attempt, exc)
                return attempt, []
            return attempt, batch

        sem = asyncio.Semaphore(self.max_concurrency)

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
            "chatml": self._validate_chatml,
            "sharegpt": self._validate_sharegpt,
            "dpo": self._validate_dpo,
            "completion": self._validate_completion,
        }[format_type]

    def _prompt_for(self, format_type: str) -> ChatPromptTemplate:
        return {
            "alpaca": self._prompt_alpaca,
            "chatml": self._prompt_chatml,
            "sharegpt": self._prompt_sharegpt,
            "dpo": self._prompt_dpo,
            "completion": self._prompt_completion,
        }[format_type]()

    async def generate_stream(
        self,
        text: str,
        format_type: Literal["alpaca", "chatml", "sharegpt", "dpo", "completion"],
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
        if num_samples <= self.batch_size:
            num_batches = 1
            extra = 0
        else:
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            extra = self.extra_attempts
        total_attempts = num_batches + extra

        yield {
            "type": "start",
            "format_type": format_type,
            "num_samples": num_samples,
            "num_batches": total_attempts,
            "batch_size": self.batch_size,
        }

        results: List[dict] = []
        seen: set[str] = set()
        completed_batches = 0

        for attempt in range(total_attempts):
            if len(results) >= num_samples:
                break

            window = self._window(text, attempt, total_attempts)
            batch_size = min(self.batch_size, num_samples - len(results))
            yield {
                "type": "batch_start",
                "index": attempt,
                "total": total_attempts,
                "batch_size": batch_size,
            }

            accumulated: List[str] = []
            try:
                async for ev in self._run_chain_stream(prompt, {
                    "text": window[:self.source_char_limit],
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


class QAState(TypedDict, total=False):
    """State carried across the per-batch QA subgraph.

    One instance per batch; the outer `generate` / `generate_stream` methods
    run the graph (or walk the node methods manually, for the streaming path)
    once per batch and then dedupe across batches themselves.
    """
    source: str                 # source text slice for this batch
    batch_size: int             # requested samples for this batch
    validate: Any                # format validator fn (format-specific)
    handler: Any                # format generate handler (format-specific)
    candidates: List[dict]      # initial generated candidates (step 1)
    current: List[dict]         # samples being critiqued this round
    verdicts: List[dict]        # critic verdicts (step 2): {"index","verdict","reason"}
    accepted: List[dict]         # samples accepted as-is or after revision
    pending: List[dict]         # [{"sample","reason"}] still needing revision this round
    round: int                  # current revise round (0-based)
    samples: List[dict]         # final validated output (finalize node)


class QualityAgent(DatasetAgent):
    """
    Higher-accuracy generator: plan -> generate -> critique -> revise per batch.

    The per-batch quality loop is implemented as a LangGraph `StateGraph`
    (see `build_qa_graph`). Nodes:

      1. ``generate``  -- produce candidate samples (reuses the parent's
         prompts/validators via a handler).
      2. ``critique``  -- ask a *critic* model (same LLM, critic prompt) to
         score each sample against format-specific quality criteria and
         return a list of verdicts: "ok" or "revise:<reason>".
      3. ``revise``    -- for any sample marked "revise", ask the *reviser*
         (same LLM, revise prompt) to rewrite the sample using the critic's
         feedback, then loop back to ``critique`` while rounds remain.
      4. ``finalize``  -- re-validate and return the accepted samples.

    The graph is driven once per batch; the outer `generate` /
    `generate_stream` methods handle batching, windowing, concurrency and
    cross-batch dedup (same as `DatasetAgent`).

    Trade-off: ~2-3x the LLM calls vs `DatasetAgent` for the same sample
    count (generate + critique + revise), so it is slower and costlier.
    Accuracy / quality of accepted samples should be noticeably higher.

    The class intentionally shares the parent's prompts, validators,
    `_extract_json_array`, `_window`, and `_dedupe_key` so behaviour stays
    consistent and only the orchestration loop differs.
    """

    # How many revise rounds a single sample may go through before we give
    # up and accept the latest revision (or drop the sample). Default; the
    # effective value is read from `[generation].max_revise_rounds` in
    # config.toml at construction time (see `__init__`).
    MAX_REVISE_ROUNDS = 2

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        critic_llm: Optional[BaseChatModel] = None,
        config: Optional[dict] = None,
    ):
        """
        Args:
            llm: Generator + reviser model. Defaults to `ModelFactory().create()`.
            critic_llm: Optional separate model for the critic step. When None,
                the same `llm` is reused (fine for local Ollama; use a stronger
                cloud model here for best results).
            config: Optional pre-loaded config dict. When None, config.toml is
                loaded via `load_config`. Inherits `[generation]` settings from
                `DatasetAgent` and additionally reads `max_revise_rounds`.
        """
        super().__init__(llm=llm, config=config)
        self.critic_llm = critic_llm or self.llm
        s = _generation_settings(config)
        self.max_revise_rounds = s["max_revise_rounds"]
        self._qa_graph = self.build_qa_graph()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def _critic_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a strict quality reviewer for fine-tuning datasets.
You are given a source text and a JSON array of candidate training samples.
Score EACH sample against these criteria:
- Faithfulness: claims must be supported by the source text.
- Clarity: instruction/question is unambiguous.
- Completeness: the output fully answers the instruction.
- Format: matches the requested schema exactly.

Return ONLY a JSON array (no prose) with one entry per input sample, in order.
Each entry: {{"index": <0-based>, "verdict": "ok" | "revise", "reason": "<short reason or empty>"}}"""),
            ("human", """Source text:
{text}

Candidate samples (JSON array):
{samples}

Return the JSON verdict array:"""),
        ])

    def _revise_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset editor. Given one training sample and a
critique, rewrite the sample to fix the problems. Keep the same schema.
Return ONLY the single revised sample object as JSON, no array, no prose."""),
            ("human", """Original sample:
{sample}

Critique:
{critique}

Source text (for reference):
{text}

Return the revised sample JSON:"""),
        ])

    # ------------------------------------------------------------------
    # Critique + revise helpers
    # ------------------------------------------------------------------
    async def _critique(
        self, source: str, samples: List[dict]
    ) -> List[dict]:
        """Return one verdict dict per sample: {"index","verdict","reason"}."""
        if not samples:
            return []
        prompt = self._critic_prompt()
        chain = prompt | self.critic_llm | self.parser
        try:
            raw = await asyncio.wait_for(
                chain.ainvoke({
                    "text": source[:self.source_char_limit],
                    "samples": json.dumps(samples, ensure_ascii=False),
                }),
                timeout=self.llm_timeout,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise map_llm_error(exc) from exc

        verdicts = self._extract_json_array(raw)
        # Normalise to a list of dicts with index/verdict/reason.
        norm: List[dict] = []
        for i, v in enumerate(verdicts):
            if isinstance(v, dict):
                norm.append({
                    "index": v.get("index", i),
                    "verdict": str(v.get("verdict", "ok")).lower(),
                    "reason": str(v.get("reason", "")),
                })
            else:
                norm.append({"index": i, "verdict": "ok", "reason": ""})
        # Pad / trim to match sample count.
        while len(norm) < len(samples):
            norm.append({"index": len(norm), "verdict": "ok", "reason": ""})
        return norm[: len(samples)]

    async def _revise_one(
        self, source: str, sample: dict, critique: str
    ) -> dict:
        """Return a revised single sample (raw dict)."""
        prompt = self._revise_prompt()
        chain = prompt | self.llm | self.parser
        try:
            raw = await asyncio.wait_for(
                chain.ainvoke({
                    "text": source[:self.source_char_limit],
                    "sample": json.dumps(sample, ensure_ascii=False),
                    "critique": critique,
                }),
                timeout=self.llm_timeout,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise map_llm_error(exc) from exc

        # The reviser should return a single object, not an array, but be
        # lenient: accept a one-element array too.
        parsed = self._extract_json_array(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        # Fallback: try to parse a bare object.
        try:
            obj = json.loads(raw.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Last resort: return the original unchanged.
        return sample

    # ------------------------------------------------------------------
    # QA subgraph nodes
    # ------------------------------------------------------------------
    # The per-batch generate -> critique -> revise loop is modelled as a
    # LangGraph StateGraph (see `build_qa_graph`). One round of the graph
    # loop = one batch-level revise pass over all samples flagged by the
    # critic. The non-streaming `generate` drives the compiled graph via
    # `ainvoke`; `generate_stream` walks the same node methods manually so
    # it can emit per-step SSE events in a stable order (same pattern as
    # `ResearchAgent.run_stream`).
    async def _qa_generate_node(self, state: QAState) -> dict:
        """Step 1: produce candidate samples via the format handler."""
        handler = state["handler"]
        candidates = await handler(state["source"], state["batch_size"])
        return {
            "candidates": candidates,
            "current": candidates,
            "accepted": [],
            "round": 0,
            "pending": [],
        }

    async def _qa_critique_node(self, state: QAState) -> dict:
        """Step 2: critique the current samples, partition into accepted/pending."""
        current = state.get("current", [])
        if not current:
            return {"pending": [], "verdicts": []}
        verdicts = await self._critique(state["source"], current)
        accepted = list(state.get("accepted", []))
        pending: List[dict] = []
        for i, v in enumerate(verdicts):
            if i >= len(current):
                continue
            if v.get("verdict") != "revise":
                accepted.append(current[i])
            else:
                pending.append({"sample": current[i], "reason": v.get("reason", "")})
        return {"verdicts": verdicts, "accepted": accepted, "pending": pending}

    async def _qa_revise_node(self, state: QAState) -> dict:
        """Step 3: revise each pending sample once, producing the next `current`."""
        pending = state.get("pending", [])
        source = state["source"]
        validate = state["validate"]
        new_current: List[dict] = []
        for item in pending:
            sample = item["sample"]
            reason = item["reason"]
            try:
                revised = await self._revise_one(source, sample, reason)
            except GenerationError:
                new_current.append(sample)
                continue
            revalidated = validate([revised], 1)
            new_current.append(revalidated[0] if revalidated else sample)
        return {
            "current": new_current,
            "round": state.get("round", 0) + 1,
            "pending": [],
        }

    async def _qa_finalize_node(self, state: QAState) -> dict:
        """Step 4: accept any still-current samples (rounds exhausted) and validate."""
        accepted = list(state.get("accepted", []))
        accepted.extend(state.get("current", []))
        samples = state["validate"](accepted, state["batch_size"])
        return {"samples": samples}

    def _qa_route_after_critique(self, state: QAState) -> str:
        """Conditional edge: revise while pending samples remain and rounds left."""
        if not state.get("pending"):
            return "finalize"
        if state.get("round", 0) >= self.max_revise_rounds:
            return "finalize"
        return "revise"

    def build_qa_graph(self) -> Any:
        """Return a compiled LangGraph StateGraph for the per-batch QA loop.

        Exposed so callers can compose the QA pipeline into a larger graph.
        `generate` drives this via `ainvoke`; `generate_stream` walks the
        node methods directly for stable event ordering.
        """
        g = StateGraph(QAState)
        g.add_node("generate", self._qa_generate_node)
        g.add_node("critique", self._qa_critique_node)
        g.add_node("revise", self._qa_revise_node)
        g.add_node("finalize", self._qa_finalize_node)
        g.set_entry_point("generate")
        g.add_edge("generate", "critique")
        g.add_conditional_edges(
            "critique",
            self._qa_route_after_critique,
            {"revise": "revise", "finalize": "finalize"},
        )
        g.add_edge("revise", "critique")
        g.add_edge("finalize", END)
        return g.compile()

    async def _generate_batch_with_qa(
        self,
        handler,
        source: str,
        batch_size: int,
        validate,
    ) -> List[dict]:
        """Run the QA subgraph for one batch, return validated samples."""
        init: QAState = {
            "source": source[:self.source_char_limit],
            "batch_size": batch_size,
            "validate": validate,
            "handler": handler,
        }
        result = await self._qa_graph.ainvoke(init)
        return list(result.get("samples", []))

    # ------------------------------------------------------------------
    # Public API mirrors DatasetAgent.generate / generate_stream
    # ------------------------------------------------------------------
    async def generate(
        self,
        text: str,
        format_type: Literal["alpaca", "chatml", "sharegpt", "dpo", "completion"],
        num_samples: int = 5,
    ) -> List[dict]:
        """High-accuracy generate (non-streaming)."""
        handler = {
            "alpaca": self.generate_alpaca,
            "chatml": self.generate_chatml,
            "sharegpt": self.generate_sharegpt,
            "dpo": self.generate_dpo,
            "completion": self.generate_completion,
        }.get(format_type)
        if handler is None:
            raise ValueError(f"Unknown format type: {format_type}")
        validate = self._validator_for(format_type)

        if num_samples <= self.batch_size:
            return await self._generate_batch_with_qa(
                handler, text[:self.source_char_limit], num_samples, validate
            )

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        total_attempts = num_batches + self.extra_attempts

        async def run_batch(attempt: int) -> List[dict]:
            window = self._window(text, attempt, total_attempts)
            try:
                return await self._generate_batch_with_qa(
                    handler, window, self.batch_size, validate
                )
            except GenerationError:
                raise
            except Exception as exc:
                logger.warning("QA batch %d failed: %s", attempt, exc)
                return []

        sem = asyncio.Semaphore(self.max_concurrency)

        async def guarded(attempt: int) -> List[dict]:
            async with sem:
                return await run_batch(attempt)

        batch_results = await asyncio.gather(
            *(guarded(i) for i in range(total_attempts))
        )

        results: List[dict] = []
        seen: set[str] = set()
        for batch in batch_results:
            for item in batch:
                key = self._dedupe_key(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                results.append(item)
            if len(results) >= num_samples:
                break
        return results[:num_samples]

    async def generate_stream(
        self,
        text: str,
        format_type: Literal["alpaca", "chatml", "sharegpt", "dpo", "completion"],
        num_samples: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """High-accuracy streaming generate.

        Same event protocol as `DatasetAgent.generate_stream`, plus extra
        per-batch events:
          - {"type": "critique_start", "index"}
          - {"type": "revise_start", "index", "count"}
          - {"type": "revise_done", "index", "count"}
        """
        handler = {
            "alpaca": self.generate_alpaca,
            "chatml": self.generate_chatml,
            "sharegpt": self.generate_sharegpt,
            "dpo": self.generate_dpo,
            "completion": self.generate_completion,
        }.get(format_type)
        if handler is None:
            raise ValueError(f"Unknown format type: {format_type}")
        validate = self._validator_for(format_type)

        if num_samples <= self.batch_size:
            num_batches = 1
            extra = 0
        else:
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            extra = self.extra_attempts
        total_attempts = num_batches + extra

        yield {
            "type": "start",
            "format_type": format_type,
            "num_samples": num_samples,
            "num_batches": total_attempts,
            "batch_size": self.batch_size,
            "mode": "quality",
        }

        results: List[dict] = []
        seen: set[str] = set()
        completed_batches = 0

        for attempt in range(total_attempts):
            if len(results) >= num_samples:
                break

            window = self._window(text, attempt, total_attempts)
            batch_size = min(self.batch_size, num_samples - len(results))
            yield {
                "type": "batch_start",
                "index": attempt,
                "total": total_attempts,
                "batch_size": batch_size,
            }

            try:
                # Drive the QA subgraph nodes manually so SSE events emit
                # in stable order. Mirrors ResearchAgent.run_stream which
                # walks nodes by hand rather than relying on graph astream.
                source = window[:self.source_char_limit]
                state: QAState = {
                    "source": source,
                    "batch_size": batch_size,
                    "validate": validate,
                    "handler": handler,
                }

                # 1. Generate candidates.
                state.update(await self._qa_generate_node(state))
                yield {
                    "type": "generate_done",
                    "index": attempt,
                    "count": len(state.get("current", [])),
                }

                # 2-3. Critique -> (revise -> critique)* until clean / round cap.
                while True:
                    yield {"type": "critique_start", "index": attempt}
                    state.update(await self._qa_critique_node(state))
                    pending = state.get("pending", [])
                    if not pending:
                        break
                    if state.get("round", 0) >= self.max_revise_rounds:
                        # Rounds exhausted; pending samples are accepted as-is
                        # by finalize. No revision happens this round.
                        yield {
                            "type": "revise_done",
                            "index": attempt,
                            "count": len(pending),
                        }
                        break
                    yield {"type": "revise_start", "index": attempt, "count": len(pending)}
                    state.update(await self._qa_revise_node(state))
                    yield {
                        "type": "revise_done",
                        "index": attempt,
                        "count": len(pending),
                    }

                # 4. Finalize: accept remaining current + validate + dedupe.
                state.update(await self._qa_finalize_node(state))
                batch_samples = state.get("samples", [])
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
                yield {"type": "warning", "message": str(exc.user_message), "index": attempt}
                logger.warning("QA batch %d failed in stream: %s", attempt, exc)
                completed_batches += 1
                yield {
                    "type": "progress",
                    "done": completed_batches,
                    "total": total_attempts,
                    "samples_so_far": len(results),
                }
                continue
            except Exception as exc:
                logger.warning("QA batch %d failed: %s", attempt, exc)
                yield {"type": "warning", "message": str(exc), "index": attempt}
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
