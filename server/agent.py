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
    # --- Fact-driven pipeline (QualityAgent) ---
    # When true, QualityAgent runs: extract facts -> reconcile -> plan per
    # fact -> canonical answers -> generate per manifest -> critique -> revise.
    # When false, QualityAgent falls back to the original batched loop.
    "plan_per_fact": True,
    "canonicalize_facts": True,
    "capacity_gate": True,
    # Max distinct samples (paraphrases) that may be generated per fact.
    "max_paraphrases_per_fact": 2,
    # Inject out-of-scope samples whose assistant answer is a polite refusal.
    "allow_negatives": False,
    "negatives_ratio": 0.1,
    # Dedup strategy: "exact" | "question_hash" | "embedding".
    "dedup_mode": "question_hash",
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
    out["plan_per_fact"] = bool(out["plan_per_fact"])
    out["canonicalize_facts"] = bool(out["canonicalize_facts"])
    out["capacity_gate"] = bool(out["capacity_gate"])
    out["max_paraphrases_per_fact"] = int(out["max_paraphrases_per_fact"])
    out["allow_negatives"] = bool(out["allow_negatives"])
    out["negatives_ratio"] = float(out["negatives_ratio"])
    out["dedup_mode"] = str(out["dedup_mode"])
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
        self.dedup_mode = s["dedup_mode"]
        self.max_paraphrases_per_fact = s["max_paraphrases_per_fact"]
        # `_dedupe_key` is a staticmethod and reads the mode off the class
        # attribute so it stays usable from QualityAgent (which shares the
        # method) without threading an instance arg through every call site.
        DatasetAgent._dedup_mode_runtime = self.dedup_mode
    
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
    def _question_text(item: dict) -> str:
        """Return the user-facing question/prompt portion of a sample.

        Used by dedup so two samples that differ only in the system role or
        the assistant answer wording still count as the *same* question and
        can be deduped. Returns "" if no recognizable question field exists.
        """
        if "messages" in item and isinstance(item["messages"], list):
            for m in item["messages"]:
                if isinstance(m, dict) and m.get("role") == "user":
                    return str(m.get("content", "")).strip().lower()
            # Fallback: join everything (no user turn found).
            return " ".join(
                str(m.get("content", "")).strip().lower()
                for m in item["messages"]
                if isinstance(m, dict)
            )
        if "conversations" in item and isinstance(item["conversations"], list):
            for m in item["conversations"]:
                if isinstance(m, dict) and m.get("from") == "human":
                    return str(m.get("value", "")).strip().lower()
            return " ".join(
                str(m.get("value", "")).strip().lower()
                for m in item["conversations"]
                if isinstance(m, dict)
            )
        if "instruction" in item:
            return str(item.get("instruction", "")).strip().lower()
        if "prompt" in item and "chosen" in item:
            return str(item.get("prompt", "")).strip().lower()
        if "text" in item:
            return str(item["text"]).strip().lower()
        return ""

    @staticmethod
    def _dedupe_key(item: dict) -> str:
        """Return a canonical key used to detect duplicate samples.

        Comparison keys on the *question/prompt* portion only (see
        `_question_text`), not the entire sample. This means 10 samples with
        the same question but different system roles / answer phrasings now
        collapse to a single entry instead of all surviving dedup.

        A small `dedup_mode` knob (config `[generation].dedup_mode`) selects
        between:
          - "exact":         full question string (default-ish, tightest)
          - "question_hash": normalized token bag (whitespace/punct-insensitive)
          - "embedding":     same as "question_hash" for now; an embedding
                             backend can be plugged in later without changing
                             the call sites.
        """
        q = DatasetAgent._question_text(item)
        if not q:
            # Fall back to the whole-object hash so empty/unknown shapes
            # still dedupe against themselves.
            return json.dumps(item, sort_keys=True).lower()
        mode = getattr(DatasetAgent, "_dedup_mode_runtime", "question_hash")
        if mode == "exact":
            return q
        # Normalized token bag: lowercase, strip punctuation, sort tokens.
        # This treats "What is Sidharth's focus?" and "what is sidharth's focus?"
        # as the same question, while still distinguishing different questions.
        tokens = re.sub(r"[^a-z0-9\s]", " ", q).split()
        tokens.sort()
        return " ".join(tokens)

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
        self.plan_per_fact = s["plan_per_fact"]
        self.canonicalize_facts = s["canonicalize_facts"]
        self.capacity_gate = s["capacity_gate"]
        self.max_paraphrases_per_fact = s["max_paraphrases_per_fact"]
        self.allow_negatives = s["allow_negatives"]
        self.negatives_ratio = s["negatives_ratio"]
        self._qa_graph = self.build_qa_graph()

    # ------------------------------------------------------------------
    # Prompts + critique/revise helpers
    # ------------------------------------------------------------------
    # The critic and revise prompts/handlers are defined later in this
    # class (in the fact-driven pipeline section) where they are extended
    # with uniqueness / consistency / canonical-answer checks. They are
    # referenced by the QA subgraph nodes below via `self._critic_prompt()`
    # / `self._critique()` / `self._revise_one()`.

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
    # Fact-driven pipeline (Phases 1-9)
    # ------------------------------------------------------------------
    # When `plan_per_fact` is enabled in config, QualityAgent replaces the
    # batched generate->critique->revise loop with a fact-anchored pipeline:
    #
    #   1. extract_facts     - one LLM call returns discrete facts w/ evidence
    #   2. reconcile_facts   - collapse ambiguous / conflicting facts so each
    #                          fact has a single canonical statement
    #   3. plan_manifest     - decide how many samples per fact (<= k), plus
    #                          optional negative (out-of-scope) slots
    #   4. canonical_answers - one answer per fact; paraphrases must agree
    #   5. generate_anchored - each sample generated with its fact_id + plan
    #   6. critique           - 6 criteria incl. uniqueness & consistency
    #   7. revise             - reviser sees the canonical answer as ground truth
    #   8. dedup              - per-fact cap + question-aware dedup (see
    #                          `_dedupe_key`)
    #   9. capacity_gate      - if requested > capacity, stop early + warn

    def _fact_extraction_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a fact extractor. Given a source text, return a JSON array of DISTINCT, ATOMIC facts that can be learned from it.

Each fact must be:
- Atomic: a single, standalone piece of information (one fact per entry, NOT a cluster).
- Faithful: directly supported by the source text (quote an evidence span).
- Non-duplicative: do NOT repeat the same fact in different wording. If the source says the same thing twice, emit it ONCE and prefer the most specific phrasing.

For each fact output: {{"id": "f1", "fact": "<one sentence>", "evidence": "<short quote from source>", "confidence": 0.0-1.0}}

Order facts by importance / centrality to the source. Return ONLY the JSON array."""),
            ("human", """Source text:
{text}

Return the facts JSON array:"""),
        ])

    def _reconcile_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You reconcile a list of facts that may overlap or contradict each other.
Return a single deduplicated, consistent list of facts. When two facts say the same thing in different wording, keep the more SPECIFIC/COMPLETE one and drop the other. When two facts contradict, keep the one with the stronger evidence (longer quote, more specific) and drop the other. Preserve original ids when possible; renumber only if needed.
Return ONLY a JSON array of {{"id","fact","evidence","confidence"}} objects."""),
            ("human", """Facts (JSON):
{facts}

Return the reconciled facts JSON array:"""),
        ])

    def _plan_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a sample planner. Given a list of facts and a target sample count, decide how many training samples to generate for each fact.

Rules:
- Each fact may get between 0 and {max_per_fact} samples. Most facts should get 1; only central/important facts should get 2.
- The total across all facts MUST be <= {capacity} (do not exceed the source's information capacity).
- If {allow_negatives} is true, reserve roughly {neg_ratio_pct}% of the slots as "negative" samples whose answer is a polite refusal ("I don't have that information based on what I know."). Negatives must ask about facts NOT present in the source.
- Distribute samples across facts broadly; do NOT concentrate samples on a single fact.

Return a JSON array of manifest entries, each:
{{"fact_id": "f1", "kind": "positive|negative", "style": "<short style hint e.g. direct-question|rephrase|role-play|comparison|out-of-scope>"}}

Return ONLY the JSON array."""),
            ("human", """Facts (JSON):
{facts}

Capacity (max samples): {capacity}
Requested: {requested}

Return the manifest JSON array:"""),
        ])

    def _canonical_answer_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a canonical answer writer. Given a source text and one fact, write the SINGLE authoritative answer to the question "What about <fact>?". This answer is the ground truth; all paraphrases of the question must produce an answer consistent with it. Be precise, complete, and faithful to the source. Do not add information that is not in the source.
Return ONLY a single sentence/paragraph (no JSON)."""),
            ("human", """Source text:
{text}

Fact:
{fact}

Return the canonical answer:"""),
        ])

    def _anchored_generate_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset generator. Produce ONE training sample anchored to a specific fact and its canonical answer.

Hard rules:
- The sample's assistant/value/output MUST be consistent with the canonical answer. Do NOT introduce facts that are not in the canonical answer.
- The user/value/instruction/question MUST match the requested style hint.
- If kind is "negative", the question must ask about something NOT covered by the source, and the answer MUST be a polite refusal: "I don't have that information based on what I know."
- Match the requested output schema EXACTLY.
- Return ONLY the single sample object as JSON (no array, no prose)."""),
            ("human", """Source text (for reference):
{text}

Fact:
{fact}

Canonical answer (ground truth):
{canonical_answer}

Requested style:
{style}

Kind: {kind}

Format: {format_spec}

Return the single sample JSON:"""),
        ])

    async def _extract_facts(self, text: str) -> List[dict]:
        """Phase 1: return a list of {id, fact, evidence, confidence} dicts."""
        prompt = self._fact_extraction_prompt()
        raw = await self._run_chain(prompt, {"text": text[: self.source_char_limit]})
        facts = self._extract_json_array(raw)
        norm: List[dict] = []
        for i, f in enumerate(facts):
            if not isinstance(f, dict):
                continue
            norm.append({
                "id": str(f.get("id", f"f{i + 1}")),
                "fact": str(f.get("fact", "")).strip(),
                "evidence": str(f.get("evidence", "")).strip(),
                "confidence": float(f.get("confidence", 1.0)),
            })
        return [f for f in norm if f["fact"]]

    async def _reconcile_facts(self, facts: List[dict], text: str) -> List[dict]:
        """Phase 2: collapse overlapping / conflicting facts."""
        if not facts or len(facts) <= 1:
            return facts
        prompt = self._reconcile_prompt()
        raw = await self._run_chain(
            prompt,
            {
                "facts": json.dumps(facts, ensure_ascii=False),
                "text": text[: self.source_char_limit],
            },
        )
        out = self._extract_json_array(raw)
        norm: List[dict] = []
        for i, f in enumerate(out):
            if not isinstance(f, dict):
                continue
            norm.append({
                "id": str(f.get("id", f"f{i + 1}")),
                "fact": str(f.get("fact", "")).strip(),
                "evidence": str(f.get("evidence", "")).strip(),
                "confidence": float(f.get("confidence", 1.0)),
            })
        return norm or facts

    def _compute_capacity(self, num_facts: int, requested: int) -> int:
        """Phase 2b: max samples the source can support without forcing.

        capacity = num_facts * max_paraphrases_per_fact (optionally +negatives).
        If `capacity_gate` is off, returns `requested` (legacy behaviour).
        """
        if not self.capacity_gate:
            return requested
        base = num_facts * max(1, self.max_paraphrases_per_fact)
        if self.allow_negatives:
            base = int(base * (1.0 + self.negatives_ratio))
        return base

    async def _plan_manifest(
        self, facts: List[dict], capacity: int, requested: int
    ) -> List[dict]:
        """Phase 3: produce a manifest of {fact_id, kind, style} entries."""
        prompt = self._plan_prompt()
        neg_pct = int(self.negatives_ratio * 100) if self.allow_negatives else 0
        raw = await self._run_chain(
            prompt,
            {
                "facts": json.dumps(facts, ensure_ascii=False),
                "capacity": capacity,
                "requested": min(requested, capacity),
                "max_per_fact": self.max_paraphrases_per_fact,
                "allow_negatives": "true" if self.allow_negatives else "false",
                "neg_ratio_pct": neg_pct,
            },
        )
        manifest = self._extract_json_array(raw)
        norm: List[dict] = []
        valid_ids = {f["id"] for f in facts}
        for m in manifest:
            if not isinstance(m, dict):
                continue
            kind = str(m.get("kind", "positive")).lower()
            if kind not in ("positive", "negative"):
                kind = "positive"
            entry = {
                "fact_id": str(m.get("fact_id", "")),
                "kind": kind,
                "style": str(m.get("style", "direct-question")),
            }
            # Negative slots have no fact_id by design; positive must reference a real fact.
            if kind == "positive" and entry["fact_id"] not in valid_ids:
                # Skip orphan positive entries.
                continue
            norm.append(entry)
        # Hard cap at capacity so the planner can never exceed it.
        return norm[:capacity]

    async def _canonical_answer(self, text: str, fact: dict) -> str:
        """Phase 4: produce one canonical answer for a fact."""
        prompt = self._canonical_answer_prompt()
        raw = await self._run_chain(
            prompt,
            {
                "text": text[: self.source_char_limit],
                "fact": fact.get("fact", ""),
            },
        )
        return raw.strip()

    async def _generate_anchored_sample(
        self,
        text: str,
        fact: dict,
        canonical_answer: str,
        style: str,
        kind: str,
        format_type: str,
    ) -> dict:
        """Phase 5: generate one sample anchored to a fact + canonical answer."""
        prompt = self._anchored_generate_prompt()
        format_spec = self._format_spec_for(format_type)
        raw = await self._run_chain(
            prompt,
            {
                "text": text[: self.source_char_limit],
                "fact": fact.get("fact", "") if kind == "positive" else "(out-of-scope)",
                "canonical_answer": canonical_answer if kind == "positive" else "(none)",
                "style": style,
                "kind": kind,
                "format_spec": format_spec,
            },
        )
        parsed = self._extract_json_array(raw)
        if parsed and isinstance(parsed[0], dict):
            return parsed[0]
        try:
            obj = json.loads(raw.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return {}

    def _format_spec_for(self, format_type: str) -> str:
        """Return a short, machine-readable schema string per format."""
        return {
            "alpaca": '{"instruction": str, "input": str, "output": str}',
            "chatml": '{"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}',
            "sharegpt": '{"conversations": [{"from": "human"|"gpt", "value": str}, ...]}',
            "dpo": '{"prompt": str, "chosen": str, "rejected": str}',
            "completion": '{"text": str}',
        }.get(format_type, "")

    async def _generate_fact_driven(
        self,
        text: str,
        format_type: str,
        num_samples: int,
        validate,
        on_event=None,
    ) -> List[dict]:
        """Run the full Phase 1-8 fact-driven pipeline. Returns final samples.

        `on_event` is an optional async callback `await on_event(event_dict)`
        used by the streaming path to forward SSE-style events. When None,
        events are dropped (non-streaming caller just wants the final list).

        Emits these event types (when `on_event` is provided):
          - {"type": "facts_extracted", "count": int}
          - {"type": "facts_reconciled", "count": int}
          - {"type": "capacity_warning", "capacity": int, "requested": int}
          - {"type": "plan_done", "count": int}
          - {"type": "canonical_done", "count": int}
          - {"type": "sample_generated", "index": int, "fact_id": str, "kind": str}
          - {"type": "critique_start"}
          - {"type": "revise_start", "count": int}
          - {"type": "revise_done", "count": int}
          - {"type": "progress", "done": int, "total": int, "samples_so_far": int}
          - {"type": "complete", "data": [...], "count": int, "undergenerated": bool}
        """
        source = text[: self.source_char_limit]

        async def emit(ev: dict) -> None:
            if on_event is not None:
                await on_event(ev)

        # Phase 1: extract facts.
        facts = await self._extract_facts(source)
        await emit({"type": "facts_extracted", "count": len(facts)})
        if not facts:
            await emit({"type": "complete", "data": [], "count": 0, "undergenerated": True})
            return []

        # Phase 2: reconcile (collapse overlaps / contradictions).
        if self.canonicalize_facts:
            facts = await self._reconcile_facts(facts, source)
        await emit({"type": "facts_reconciled", "count": len(facts)})

        # Phase 2b: capacity gate (don't force).
        capacity = self._compute_capacity(len(facts), num_samples)
        requested = min(num_samples, capacity)
        undergenerated = self.capacity_gate and num_samples > capacity
        if undergenerated:
            await emit({
                "type": "capacity_warning",
                "capacity": capacity,
                "requested": num_samples,
            })

        # Phase 3: plan the manifest.
        manifest = await self._plan_manifest(facts, capacity, requested)
        await emit({"type": "plan_done", "count": len(manifest)})

        # Phase 4: canonical answers (one per fact; bounded concurrency).
        canonical: Dict[str, str] = {}
        sem = asyncio.Semaphore(self.max_concurrency)

        async def canon_one(fact: dict) -> None:
            async with sem:
                canonical[fact["id"]] = await self._canonical_answer(source, fact)

        await asyncio.gather(*(canon_one(f) for f in facts))
        await emit({"type": "canonical_done", "count": len(canonical)})

        # Phase 5 + 6 + 7: generate anchored samples, then critique/revise.
        validate_fn = validate
        results: List[dict] = []
        seen_keys: set[str] = set()
        accepted_questions: List[str] = []
        per_fact_count: Dict[str, int] = {}
        total = len(manifest)

        for i, entry in enumerate(manifest):
            fact_id = entry.get("fact_id", "")
            kind = entry.get("kind", "positive")
            style = entry.get("style", "direct-question")
            fact = next((f for f in facts if f["id"] == fact_id), {})
            canon = canonical.get(fact_id, "") if kind == "positive" else ""

            try:
                sample = await self._generate_anchored_sample(
                    source, fact, canon, style, kind, format_type
                )
            except GenerationError as exc:
                logger.warning("Anchored gen %d failed: %s", i, exc)
                sample = {}
            except Exception as exc:
                logger.warning("Anchored gen %d failed: %s", i, exc)
                sample = {}

            await emit({
                "type": "sample_generated",
                "index": i,
                "fact_id": fact_id,
                "kind": kind,
            })

            if not sample:
                await emit({
                    "type": "progress",
                    "done": i + 1,
                    "total": total,
                    "samples_so_far": len(results),
                })
                continue

            # Phase 8 (a): per-fact cap + question dedup BEFORE critique so
            # we never exceed max_paraphrases_per_fact for a single fact.
            key = self._dedupe_key(sample)
            if kind == "positive":
                if per_fact_count.get(fact_id, 0) >= self.max_paraphrases_per_fact:
                    await emit({
                        "type": "progress",
                        "done": i + 1,
                        "total": total,
                        "samples_so_far": len(results),
                    })
                    continue
            if not key or key in seen_keys:
                await emit({
                    "type": "progress",
                    "done": i + 1,
                    "total": total,
                    "samples_so_far": len(results),
                })
                continue

            # Phase 6: critique (extended).
            await emit({"type": "critique_start"})
            try:
                verdicts = await self._critique(
                    source,
                    [sample],
                    accepted_questions=accepted_questions,
                    canonical_answers={fact_id: canon} if canon else {},
                )
            except GenerationError as exc:
                logger.warning("Critique %d failed: %s", i, exc)
                verdicts = [{"index": 0, "verdict": "ok", "reason": ""}]

            current = sample
            rounds = 0
            while verdicts and verdicts[0].get("verdict") == "revise" and rounds < self.max_revise_rounds:
                await emit({"type": "revise_start", "count": 1})
                try:
                    revised = await self._revise_one(
                        source, current, verdicts[0].get("reason", ""),
                        canonical_answer=canon,
                    )
                except GenerationError as exc:
                    logger.warning("Revise %d failed: %s", i, exc)
                    revised = current
                current = revised
                rounds += 1
                await emit({"type": "revise_done", "count": 1})
                try:
                    verdicts = await self._critique(
                        source,
                        [current],
                        accepted_questions=accepted_questions,
                        canonical_answers={fact_id: canon} if canon else {},
                    )
                except GenerationError as exc:
                    logger.warning("Critique (revise) %d failed: %s", i, exc)
                    break

            # Phase 8 (b): re-validate schema, then accept.
            validated = validate_fn([current], 1)
            if not validated:
                await emit({
                    "type": "progress",
                    "done": i + 1,
                    "total": total,
                    "samples_so_far": len(results),
                })
                continue

            final = validated[0]
            # Re-check dedup after potential revision changes the question.
            final_key = self._dedupe_key(final)
            if not final_key or final_key in seen_keys:
                await emit({
                    "type": "progress",
                    "done": i + 1,
                    "total": total,
                    "samples_so_far": len(results),
                })
                continue
            seen_keys.add(final_key)
            if kind == "positive":
                per_fact_count[fact_id] = per_fact_count.get(fact_id, 0) + 1
            results.append(final)
            accepted_questions.append(self._question_text(final))
            await emit({
                "type": "progress",
                "done": i + 1,
                "total": total,
                "samples_so_far": len(results),
            })

        trimmed = results[:requested]
        await emit({
            "type": "complete",
            "data": trimmed,
            "count": len(trimmed),
            "undergenerated": undergenerated or len(trimmed) < num_samples,
        })
        return trimmed

    # ---- Critic (Phase 6): extended criteria incl. uniqueness/consistency
    def _critic_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a strict quality reviewer for fine-tuning datasets.
You are given a source text and a JSON array of candidate training samples.
Score EACH sample against these criteria:
- Faithfulness: claims must be supported by the source text.
- Clarity: instruction/question is unambiguous.
- Completeness: the output fully answers the instruction.
- Format: matches the requested schema exactly.
- Uniqueness: the sample's question is NOT a near-duplicate of another sample's question already accepted (provided as `accepted_questions`).
- Consistency: if a `canonical_answer` is provided, the sample's answer must agree with it (no new facts, no contradictions).
- Out-of-scope: if the sample is marked kind="negative", the answer must be a polite refusal, NOT a fabricated fact.

Return ONLY a JSON array (no prose) with one entry per input sample, in order.
Each entry: {{"index": <0-based>, "verdict": "ok" | "revise", "reason": "<short reason or empty>"}}"""),
            ("human", """Source text:
{text}

Accepted questions so far (for uniqueness check):
{accepted_questions}

Canonical answers keyed by fact_id (for consistency check):
{canonical_answers}

Candidate samples (JSON array):
{samples}

Return the JSON verdict array:"""),
        ])

    async def _critique(
        self,
        source: str,
        samples: List[dict],
        accepted_questions: Optional[List[str]] = None,
        canonical_answers: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Phase 6: extended critique with uniqueness + consistency checks."""
        if not samples:
            return []
        prompt = self._critic_prompt()
        chain = prompt | self.critic_llm | self.parser
        try:
            raw = await asyncio.wait_for(
                chain.ainvoke({
                    "text": source[: self.source_char_limit],
                    "samples": json.dumps(samples, ensure_ascii=False),
                    "accepted_questions": json.dumps(
                        accepted_questions or [], ensure_ascii=False
                    ),
                    "canonical_answers": json.dumps(
                        canonical_answers or {}, ensure_ascii=False
                    ),
                }),
                timeout=self.llm_timeout,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise map_llm_error(exc) from exc

        verdicts = self._extract_json_array(raw)
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
        while len(norm) < len(samples):
            norm.append({"index": len(norm), "verdict": "ok", "reason": ""})
        return norm[: len(samples)]

    # ---- Reviser (Phase 7): sees canonical answer as ground truth
    def _revise_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dataset editor. Given one training sample and a critique, rewrite the sample to fix the problems. Keep the same schema.

If a `canonical_answer` is provided, the revised answer MUST agree with it (no new facts, no contradictions). If kind is "negative", the answer must remain a polite refusal.

Return ONLY the single revised sample object as JSON, no array, no prose."""),
            ("human", """Original sample:
{sample}

Critique:
{critique}

Canonical answer (ground truth, if any):
{canonical_answer}

Source text (for reference):
{text}

Return the revised sample JSON:"""),
        ])

    async def _revise_one(
        self,
        source: str,
        sample: dict,
        critique: str,
        canonical_answer: Optional[str] = None,
    ) -> dict:
        """Phase 7: revise one sample, anchored to the canonical answer."""
        prompt = self._revise_prompt()
        chain = prompt | self.llm | self.parser
        try:
            raw = await asyncio.wait_for(
                chain.ainvoke({
                    "text": source[: self.source_char_limit],
                    "sample": json.dumps(sample, ensure_ascii=False),
                    "critique": critique,
                    "canonical_answer": canonical_answer or "(none)",
                }),
                timeout=self.llm_timeout,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise map_llm_error(exc) from exc

        parsed = self._extract_json_array(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        try:
            obj = json.loads(raw.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return sample

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

        # Fact-driven pipeline (Phases 1-8) when enabled; otherwise the
        # original batched generate->critique->revise loop.
        if self.plan_per_fact:
            return await self._generate_fact_driven(
                text, format_type, num_samples, validate, on_event=None
            )

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

        # Fact-driven pipeline: forward the fact-driven events straight
        # through, wrapped in a "start" / final "complete".
        if self.plan_per_fact:
            yield {
                "type": "start",
                "format_type": format_type,
                "num_samples": num_samples,
                "mode": "quality",
                "pipeline": "fact_driven",
            }
            # The fact-driven pipeline calls `on_event` synchronously while
            # it awaits; we bridge those callbacks into this generator by
            # pushing events onto a queue and yielding them in order.
            ev_queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
            _SENTINEL = object()

            async def on_event(ev: dict) -> None:
                await ev_queue.put(ev)

            async def run_pipeline() -> None:
                try:
                    await self._generate_fact_driven(
                        text, format_type, num_samples, validate, on_event=on_event
                    )
                except Exception as exc:
                    await ev_queue.put({"type": "error", "message": str(exc)})
                finally:
                    await ev_queue.put(None)  # type: ignore[arg-type]

            task = asyncio.create_task(run_pipeline())
            try:
                while True:
                    ev = await ev_queue.get()
                    if ev is None:
                        break
                    yield ev
            finally:
                if not task.done():
                    task.cancel()
                await task
            return

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
