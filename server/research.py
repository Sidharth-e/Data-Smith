"""
LangGraph research agent for Data Smith.

Given a topic, this agent runs a small multi-node pipeline and streams its
progress back to the client as Server-Sent Events:

  planner  -> break the topic into 4-6 focused research questions
  researcher -> run the `web_search` tool (DuckDuckGo) for each question
  gap      -> compare collected snippets against the plan, ask follow-up
              questions for any angle still missing
  quality  -> drop duplicate / irrelevant / low-signal snippets
  writer   -> synthesize the surviving snippets into a single source
              document (markdown) suitable for feeding back into
              `DatasetAgent.generate_stream`

The DuckDuckGo tool here is intentionally thin and swappable: anyone can
replace it with a Tavily/Serper/Bing-backed `@tool` later without touching
the graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from config import load_config
from errors import GenerationError, map_llm_error
from model_factory import ModelFactory

logger = logging.getLogger("data_smith.research")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "max_searches": 6,
    "max_gap_rounds": 1,
    "llm_timeout": 600.0,
    "snippets_per_query": 3,
}


def _research_settings(config: Optional[dict] = None) -> dict:
    cfg = (config or load_config()).get("research", {})
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if v is not None})
    out["max_searches"] = int(out["max_searches"])
    out["max_gap_rounds"] = int(out["max_gap_rounds"])
    out["llm_timeout"] = float(out["llm_timeout"])
    out["snippets_per_query"] = int(out["snippets_per_query"])
    return out


# ---------------------------------------------------------------------------
# Web search tool (DuckDuckGo, swappable)
# ---------------------------------------------------------------------------
@tool
def web_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Search the public web via DuckDuckGo and return a list of snippets.

    Each result is a dict: {"title", "snippet", "url"}.
    `top_k` caps the number of results (default 3).

    Swap this function for a Tavily / Serper / Bing implementation if you
    have an API key; the rest of the agent only depends on this return
    shape.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:  # pragma: no cover - optional dep
        raise GenerationError(
            "duckduckgo_search is not installed. Run `pip install duckduckgo-search`.",
            detail={"reason": str(exc)},
        ) from exc

    results: List[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=top_k):
                results.append({
                    "title": str(r.get("title", "") or ""),
                    "snippet": str(r.get("body", r.get("snippet", "")) or ""),
                    "url": str(r.get("href", r.get("url", "")) or ""),
                })
    except Exception as exc:  # pragma: no cover - network / rate limit
        logger.warning("web_search(%r) failed: %s", query, exc)
    return results


ALL_TOOLS = [web_search]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    topic: str
    plan: List[str]
    queries_done: List[str]
    snippets: List[Dict[str, str]]
    gap_questions: List[str]
    gap_round: int
    quality_snippets: List[Dict[str, str]]
    document: str
    log: List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class ResearchAgent:
    """
    Multi-agent research pipeline implemented as a LangGraph StateGraph.

    Nodes:
      - planner
      - researcher
      - gap
      - quality
      - writer

    The graph runs synchronously inside `run_stream`, which wraps each node
    transition into an SSE event so the UI can show what each agent is doing
    in real time. LLM calls are async; the graph itself is small enough that
    we drive it node-by-node rather than relying on `astream` so we can emit
    per-agent events in a stable order.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[dict] = None,
    ):
        self.llm = llm or ModelFactory(config=config).create()
        self.parser = StrOutputParser()
        s = _research_settings(config)
        self.max_searches = s["max_searches"]
        self.max_gap_rounds = s["max_gap_rounds"]
        self.llm_timeout = s["llm_timeout"]
        self.snippets_per_query = s["snippets_per_query"]

    # ---- LLM helper ----
    async def _run_chain(self, prompt: ChatPromptTemplate, vars: dict) -> str:
        chain = prompt | self.llm | self.parser
        try:
            return await asyncio.wait_for(
                chain.ainvoke(vars), timeout=self.llm_timeout
            )
        except GenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise GenerationError(
                f"Research LLM call timed out after {self.llm_timeout}s"
            ) from exc
        except Exception as exc:
            raise map_llm_error(exc) from exc

    # ---- Prompts ----
    def _planner_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research planner. Given a topic, produce a focused
list of {n} research questions that, when answered, will give full coverage of the
topic for a fine-tuning dataset source document.

Return ONLY a JSON array of strings, no prose. Each string is one question."""),
            ("human", "Topic: {topic}\n\nReturn the JSON array of {n} questions:"),
        ])

    def _gap_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research gap analyst. You are given a topic, a list
of questions already researched, and the collected snippets.

Identify up to {n} important questions that are STILL NOT covered by the snippets
and would meaningfully improve the source document. If coverage is already good,
return an empty array.

Return ONLY a JSON array of strings (questions), no prose."""),
            ("human", """Topic: {topic}

Questions already researched:
{done}

Collected snippets (title | snippet):
{snippets}

Return the JSON array of up to {n} follow-up questions (or [] if none):"""),
        ])

    def _quality_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research quality reviewer. You are given a list of
search snippets (each with title, snippet, url). Drop duplicates, off-topic items,
and low-signal fragments. Keep the most informative, on-topic ones.

Return ONLY a JSON array of objects with the same keys: "title", "snippet", "url".
No prose."""),
            ("human", """Topic: {topic}

Snippets:
{snippets}

Return the filtered JSON array:"""),
        ])

    def _writer_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research writer. Given a topic and a set of source
snippets, write a single, well-structured markdown document that synthesizes the
information into a coherent reference suitable for generating fine-tuning samples.

Use headings (##) and bullet points where helpful. Cite source URLs inline as
[1], [2], ... and list them at the bottom under "## Sources". Do not invent facts
beyond the snippets. Aim for ~600-1200 words.

Return ONLY the markdown document, no commentary."""),
            ("human", """Topic: {topic}

Snippets:
{snippets}

Write the markdown source document:"""),
        ])

    # ---- JSON helper ----
    @staticmethod
    def _extract_json_array(text: str) -> list:
        import re

        # Fenced block first.
        fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if fence:
            try:
                parsed = json.loads(fence.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        # First balanced top-level array.
        start = text.find("[")
        if start != -1:
            depth = 0
            in_str = False
            quote = ""
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if ch == "\\":
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
                            try:
                                return json.loads(text[start:i + 1])
                            except json.JSONDecodeError:
                                break
        # Last resort: individual strings.
        return re.findall(r'"([^"]+\?)"', text)

    @staticmethod
    def _snippets_to_text(snippets: List[dict]) -> str:
        if not snippets:
            return "(none yet)"
        lines = []
        for i, s in enumerate(snippets, 1):
            title = str(s.get("title", "")).strip()
            snippet = str(s.get("snippet", "")).strip()
            url = str(s.get("url", "")).strip()
            lines.append(f"[{i}] {title}\n    {snippet}\n    {url}")
        return "\n".join(lines)

    # ---- Nodes (as async methods so we control streaming) ----
    async def planner_node(self, state: ResearchState) -> ResearchState:
        topic = state["topic"]
        prompt = self._planner_prompt()
        raw = await self._run_chain(prompt, {"topic": topic, "n": self.max_searches})
        plan = [q for q in self._extract_json_array(raw) if isinstance(q, str) and q.strip()]
        if not plan:
            plan = [topic]
        return {**state, "plan": plan, "queries_done": [], "snippets": [], "gap_round": 0}

    async def researcher_node(self, state: ResearchState) -> ResearchState:
        plan = state["plan"]
        done: List[str] = list(state.get("queries_done", []))
        snippets: List[dict] = list(state.get("snippets", []))

        # Cap total queries at max_searches across plan + gap follow-ups.
        remaining = self.max_searches - len(done)
        queries = plan[:remaining]
        for q in queries:
            if q in done:
                continue
            results = web_search.invoke({"query": q, "top_k": self.snippets_per_query})
            snippets.extend(results)
            done.append(q)
        return {**state, "snippets": snippets, "queries_done": done}

    async def gap_node(self, state: ResearchState) -> ResearchState:
        round_ = int(state.get("gap_round", 0))
        if round_ >= self.max_gap_rounds:
            return {**state, "gap_questions": []}
        prompt = self._gap_prompt()
        raw = await self._run_chain(prompt, {
            "topic": state["topic"],
            "done": json.dumps(state.get("queries_done", []), ensure_ascii=False),
            "snippets": self._snippets_to_text(state.get("snippets", [])),
            "n": max(1, self.max_searches // 2),
        })
        gap = [q for q in self._extract_json_array(raw) if isinstance(q, str) and q.strip()]
        return {**state, "gap_questions": gap, "gap_round": round_ + 1}

    async def quality_node(self, state: ResearchState) -> ResearchState:
        snippets = state.get("snippets", [])
        if not snippets:
            return {**state, "quality_snippets": []}
        prompt = self._quality_prompt()
        raw = await self._run_chain(prompt, {
            "topic": state["topic"],
            "snippets": json.dumps(snippets, ensure_ascii=False),
        })
        parsed = self._extract_json_array(raw)
        kept: List[dict] = []
        for item in parsed:
            if isinstance(item, dict):
                kept.append({
                    "title": str(item.get("title", "")),
                    "snippet": str(item.get("snippet", "")),
                    "url": str(item.get("url", "")),
                })
        if not kept:
            kept = snippets
        return {**state, "quality_snippets": kept}

    async def writer_node(self, state: ResearchState) -> ResearchState:
        prompt = self._writer_prompt()
        raw = await self._run_chain(prompt, {
            "topic": state["topic"],
            "snippets": json.dumps(state.get("quality_snippets", []), ensure_ascii=False),
        })
        return {**state, "document": raw.strip()}

    # ---- Streaming driver ----
    async def run_stream(self, topic: str) -> AsyncIterator[Dict[str, Any]]:
        """Run the full research pipeline, yielding SSE-friendly events.

        Event types:
          - {"type": "start", "topic"}
          - {"type": "agent_start", "agent": "planner"|"researcher"|"gap"|
             "quality"|"writer", "detail"?}
          - {"type": "agent_done", "agent"}  (emitted when a node finishes)
          - {"type": "agent_message", "agent", "message"}
          - {"type": "plan", "questions": [...]}
          - {"type": "search", "query", "results": [...]}
          - {"type": "snippets", "count", "snippets"?}
          - {"type": "gap_questions", "questions": [...]}
          - {"type": "quality_snippets", "count"}
          - {"type": "document_chunk", "content"}  (writer output streamed)
          - {"type": "document_done", "document"}
          - {"type": "complete", "document"}
          - {"type": "error", "message"}
        """
        if not topic or not topic.strip():
            yield {"type": "error", "message": "Topic is empty"}
            return

        yield {"type": "start", "topic": topic}

        state: ResearchState = {"topic": topic.strip()}

        try:
            # 1. Planner
            yield {"type": "agent_start", "agent": "planner"}
            state = await self.planner_node(state)
            plan = state.get("plan", [])
            yield {"type": "plan", "questions": plan}
            yield {
                "type": "agent_message",
                "agent": "planner",
                "message": f"Planned {len(plan)} research questions",
            }
            yield {"type": "agent_done", "agent": "planner"}

            # 2. Researcher (plan pass)
            yield {"type": "agent_start", "agent": "researcher"}
            state = await self.researcher_node(state)
            snippets = state.get("snippets", [])
            done = state.get("queries_done", [])
            for q in plan:
                if q in done:
                    yield {
                        "type": "search",
                        "query": q,
                        "results": [],  # results already merged; emit per-query shell
                    }
            yield {
                "type": "snippets",
                "count": len(snippets),
                "snippets": snippets[:12],
            }
            yield {
                "type": "agent_message",
                "agent": "researcher",
                "message": f"Collected {len(snippets)} snippets from {len(done)} queries",
            }
            yield {"type": "agent_done", "agent": "researcher"}

            # 3. Gap (one follow-up pass by default)
            yield {"type": "agent_start", "agent": "gap"}
            state = await self.gap_node(state)
            gap_qs = state.get("gap_questions", [])
            yield {"type": "gap_questions", "questions": gap_qs}
            if gap_qs:
                yield {
                    "type": "agent_message",
                    "agent": "gap",
                    "message": f"Found {len(gap_qs)} follow-up questions",
                }
                # Run researcher again for the gap questions (capacity permitting).
                state = {**state, "plan": gap_qs}
                state = await self.researcher_node(state)
                yield {
                    "type": "snippets",
                    "count": len(state.get("snippets", [])),
                    "snippets": state.get("snippets", [])[:12],
                }
            else:
                yield {
                    "type": "agent_message",
                    "agent": "gap",
                    "message": "No gaps found — coverage looks good",
                }
            yield {"type": "agent_done", "agent": "gap"}

            # 4. Quality
            yield {"type": "agent_start", "agent": "quality"}
            state = await self.quality_node(state)
            kept = state.get("quality_snippets", [])
            yield {"type": "quality_snippets", "count": len(kept)}
            yield {
                "type": "agent_message",
                "agent": "quality",
                "message": f"Kept {len(kept)} high-signal snippets",
            }
            yield {"type": "agent_done", "agent": "quality"}

            # 5. Writer (stream tokens if the model supports it)
            yield {"type": "agent_start", "agent": "writer"}
            chunk_queue: "asyncio.Queue[str]" = asyncio.Queue()

            async def _on_chunk(content: str) -> None:
                await chunk_queue.put(content)

            writer_task = asyncio.create_task(
                self._writer_stream(state, on_chunk=_on_chunk)
            )
            # Drain chunks as they arrive so the UI shows the document
            # being written in real time, instead of freezing until the
            # writer finishes.
            while True:
                if writer_task.done() and chunk_queue.empty():
                    break
                try:
                    chunk = await asyncio.wait_for(
                        chunk_queue.get(), timeout=0.1
                    )
                    yield {"type": "document_chunk", "content": chunk}
                except asyncio.TimeoutError:
                    continue
            document = await writer_task
            # Drain any stragglers.
            while not chunk_queue.empty():
                yield {
                    "type": "document_chunk",
                    "content": chunk_queue.get_nowait(),
                }
            yield {"type": "document_done", "document": document}
            yield {"type": "agent_done", "agent": "writer"}
            yield {"type": "complete", "document": document}

        except GenerationError as exc:
            yield {"type": "error", "message": exc.user_message}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Research pipeline failed: %s", exc, exc_info=True)
            yield {"type": "error", "message": f"Research failed: {exc}"}

    async def _writer_stream(
        self,
        state: ResearchState,
        on_chunk=None,
    ) -> str:
        """Invoke the writer prompt, streaming chunks if the model supports it.

        If `on_chunk` is provided, it is awaited with each text delta so the
        caller can emit `document_chunk` SSE events and the UI doesn't appear
        frozen while the writer is producing the (long) final document.
        """
        prompt = self._writer_prompt()
        prompt_vars = {
            "topic": state["topic"],
            "snippets": json.dumps(state.get("quality_snippets", []), ensure_ascii=False),
        }
        chain = prompt | self.llm
        try:
            if hasattr(chain, "astream"):
                buffer: List[str] = []
                async for chunk in chain.astream(prompt_vars):
                    content = getattr(chunk, "content", None)
                    if content is None:
                        content = str(chunk) if chunk else ""
                    if content:
                        buffer.append(content)
                        if on_chunk is not None:
                            await on_chunk(content)
                return "".join(buffer).strip()
            raw = await asyncio.wait_for(
                chain.ainvoke(prompt_vars), timeout=self.llm_timeout
            )
            text = getattr(raw, "content", str(raw)) or ""
            if text and on_chunk is not None:
                await on_chunk(text)
            return text.strip()
        except GenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise GenerationError(
                f"Writer LLM call timed out after {self.llm_timeout}s"
            ) from exc
        except Exception as exc:
            raise map_llm_error(exc) from exc


# ---------------------------------------------------------------------------
# Graph (built for completeness / future use; the streaming driver above
# walks the nodes manually so it can emit per-agent SSE events in order).
# ---------------------------------------------------------------------------
def build_research_graph(agent: ResearchAgent) -> Any:
    """Return a compiled LangGraph StateGraph for the research pipeline.

    This is exposed so callers can plug the agent into a larger LangGraph
    workflow. The HTTP layer uses `run_stream` directly for stable event
    ordering.
    """
    g = StateGraph(ResearchState)

    def _sync_wrap(coro_fn):
        async def fn(state):
            return await coro_fn(state)
        return fn

    g.add_node("planner", _sync_wrap(agent.planner_node))
    g.add_node("researcher", _sync_wrap(agent.researcher_node))
    g.add_node("gap", _sync_wrap(agent.gap_node))
    g.add_node("quality", _sync_wrap(agent.quality_node))
    g.add_node("writer", _sync_wrap(agent.writer_node))

    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "gap")
    g.add_edge("gap", "quality")
    g.add_edge("quality", "writer")
    g.add_edge("writer", END)
    return g.compile()