"use client";

import { useCallback, useRef, useState } from "react";
import {
  researchTopicStream,
  ResearchEvent,
  ResearchAgentName,
  ResearchSnippet,
} from "@/app/api";

export interface ResearchAgentState {
  name: ResearchAgentName;
  status: "idle" | "running" | "done";
  message?: string;
}

export interface ResearchState {
  status: "idle" | "streaming" | "done" | "error";
  topic: string;
  plan: string[];
  snippets: ResearchSnippet[];
  gapQuestions: string[];
  qualityCount: number;
  document: string;
  agents: ResearchAgentState[];
  logs: { id: number; text: string; ts: number }[];
  errorMessage: string | null;
}

const AGENT_ORDER: ResearchAgentName[] = [
  "planner",
  "researcher",
  "gap",
  "quality",
  "writer",
];

const initialState: ResearchState = {
  status: "idle",
  topic: "",
  plan: [],
  snippets: [],
  gapQuestions: [],
  qualityCount: 0,
  document: "",
  agents: AGENT_ORDER.map((name) => ({ name, status: "idle" })),
  logs: [],
  errorMessage: null,
};

export function useResearchStream() {
  const [state, setState] = useState<ResearchState>(initialState);
  const logIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    logIdRef.current = 0;
    setState(initialState);
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState((s) => ({ ...s, status: "idle" }));
  }, []);

  const pushLog = useCallback((text: string) => {
    setState((s) => ({
      ...s,
      logs: [
        ...s.logs,
        { id: logIdRef.current++, text, ts: Date.now() },
      ].slice(-200),
    }));
  }, []);

  const mutateAsync = useCallback(
    async (topic: string) => {
      if (!topic.trim()) {
        throw new Error("Please enter a topic to research");
      }

      // Fresh state for this run.
      logIdRef.current = 0;
      setState({
        ...initialState,
        status: "streaming",
        topic: topic.trim(),
      });

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await researchTopicStream(
          topic,
          (event: ResearchEvent) => {
            setState((s) => {
              switch (event.type) {
                case "start":
                  return { ...s, topic: event.topic };
                case "agent_start":
                  return {
                    ...s,
                    agents: s.agents.map((a) =>
                      a.name === event.agent
                        ? { ...a, status: "running", message: event.detail }
                        : a
                    ),
                  };
                case "agent_done":
                  return {
                    ...s,
                    agents: s.agents.map((a) =>
                      a.name === event.agent
                        ? { ...a, status: "done" }
                        : a
                    ),
                  };
                case "agent_message":
                  return {
                    ...s,
                    agents: s.agents.map((a) =>
                      a.name === event.agent
                        ? { ...a, message: event.message }
                        : a
                    ),
                  };
                case "plan":
                  return { ...s, plan: event.questions };
                case "search":
                  return {
                    ...s,
                    snippets: [
                      ...s.snippets,
                      ...event.results.filter(
                        (r) =>
                          !s.snippets.some(
                            (ex) => ex.url === r.url && ex.title === r.title
                          )
                      ),
                    ],
                  };
                case "snippets":
                  return {
                    ...s,
                    snippets: event.snippets
                      ? event.snippets
                      : s.snippets.slice(0, event.count),
                  };
                case "gap_questions":
                  return { ...s, gapQuestions: event.questions };
                case "quality_snippets":
                  return { ...s, qualityCount: event.count };
                case "document_chunk":
                  return {
                    ...s,
                    document: s.document + event.content,
                  };
                case "document_done":
                  return { ...s, document: event.document };
                case "complete":
                  return {
                    ...s,
                    status: "done",
                    document: event.document,
                  };
                case "error":
                  return {
                    ...s,
                    status: "error",
                    errorMessage: event.message,
                  };
                default:
                  return s;
              }
            });

            // Side-effect logs.
            switch (event.type) {
              case "start":
                pushLog(`Research started: ${event.topic}`);
                break;
              case "agent_start":
                pushLog(`▶ ${event.agent} agent started`);
                break;
              case "agent_done":
                pushLog(`✓ ${event.agent} agent done`);
                break;
              case "agent_message":
                pushLog(`${event.agent}: ${event.message}`);
                break;
              case "plan":
                pushLog(`Planner produced ${event.questions.length} questions`);
                break;
              case "search":
                pushLog(`Searching: "${event.query}"`);
                break;
              case "snippets":
                pushLog(`${event.count} snippets collected so far`);
                break;
              case "gap_questions":
                pushLog(
                  event.questions.length
                    ? `Gap found ${event.questions.length} follow-ups`
                    : "Gap: no follow-ups needed"
                );
                break;
              case "quality_snippets":
                pushLog(`Quality kept ${event.count} snippets`);
                break;
              case "document_done":
                pushLog(`Writer finished (${event.document.length} chars)`);
                break;
              case "error":
                pushLog(`Error: ${event.message}`);
                break;
            }
          },
          controller.signal
        );
      } catch (err) {
        setState((s) => ({
          ...s,
          status: "error",
          errorMessage:
            err instanceof Error ? err.message : "Research failed",
        }));
        throw err;
      }
    },
    [pushLog]
  );

  return {
    state,
    mutateAsync,
    reset,
    abort,
    isPending: state.status === "streaming",
    isError: state.status === "error",
    isDone: state.status === "done",
  };
}