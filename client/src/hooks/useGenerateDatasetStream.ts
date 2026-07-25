"use client";

import { useCallback, useRef, useState } from "react";
import {
  generateDatasetStream,
  StreamEvent,
} from "@/app/api";

export interface BatchState {
  index: number;
  thinking: string;
  content: string;
  done: boolean;
  samples: Record<string, unknown>[];
}

export interface StreamLogEntry {
  id: number;
  kind: "info" | "thinking" | "token" | "batch" | "warning" | "error";
  message: string;
  ts: number;
}

export interface StreamState {
  status: "idle" | "streaming" | "done" | "error";
  formatType: string | null;
  numSamples: number;
  numBatches: number;
  batchSize: number;
  batches: BatchState[];
  samples: Record<string, unknown>[];
  progressDone: number;
  progressTotal: number;
  logs: StreamLogEntry[];
  errorMessage: string | null;
}

const initialState: StreamState = {
  status: "idle",
  formatType: null,
  numSamples: 0,
  numBatches: 0,
  batchSize: 0,
  batches: [],
  samples: [],
  progressDone: 0,
  progressTotal: 0,
  logs: [],
  errorMessage: null,
};

export interface GenerateStreamParams {
  file: File | null;
  textInput: string;
  inputMode: "file" | "text";
  formatType: "alpaca" | "chat" | "completion";
  numSamples: number;
}

export function useGenerateDatasetStream() {
  const [state, setState] = useState<StreamState>(initialState);
  const logIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const pushLog = useCallback(
    (kind: StreamLogEntry["kind"], message: string) => {
      setState((s) => ({
        ...s,
        logs: [
          ...s.logs,
          { id: logIdRef.current++, kind, message, ts: Date.now() },
        ].slice(-200), // cap log size to keep the UI snappy
      }));
    },
    []
  );

  const reset = useCallback(() => {
    logIdRef.current = 0;
    setState(initialState);
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState((s) => ({ ...s, status: "idle" }));
  }, []);

  const mutateAsync = useCallback(
    async (params: GenerateStreamParams) => {
      const { file, textInput, inputMode, formatType, numSamples } = params;
      const formData = new FormData();
      formData.append("format_type", formatType);
      formData.append("num_samples", numSamples.toString());
      const isTextMode = inputMode === "text";

      if (inputMode === "file" && file) {
        formData.append("file", file);
      } else if (inputMode === "text" && textInput.trim()) {
        formData.append("text", textInput);
      } else {
        throw new Error("Please provide input text or upload a .txt file");
      }

      // Fresh state for this run.
      logIdRef.current = 0;
      setState({
        ...initialState,
        status: "streaming",
        formatType,
        numSamples,
      });

      const controller = new AbortController();
      abortRef.current = controller;

      await generateDatasetStream(
        formData,
        isTextMode,
        (event: StreamEvent) => {
          setState((s) => {
            switch (event.type) {
              case "start":
                return {
                  ...s,
                  formatType: event.format_type,
                  numSamples: event.num_samples,
                  numBatches: event.num_batches,
                  batchSize: event.batch_size,
                };
              case "batch_start": {
                const batches = [...s.batches];
                if (!batches[event.index]) {
                  batches[event.index] = {
                    index: event.index,
                    thinking: "",
                    content: "",
                    done: false,
                    samples: [],
                  };
                }
                return { ...s, batches };
              }
              case "thinking": {
                const batches = [...s.batches];
                const b = batches[event.index] || {
                  index: event.index,
                  thinking: "",
                  content: "",
                  done: false,
                  samples: [],
                };
                batches[event.index] = {
                  ...b,
                  thinking: b.thinking + event.content,
                };
                return { ...s, batches };
              }
              case "token": {
                const batches = [...s.batches];
                const b = batches[event.index] || {
                  index: event.index,
                  thinking: "",
                  content: "",
                  done: false,
                  samples: [],
                };
                batches[event.index] = {
                  ...b,
                  content: b.content + event.content,
                };
                return { ...s, batches };
              }
              case "batch_done": {
                const batches = [...s.batches];
                const b = batches[event.index] || {
                  index: event.index,
                  thinking: "",
                  content: "",
                  done: false,
                  samples: [],
                };
                batches[event.index] = {
                  ...b,
                  done: true,
                  samples: event.samples,
                };
                return {
                  ...s,
                  batches,
                  samples: [...s.samples, ...event.samples],
                };
              }
              case "progress":
                return {
                  ...s,
                  progressDone: event.done,
                  progressTotal: event.total,
                };
              case "warning":
                return s;
              case "complete":
                return {
                  ...s,
                  status: "done",
                  samples: event.data,
                  progressDone: s.progressTotal || s.batches.length,
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

          // Side-effect logs (kept out of the reducer to avoid stale reads).
          switch (event.type) {
            case "start":
              pushLog(
                "info",
                `Started: ${event.format_type} · ${event.num_samples} samples · ${event.num_batches} batches`
              );
              break;
            case "batch_start":
              pushLog("info", `Batch ${event.index + 1}/${event.total} started`);
              break;
            case "batch_done":
              pushLog(
                "batch",
                `Batch ${event.index + 1} done — ${event.count} new samples`
              );
              break;
            case "progress":
              pushLog(
                "info",
                `Progress ${event.done}/${event.total} · ${event.samples_so_far} samples`
              );
              break;
            case "warning":
              pushLog("warning", `Batch ${event.index + 1}: ${event.message}`);
              break;
            case "error":
              pushLog("error", event.message);
              break;
            case "complete":
              pushLog("info", `Complete — ${event.count} samples`);
              break;
          }
        },
        controller.signal
      );
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
    error: state.errorMessage ? new Error(state.errorMessage) : null,
  };
}