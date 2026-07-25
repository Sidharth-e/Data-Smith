import { apiClient } from "@/lib/axios";

export interface GenerateResponse {
  success: boolean;
  format_type: string;
  data: Record<string, unknown>[];
  message: string;
}

export interface HealthResponse {
  status: string;
  provider?: string;
  model?: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
}

export async function generateDataset(
  formData: FormData,
  isTextMode: boolean
): Promise<GenerateResponse> {
  const endpoint = isTextMode ? "/generate-text" : "/generate";
  const response = await apiClient.post<GenerateResponse>(endpoint, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

// ---- Research Streaming ----

export type ResearchEvent =
  | { type: "start"; topic: string }
  | { type: "agent_start"; agent: ResearchAgentName; detail?: string }
  | { type: "agent_done"; agent: ResearchAgentName }
  | { type: "agent_message"; agent: ResearchAgentName; message: string }
  | { type: "plan"; questions: string[] }
  | { type: "search"; query: string; results: ResearchSnippet[] }
  | { type: "snippets"; count: number; snippets?: ResearchSnippet[] }
  | { type: "gap_questions"; questions: string[] }
  | { type: "quality_snippets"; count: number }
  | { type: "document_chunk"; content: string }
  | { type: "document_done"; document: string }
  | { type: "complete"; document: string }
  | { type: "error"; message: string };

export type ResearchAgentName =
  | "planner"
  | "researcher"
  | "gap"
  | "quality"
  | "writer";

export interface ResearchSnippet {
  title: string;
  snippet: string;
  url: string;
}

/**
 * Stream a multi-agent research run as Server-Sent Events.
 *
 * Calls `onEvent` for every parsed event. Resolves with the final
 * synthesized document on `complete`, or rejects on `error` / network
 * failure.
 */
export async function researchTopicStream(
  topic: string,
  onEvent: (event: ResearchEvent) => void,
  signal?: AbortSignal
): Promise<string> {
  const baseURL =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
  const url = `${baseURL}/research-stream`;

  const formData = new FormData();
  formData.append("topic", topic);

  const resp = await fetch(url, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!resp.ok || !resp.body) {
    let detail = `HTTP ${resp.status}`;
    try {
      const err = await resp.json();
      detail = err.detail || err.message || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let document = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      const lines = rawEvent.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload) continue;
        try {
          const event = JSON.parse(payload) as ResearchEvent;
          if (event.type === "complete" || event.type === "document_done") {
            document = event.document;
          }
          onEvent(event);
        } catch {
          // Ignore malformed chunks.
        }
      }
    }
  }

  return document;
}

// ---- Dataset Streaming ----

export type StreamEvent =
  | { type: "start"; format_type: string; num_samples: number; num_batches: number; batch_size: number }
  | { type: "batch_start"; index: number; total: number; batch_size: number }
  | { type: "thinking"; content: string; index: number }
  | { type: "token"; content: string; index: number }
  | { type: "batch_done"; index: number; samples: Record<string, unknown>[]; count: number }
  | { type: "progress"; done: number; total: number; samples_so_far: number }
  | { type: "warning"; message: string; index: number }
  | { type: "complete"; data: Record<string, unknown>[]; count: number }
  | { type: "error"; message: string };

/**
 * Stream a dataset generation request as Server-Sent Events.
 *
 * Calls `onEvent` for every parsed event. Returns when the stream
 * closes (either via "complete"/"error" or network end). Throws on a
 * network failure that produces no SSE error event.
 */
export async function generateDatasetStream(
  formData: FormData,
  isTextMode: boolean,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const endpoint = isTextMode ? "/generate-text-stream" : "/generate-stream";
  const baseURL =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
  const url = `${baseURL}${endpoint}`;

  const resp = await fetch(url, {
    method: "POST",
    body: formData,
    signal,
    // Don't set Content-Type; the browser will set multipart boundary itself.
  });

  if (!resp.ok || !resp.body) {
    let detail = `HTTP ${resp.status}`;
    try {
      const err = await resp.json();
      detail = err.detail || err.message || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line; each "data: ..." may span
    // multiple lines but our server emits one JSON object per event.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      const lines = rawEvent.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload) continue;
        try {
          const event = JSON.parse(payload) as StreamEvent;
          onEvent(event);
        } catch {
          // Ignore malformed chunks (keep the stream alive).
        }
      }
    }
  }
}
