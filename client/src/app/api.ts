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

// ---- Streaming ----

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
