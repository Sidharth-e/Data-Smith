export interface GenerateResponse {
  success: boolean;
  format_type: string;
  data: Record<string, unknown>[];
  message: string;
}

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const DEFAULT_TIMEOUT = 120000;

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit & { timeout?: number } = {}
): Promise<T> {
  const { timeout = DEFAULT_TIMEOUT, headers, ...customConfig } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      ...customConfig,
      headers: {
        Accept: "application/json",
        ...headers,
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const message =
        errorData.detail ||
        errorData.message ||
        `HTTP Error ${response.status}: ${response.statusText}`;
      throw new Error(message);
    }

    return (await response.json()) as T;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Request timed out");
    }
    throw error;
  }
}

export async function generateDataset(
  formData: FormData,
  isTextMode: boolean
): Promise<GenerateResponse> {
  const endpoint = isTextMode ? "/generate-text" : "/generate";
  return apiClient<GenerateResponse>(endpoint, {
    method: "POST",
    body: formData,
  });
}
