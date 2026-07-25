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
