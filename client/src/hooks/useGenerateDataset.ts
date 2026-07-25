import { useMutation, useQuery } from "@tanstack/react-query";
import { generateDataset, fetchHealth, GenerateResponse, HealthResponse } from "@/app/api";

import { FormatType } from "@/store/useWorkbenchStore";

export interface GenerateParams {
  file: File | null;
  textInput: string;
  inputMode: "file" | "text";
  formatType: FormatType;
  numSamples: number;
}

export function useGenerateDataset() {
  return useMutation<GenerateResponse, Error, GenerateParams>({
    mutationFn: async ({ file, textInput, inputMode, formatType, numSamples }) => {
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

      return generateDataset(formData, isTextMode);
    },
  });
}

export function useApiHealth() {
  return useQuery<HealthResponse, Error>({
    queryKey: ["apiHealth"],
    queryFn: fetchHealth,
    refetchInterval: 30000,
    retry: 1,
  });
}
