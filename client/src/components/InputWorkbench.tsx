"use client";

import { useState, useCallback, DragEvent, ChangeEvent } from "react";
import { useWorkbenchStore, FormatType } from "@/store/useWorkbenchStore";
import { useGenerateDataset } from "@/hooks/useGenerateDataset";
import { GenerateResponse } from "@/app/api";
import { FileText, Type, Upload, Check, Sparkles, Sliders, Layers, AlertCircle, Loader2 } from "lucide-react";

const formatExamples: Record<FormatType, string> = {
  alpaca: `{\n  "instruction": "...",\n  "input": "...",\n  "output": "..."\n}`,
  chat: `{\n  "messages": [\n    {"role": "system", "content": "..."},\n    {"role": "user", "content": "..."},\n    {"role": "assistant", "content": "..."}\n  ]\n}`,
  completion: `{\n  "text": "..."\n}`,
};

interface InputWorkbenchProps {
  onGenerateSuccess: (data: GenerateResponse) => void;
  onGenerateError: (errorMsg: string) => void;
}

export default function InputWorkbench({ onGenerateSuccess, onGenerateError }: InputWorkbenchProps) {
  const {
    inputMode,
    setInputMode,
    formatType,
    setFormatType,
    numSamples,
    setNumSamples,
    file,
    setFile,
    textInput,
    setTextInput,
  } = useWorkbenchStore();

  const [dragActive, setDragActive] = useState(false);
  const generateMutation = useGenerateDataset();

  const handleDrag = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile.name.endsWith(".txt")) {
          setFile(droppedFile);
        }
      }
    },
    [setFile]
  );

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleGenerate = async () => {
    try {
      const res = await generateMutation.mutateAsync({
        file,
        textInput,
        inputMode,
        formatType,
        numSamples,
      });
      onGenerateSuccess(res);
    } catch (err) {
      onGenerateError(err instanceof Error ? err.message : "Dataset generation failed");
    }
  };

  const isFormValid =
    (inputMode === "file" && file !== null) ||
    (inputMode === "text" && textInput.trim().length > 0);

  return (
    <div className="space-y-6">
      <div className="bg-card border border-border rounded-2xl p-l space-y-4 shadow-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">
              1
            </span>
            <h3 className="text-base font-extrabold text-foreground">Source Content</h3>
          </div>
          <span className="text-xs text-secondary font-bold">Step 1 of 3</span>
        </div>

        <div className="flex gap-2 bg-muted p-s rounded-xl border border-border">
          <button
            type="button"
            onClick={() => setInputMode("file")}
            className={`flex-1 h-medium px-m rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${
              inputMode === "file"
                ? "bg-card text-foreground shadow-2xs border border-border"
                : "text-secondary hover:text-foreground hover:bg-card/50"
            }`}
          >
            <FileText className="w-4 h-4 text-primary" />
            File Upload
          </button>
          <button
            type="button"
            onClick={() => setInputMode("text")}
            className={`flex-1 h-medium px-m rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${
              inputMode === "text"
                ? "bg-card text-foreground shadow-2xs border border-border"
                : "text-secondary hover:text-foreground hover:bg-card/50"
            }`}
          >
            <Type className="w-4 h-4 text-primary" />
            Text Editor
          </button>
        </div>

        {inputMode === "file" ? (
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-xl text-center transition-all cursor-pointer ${
              dragActive
                ? "border-primary bg-primary-light"
                : file
                ? "border-success bg-muted/50"
                : "border-border bg-muted/30 hover:border-primary/50"
            }`}
          >
            <input
              type="file"
              accept=".txt"
              onChange={handleFileChange}
              aria-label="Upload .txt file"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            {file ? (
              <div className="space-y-2">
                <div className="w-12 h-12 mx-auto rounded-xl bg-card border border-border flex items-center justify-center text-success shadow-2xs">
                  <Check className="w-6 h-6" />
                </div>
                <p className="text-foreground font-extrabold text-sm">{file.name}</p>
                <p className="text-secondary font-bold text-xs">{(file.size / 1024).toFixed(1)} KB</p>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                  }}
                  className="text-xs text-error hover:underline pt-1 font-bold"
                >
                  Remove file
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-12 h-12 mx-auto rounded-xl bg-card border border-border flex items-center justify-center text-primary shadow-2xs">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-foreground font-extrabold text-sm">
                    Drop your .txt file here or click to browse
                  </p>
                  <p className="text-secondary font-bold text-xs mt-1">Only .txt files are supported</p>
                </div>
              </div>
            )}
          </div>
        ) : (
          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Paste raw text, articles, or documentation here..."
            aria-label="Raw text content input"
            className="w-full h-44 bg-card border border-border rounded-xl p-m text-foreground text-sm font-medium placeholder-secondary resize-none focus:outline-none focus:border-primary transition-colors"
          />
        )}
      </div>

      <div className="bg-card border border-border rounded-2xl p-l space-y-4 shadow-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">
              2
            </span>
            <h3 className="text-base font-extrabold text-foreground">Target Dataset Format</h3>
          </div>
          <Layers className="w-4 h-4 text-secondary" />
        </div>

        <div className="grid grid-cols-3 gap-3">
          {(["alpaca", "chat", "completion"] as FormatType[]).map((format) => (
            <button
              key={format}
              type="button"
              onClick={() => setFormatType(format)}
              className={`h-medium px-m rounded-xl text-sm font-bold transition-all ${
                formatType === format
                  ? "bg-primary text-white border-2 border-primary shadow-2xs"
                  : "bg-card border border-border text-secondary hover:border-primary/50 hover:text-foreground"
              }`}
            >
              {format.charAt(0).toUpperCase() + format.slice(1)}
            </button>
          ))}
        </div>

        <div className="bg-muted border border-border rounded-xl p-m space-y-2">
          <span className="text-secondary text-xs font-extrabold uppercase tracking-wider block">
            Schema Structure Preview
          </span>
          <pre className="text-foreground text-xs font-mono font-bold overflow-x-auto whitespace-pre">
            {formatExamples[formatType]}
          </pre>
        </div>
      </div>

      <div className="bg-card border border-border rounded-2xl p-l space-y-5 shadow-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">
              3
            </span>
            <h3 className="text-base font-extrabold text-foreground">Synthesis Parameters</h3>
          </div>
          <Sliders className="w-4 h-4 text-secondary" />
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label htmlFor="num-samples-input" className="text-sm font-extrabold text-foreground">
              Number of Samples
            </label>
            <input
              id="num-samples-input"
              type="number"
              min="1"
              max="1000"
              value={numSamples}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setNumSamples(isNaN(val) ? 1 : val);
              }}
              className="w-20 h-9 bg-card border-2 border-border rounded-lg text-center text-sm font-extrabold text-foreground focus:outline-none focus:border-primary"
            />
          </div>

          <input
            id="num-samples-slider"
            type="range"
            min="1"
            max="100"
            value={numSamples > 100 ? 100 : numSamples}
            onChange={(e) => setNumSamples(parseInt(e.target.value))}
            aria-label="Number of Samples Slider"
            className="w-full accent-primary bg-muted h-2 rounded-lg cursor-pointer"
          />

          <div className="flex flex-wrap gap-2">
            {[5, 20, 50, 100, 500, 1000].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setNumSamples(n)}
                className={`px-m py-s rounded-lg text-xs font-bold transition-all ${
                  numSamples === n
                    ? "bg-primary text-white shadow-2xs"
                    : "bg-card text-secondary border border-border hover:border-primary/50 hover:text-foreground"
                }`}
              >
                {n} samples
              </button>
            ))}
          </div>
        </div>

        {generateMutation.isError && (
          <div className="flex items-center gap-2 p-m bg-muted border border-border rounded-xl text-error text-xs font-bold">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{generateMutation.error?.message || "Failed to generate dataset"}</span>
          </div>
        )}

        <button
          type="button"
          onClick={handleGenerate}
          disabled={generateMutation.isPending || !isFormValid}
          className="w-full h-high bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-extrabold text-base rounded-xl transition-all shadow-md active:scale-[0.99] flex items-center justify-center gap-2"
        >
          {generateMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Synthesizing Synthetic Dataset...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Generate Dataset</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
