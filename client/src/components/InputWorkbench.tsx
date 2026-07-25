"use client";

import { useState, useCallback, DragEvent, ChangeEvent } from "react";
import { useWorkbenchStore, FormatType } from "@/store/useWorkbenchStore";
import { useGenerateDatasetStream } from "@/hooks/useGenerateDatasetStream";
import { FileText, Type, Upload, Check, Sparkles, Sliders, Layers, AlertCircle, Loader2, X, Code2, ChevronDown, ChevronUp } from "lucide-react";

const formatExamples: Record<FormatType, string> = {
  alpaca: `{\n  "instruction": "...",\n  "input": "...",\n  "output": "..."\n}`,
  chat: `{\n  "messages": [\n    {"role": "system", "content": "..."},\n    {"role": "user", "content": "..."},\n    {"role": "assistant", "content": "..."}\n  ]\n}`,
  completion: `{\n  "text": "..."\n}`,
};

interface InputWorkbenchProps {
  onGenerateStart: () => void;
  stream: ReturnType<typeof useGenerateDatasetStream>;
}

export default function InputWorkbench({ onGenerateStart, stream }: InputWorkbenchProps) {
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
  const [showSchemaPreview, setShowSchemaPreview] = useState(false);

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
    onGenerateStart();
    try {
      await stream.mutateAsync({
        file,
        textInput,
        inputMode,
        formatType,
        numSamples,
      });
    } catch {
      // Handled via stream.state
    }
  };

  const isFormValid =
    (inputMode === "file" && file !== null) ||
    (inputMode === "text" && textInput.trim().length > 0);

  return (
    <div className="bg-card border border-border rounded-2xl p-6 shadow-xs flex flex-col justify-between min-h-[640px] space-y-6">
      <div className="space-y-6">
        <div className="flex items-center justify-between border-b border-border pb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary-light text-primary flex items-center justify-center font-bold">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-foreground tracking-tight">Synthesis Workbench</h2>
              <p className="text-xs text-secondary font-medium">Configure source input & model parameters</p>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-xs font-extrabold uppercase tracking-wider text-secondary flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-primary" />
              <span>1. Source Content</span>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-1 bg-muted p-1 rounded-xl border border-border">
            <button
              type="button"
              onClick={() => setInputMode("file")}
              className={`h-9 px-3 rounded-lg text-xs font-extrabold transition-all flex items-center justify-center gap-2 cursor-pointer ${inputMode === "file"
                ? "bg-card text-foreground shadow-2xs border border-border"
                : "text-secondary hover:text-foreground"
                }`}
            >
              <Upload className="w-3.5 h-3.5 text-primary" />
              <span className="whitespace-nowrap">File Upload</span>
            </button>
            <button
              type="button"
              onClick={() => setInputMode("text")}
              className={`h-9 px-3 rounded-lg text-xs font-extrabold transition-all flex items-center justify-center gap-2 cursor-pointer ${inputMode === "text"
                ? "bg-card text-foreground shadow-2xs border border-border"
                : "text-secondary hover:text-foreground"
                }`}
            >
              <Type className="w-3.5 h-3.5 text-primary" />
              <span className="whitespace-nowrap">Text Input</span>
            </button>
          </div>

          {inputMode === "file" ? (
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${dragActive
                ? "border-primary bg-primary-light"
                : file
                  ? "border-success bg-muted/40"
                  : "border-border bg-muted/20 hover:border-primary/50"
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
                <div className="flex items-center justify-between bg-card p-3.5 rounded-xl border border-border shadow-2xs">
                  <div className="flex items-center gap-3 text-left overflow-hidden">
                    <div className="w-9 h-9 rounded-lg bg-success/10 text-success flex items-center justify-center shrink-0">
                      <Check className="w-5 h-5" />
                    </div>
                    <div className="truncate">
                      <p className="text-foreground font-extrabold text-xs truncate">{file.name}</p>
                      <p className="text-secondary font-semibold text-[11px]">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                    }}
                    className="text-xs text-error hover:underline font-bold shrink-0 ml-2 cursor-pointer"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div className="space-y-2 py-4">
                  <div className="w-11 h-11 mx-auto rounded-xl bg-card border border-border flex items-center justify-center text-primary shadow-2xs">
                    <Upload className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-foreground font-extrabold text-xs">
                      Drop your .txt file here or click to browse
                    </p>
                    <p className="text-secondary font-semibold text-[11px] mt-0.5">
                      Supports plain text documents
                    </p>
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
              className="w-full h-40 bg-card border border-border rounded-xl p-4 text-foreground text-xs font-medium placeholder-secondary resize-none focus:outline-none focus:border-primary transition-colors"
            />
          )}
        </div>

        <div className="space-y-3 pt-3 border-t border-border">
          <div className="flex items-center justify-between">
            <label className="text-xs font-extrabold uppercase tracking-wider text-secondary flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-primary" />
              <span>2. Target Dataset Format</span>
            </label>
            <button
              type="button"
              onClick={() => setShowSchemaPreview(!showSchemaPreview)}
              className="text-[11px] text-primary hover:underline font-extrabold flex items-center gap-1 cursor-pointer"
            >
              <Code2 className="w-3 h-3" />
              <span>Schema Preview</span>
              {showSchemaPreview ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          </div>

          <div className="grid grid-cols-3 gap-2.5">
            {(["alpaca", "chat", "completion"] as FormatType[]).map((format) => (
              <button
                key={format}
                type="button"
                onClick={() => setFormatType(format)}
                className={`h-10 px-3 rounded-xl text-xs font-extrabold transition-all border cursor-pointer ${formatType === format
                  ? "bg-primary text-white border-primary shadow-2xs"
                  : "bg-card border-border text-secondary hover:border-primary/50 hover:text-foreground"
                  }`}
              >
                <span className="whitespace-nowrap">{format.charAt(0).toUpperCase() + format.slice(1)}</span>
              </button>
            ))}
          </div>

          {showSchemaPreview && (
            <div className="bg-muted border border-border rounded-xl p-3.5 space-y-1 animate-in fade-in duration-150">
              <span className="text-secondary text-[10px] font-extrabold uppercase tracking-wider block">
                {formatType.toUpperCase()} JSON Schema
              </span>
              <pre className="text-foreground text-xs font-mono font-semibold overflow-x-auto whitespace-pre">
                {formatExamples[formatType]}
              </pre>
            </div>
          )}
        </div>

        <div className="space-y-3 pt-3 border-t border-border">
          <div className="flex items-center justify-between">
            <label htmlFor="num-samples-input" className="text-xs font-extrabold uppercase tracking-wider text-secondary flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-primary" />
              <span>3. Synthesis Parameters</span>
            </label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-secondary font-bold">Samples:</span>
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
                className="w-16 h-8 bg-card border border-border rounded-lg text-center text-xs font-extrabold text-foreground focus:outline-none focus:border-primary"
              />
            </div>
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

          <div className="grid grid-cols-5 gap-2">
            {[5, 20, 50, 100, 500].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setNumSamples(n)}
                className={`h-8 rounded-lg text-xs font-extrabold transition-all flex items-center justify-center cursor-pointer ${numSamples === n
                  ? "bg-primary text-white shadow-2xs"
                  : "bg-card text-secondary border border-border hover:border-primary/50 hover:text-foreground"
                  }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {stream.isError && (
          <div className="flex items-center gap-2 p-3 bg-error/10 border border-error/30 rounded-xl text-error text-xs font-bold">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{stream.error?.message || "Failed to generate dataset"}</span>
          </div>
        )}
      </div>

      <div className="flex gap-2 pt-2 border-t border-border">
        <button
          type="button"
          onClick={handleGenerate}
          disabled={stream.isPending || !isFormValid}
          className="flex-1 h-12 bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-extrabold text-sm rounded-xl transition-all shadow-xs active:scale-[0.99] flex items-center justify-center gap-2 cursor-pointer"
        >
          {stream.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Streaming Dataset...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Generate Dataset</span>
            </>
          )}
        </button>
        {stream.isPending && (
          <button
            type="button"
            onClick={stream.abort}
            className="h-12 px-5 bg-muted hover:bg-border text-foreground font-extrabold text-xs rounded-xl transition-all border border-border flex items-center gap-1.5 cursor-pointer"
          >
            <X className="w-4 h-4 text-error" />
            <span>Stop</span>
          </button>
        )}
      </div>
    </div>
  );
}
