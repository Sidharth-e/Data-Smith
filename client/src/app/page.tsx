"use client";

import { useState, useCallback } from "react";

type FormatType = "alpaca" | "chat" | "completion";

interface GenerateResponse {
  success: boolean;
  format_type: string;
  data: Record<string, unknown>[];
  message: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [textInput, setTextInput] = useState("");
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [formatType, setFormatType] = useState<FormatType>("alpaca");
  const [numSamples, setNumSamples] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith(".txt")) {
        setFile(droppedFile);
        setInputMode("file");
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setInputMode("file");
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("format_type", formatType);
      formData.append("num_samples", numSamples.toString());

      let endpoint = "http://localhost:8000/api/generate";

      if (inputMode === "file" && file) {
        formData.append("file", file);
      } else if (inputMode === "text" && textInput.trim()) {
        formData.append("text", textInput);
        endpoint = "http://localhost:8000/api/generate-text";
      } else {
        throw new Error("Please provide input text or upload a file");
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data: GenerateResponse = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        format_type: formatType,
        data: [],
        message: error instanceof Error ? error.message : "An error occurred",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.data) return;

    const jsonStr = JSON.stringify(result.data, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dataset_${formatType}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatExamples: Record<FormatType, string> = {
    alpaca: `{
  "instruction": "...",
  "input": "...",
  "output": "..."
}`,
    chat: `{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}`,
    completion: `{
  "text": "..."
}`,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur-xl bg-black/20">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Data Smith</h1>
              <p className="text-xs text-white/60">
                Dataset Generator for Fine-Tuning
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Input */}
          <div className="space-y-6">
            {/* Input Mode Toggle */}
            <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">
                Input Source
              </h2>
              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => setInputMode("file")}
                  className={`flex-1 py-2.5 px-4 rounded-xl font-medium transition-all ${
                    inputMode === "file"
                      ? "bg-violet-500 text-white shadow-lg shadow-violet-500/25"
                      : "bg-white/5 text-white/60 hover:bg-white/10"
                  }`}
                >
                  📄 File Upload
                </button>
                <button
                  onClick={() => setInputMode("text")}
                  className={`flex-1 py-2.5 px-4 rounded-xl font-medium transition-all ${
                    inputMode === "text"
                      ? "bg-violet-500 text-white shadow-lg shadow-violet-500/25"
                      : "bg-white/5 text-white/60 hover:bg-white/10"
                  }`}
                >
                  ✏️ Text Input
                </button>
              </div>

              {inputMode === "file" ? (
                <div
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                    dragActive
                      ? "border-violet-500 bg-violet-500/10"
                      : file
                      ? "border-green-500/50 bg-green-500/5"
                      : "border-white/20 hover:border-white/40"
                  }`}
                >
                  <input
                    type="file"
                    accept=".txt"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  {file ? (
                    <div className="space-y-2">
                      <div className="w-12 h-12 mx-auto rounded-full bg-green-500/20 flex items-center justify-center">
                        <svg
                          className="w-6 h-6 text-green-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                      <p className="text-white font-medium">{file.name}</p>
                      <p className="text-white/40 text-sm">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="w-12 h-12 mx-auto rounded-full bg-white/10 flex items-center justify-center">
                        <svg
                          className="w-6 h-6 text-white/60"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                          />
                        </svg>
                      </div>
                      <p className="text-white/80">
                        Drop your .txt file here or click to browse
                      </p>
                      <p className="text-white/40 text-sm">
                        Only .txt files are supported
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Paste your text content here..."
                  className="w-full h-48 bg-black/30 border border-white/10 rounded-xl p-4 text-white placeholder-white/30 resize-none focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20"
                />
              )}
            </div>

            {/* Format Selection */}
            <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">
                Output Format
              </h2>
              <div className="grid grid-cols-3 gap-3">
                {(["alpaca", "chat", "completion"] as FormatType[]).map(
                  (format) => (
                    <button
                      key={format}
                      onClick={() => setFormatType(format)}
                      className={`py-3 px-4 rounded-xl font-medium text-sm transition-all ${
                        formatType === format
                          ? "bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white shadow-lg"
                          : "bg-white/5 text-white/60 hover:bg-white/10"
                      }`}
                    >
                      {format.charAt(0).toUpperCase() + format.slice(1)}
                    </button>
                  )
                )}
              </div>

              <div className="mt-4 p-4 bg-black/30 rounded-xl">
                <p className="text-white/40 text-xs mb-2">Format preview:</p>
                <pre className="text-white/80 text-xs overflow-x-auto">
                  {formatExamples[formatType]}
                </pre>
              </div>
            </div>

            {/* Settings & Generate */}
            <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6">
              <div className="flex items-center justify-between mb-4">
                <label className="text-white font-medium">
                  Number of Samples
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={numSamples}
                    onChange={(e) => setNumSamples(parseInt(e.target.value))}
                    className="w-24 accent-violet-500"
                  />
                  <span className="text-white bg-white/10 px-3 py-1 rounded-lg min-w-[3rem] text-center">
                    {numSamples}
                  </span>
                </div>
              </div>

              <button
                onClick={handleGenerate}
                disabled={
                  loading ||
                  (inputMode === "file" && !file) ||
                  (inputMode === "text" && !textInput.trim())
                }
                className="w-full py-4 rounded-xl font-semibold text-white bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Generating...
                  </span>
                ) : (
                  "🚀 Generate Dataset"
                )}
              </button>
            </div>
          </div>

          {/* Right Panel - Output */}
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 h-fit">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">Output</h2>
              {result?.success && result.data.length > 0 && (
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 py-2 px-4 rounded-lg bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-colors text-sm font-medium"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Download JSON
                </button>
              )}
            </div>

            {result ? (
              <div className="space-y-4">
                {/* Status Badge */}
                <div
                  className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                    result.success
                      ? "bg-green-500/20 text-green-400"
                      : "bg-red-500/20 text-red-400"
                  }`}
                >
                  {result.success ? "✓" : "✕"} {result.message}
                </div>

                {/* Data Preview */}
                {result.data.length > 0 && (
                  <div className="bg-black/40 rounded-xl p-4 max-h-[500px] overflow-auto">
                    <pre className="text-white/80 text-sm whitespace-pre-wrap break-words">
                      {JSON.stringify(result.data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-16">
                <div className="w-16 h-16 mx-auto rounded-full bg-white/5 flex items-center justify-center mb-4">
                  <svg
                    className="w-8 h-8 text-white/30"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <p className="text-white/40">
                  Upload a file and generate dataset to see results
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-12">
        <div className="max-w-6xl mx-auto px-6 py-4 text-center">
          <p className="text-white/40 text-sm">
            Powered by LangChain & Ollama • Built with Next.js
          </p>
        </div>
      </footer>
    </div>
  );
}
