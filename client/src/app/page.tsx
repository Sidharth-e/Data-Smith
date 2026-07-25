"use client";

import { useState, useCallback } from "react";
import { generateDataset, GenerateResponse } from "./api";

type FormatType = "alpaca" | "chat" | "completion";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [textInput, setTextInput] = useState("");
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [formatType, setFormatType] = useState<FormatType>("alpaca");
  const [numSamples, setNumSamples] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [copied, setCopied] = useState(false);

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

      const isTextMode = inputMode === "text";

      if (inputMode === "file" && file) {
        formData.append("file", file);
      } else if (inputMode === "text" && textInput.trim()) {
        formData.append("text", textInput);
      } else {
        throw new Error("Please provide input text or upload a file");
      }

      const data = await generateDataset(formData, isTextMode);
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

  const handleCopy = () => {
    if (!result?.data) return;
    navigator.clipboard.writeText(JSON.stringify(result.data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatExamples: Record<FormatType, string> = {
    alpaca: `{\n  "instruction": "...",\n  "input": "...",\n  "output": "..."\n}`,
    chat: `{\n  "messages": [\n    {"role": "system", "content": "..."},\n    {"role": "user", "content": "..."},\n    {"role": "assistant", "content": "..."}\n  ]\n}`,
    completion: `{\n  "text": "..."\n}`,
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="max-w-7xl mx-auto px-l py-m">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary text-white flex items-center justify-center shadow-xs">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold text-foreground tracking-tight">Data Smith</h1>
              <p className="text-xs text-muted">Generate datasets for LLM fine-tuning</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-l py-l space-y-8 flex-1 w-full">
        {/* Hero Section */}
        <section className="bg-primary-light border border-primary/20 rounded-2xl p-l md:p-xl shadow-xs">
          <div className="grid md:grid-cols-3 gap-6 items-center">
            <div className="md:col-span-2 space-y-4">
              <div className="inline-flex items-center gap-1.5 bg-card text-primary text-xs font-bold px-m py-s rounded-full border border-primary/20 uppercase tracking-wider shadow-2xs">
                <span>✨</span>
                <span>Fine-tuning ready in minutes</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-extrabold text-foreground tracking-tight leading-tight">
                Create fine-tuning datasets in minutes
              </h2>
              <p className="text-muted text-sm md:text-base leading-relaxed max-w-2xl">
                Upload a text file or paste raw text, pick a training format, and let Data Smith generate ready-to-use JSON samples for your next model.
              </p>
            </div>

            {/* Stat Card */}
            <div className="bg-card border border-border rounded-2xl p-l shadow-sm space-y-4">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl md:text-4xl font-extrabold text-foreground">73%</span>
                <span className="text-xs text-muted font-medium">format match</span>
              </div>
              <div className="w-full bg-muted h-2 rounded-full overflow-hidden">
                <div className="bg-primary h-full rounded-full w-[73%] transition-all duration-500" />
              </div>
              <div className="flex items-center gap-2 pt-1">
                <span className="w-4 h-4 rounded-full bg-muted border border-border" />
                <span className="w-4 h-4 rounded-full bg-primary" />
                <span className="w-4 h-4 rounded-full bg-muted border border-border" />
                <span className="w-4 h-4 rounded-full bg-primary/70" />
              </div>
            </div>
          </div>
        </section>

        {/* Main 2-Column Interface */}
        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Left Column: Controls & Input */}
          <div className="space-y-6">
            {/* Input Mode Container */}
            <div className="bg-card border border-border rounded-2xl p-l space-y-4 shadow-xs">
              <div className="flex gap-2 bg-muted p-s rounded-xl border border-border">
                <button
                  type="button"
                  onClick={() => setInputMode("file")}
                  className={`flex-1 h-medium px-m rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
                    inputMode === "file"
                      ? "bg-card text-foreground shadow-2xs border border-border"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  File Upload
                </button>
                <button
                  type="button"
                  onClick={() => setInputMode("text")}
                  className={`flex-1 h-medium px-m rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
                    inputMode === "text"
                      ? "bg-card text-foreground shadow-2xs border border-border"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                  Text Input
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
                      ? "border-success/50 bg-muted/50"
                      : "border-border bg-muted/30 hover:border-primary/40"
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
                      <div className="w-12 h-12 mx-auto rounded-xl bg-card border border-border flex items-center justify-center text-success shadow-2xs">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                      <p className="text-foreground font-semibold text-sm">{file.name}</p>
                      <p className="text-muted text-xs">{(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="w-12 h-12 mx-auto rounded-xl bg-card border border-border flex items-center justify-center text-muted shadow-2xs">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0l-4 4m4-4v12" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-foreground font-semibold text-sm">Drop your .txt file here or click to browse</p>
                        <p className="text-muted text-xs mt-1">Only .txt files are supported</p>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Paste your raw text content here..."
                  className="w-full h-44 bg-muted border border-border rounded-xl p-m text-foreground text-sm placeholder-muted resize-none focus:outline-none focus:border-primary transition-colors"
                />
              )}
            </div>

            {/* Step 2: Output Format */}
            <div className="bg-card border border-border rounded-2xl p-l space-y-4 shadow-xs">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">2</span>
                <h3 className="text-base font-bold text-foreground">Output Format</h3>
              </div>

              <div className="grid grid-cols-3 gap-3">
                {(["alpaca", "chat", "completion"] as FormatType[]).map((format) => (
                  <button
                    key={format}
                    type="button"
                    onClick={() => setFormatType(format)}
                    className={`h-medium px-m rounded-xl text-sm font-semibold transition-all ${
                      formatType === format
                        ? "bg-card border-2 border-primary text-foreground shadow-2xs"
                        : "bg-card border border-border text-muted hover:border-border/80 hover:text-foreground"
                    }`}
                  >
                    {format.charAt(0).toUpperCase() + format.slice(1)}
                  </button>
                ))}
              </div>

              <div className="bg-muted border border-border rounded-xl p-m space-y-2">
                <span className="text-muted text-xs font-bold uppercase tracking-wider block">Format Preview</span>
                <pre className="text-foreground text-xs font-mono overflow-x-auto whitespace-pre">
                  {formatExamples[formatType]}
                </pre>
              </div>
            </div>

            {/* Step 3: Settings */}
            <div className="bg-card border border-border rounded-2xl p-l space-y-5 shadow-xs">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">3</span>
                <h3 className="text-base font-bold text-foreground">Settings</h3>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label htmlFor="num-samples-input" className="text-sm font-medium text-foreground">Number of Samples</label>
                  <input
                    id="num-samples-input"
                    type="number"
                    min="1"
                    max="1000"
                    value={numSamples}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      setNumSamples(isNaN(val) ? 1 : Math.min(Math.max(val, 1), 1000));
                    }}
                    className="w-16 h-8 bg-muted border border-border rounded-lg text-center text-sm font-bold text-foreground focus:outline-none focus:border-primary"
                  />
                </div>

                <div className="flex items-center gap-4">
                  <input
                    id="num-samples-slider"
                    type="range"
                    min="1"
                    max="100"
                    value={numSamples > 100 ? 100 : numSamples}
                    onChange={(e) => setNumSamples(parseInt(e.target.value))}
                    aria-label="Number of Samples"
                    className="w-full accent-primary bg-muted h-2 rounded-lg cursor-pointer"
                  />
                </div>

                <div className="flex flex-wrap gap-2">
                  {[5, 20, 50, 100, 500, 1000].map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => setNumSamples(n)}
                      className={`px-m py-s rounded-lg text-xs font-semibold transition-all ${
                        numSamples === n
                          ? "bg-primary text-white"
                          : "bg-muted text-muted border border-border hover:text-foreground"
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button"
                onClick={handleGenerate}
                disabled={
                  loading ||
                  (inputMode === "file" && !file) ||
                  (inputMode === "text" && !textInput.trim())
                }
                className="w-full h-high bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold text-base rounded-xl transition-all shadow-md active:scale-[0.99] flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <span>✨</span>
                    <span>Generate Dataset</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right Column: Output Card */}
          <div className="bg-card border border-border rounded-2xl p-l shadow-xs flex flex-col min-h-[500px] justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-foreground">Output</h3>
                {result?.success && result.data.length > 0 && (
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={handleCopy}
                      className="h-small px-m rounded-lg border border-border bg-card text-foreground hover:bg-muted text-xs font-semibold flex items-center gap-1.5 transition-all"
                    >
                      {copied ? "✓ Copied" : "📋 Copy"}
                    </button>
                    <button
                      type="button"
                      onClick={handleDownload}
                      className="h-small px-m rounded-lg border border-border bg-card text-foreground hover:bg-muted text-xs font-semibold flex items-center gap-1.5 transition-all shadow-2xs"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download JSON
                    </button>
                  </div>
                )}
              </div>

              {result ? (
                <div className="space-y-3">
                  <div className={`inline-flex items-center gap-2 px-m py-s rounded-lg text-xs font-semibold ${
                    result.success ? "bg-primary-light text-primary border border-primary/20" : "bg-muted text-foreground border border-border"
                  }`}>
                    <span>{result.success ? "✓" : "✕"}</span>
                    <span>{result.message}</span>
                  </div>

                  {result.data.length > 0 && (
                    <div className="bg-muted border border-border rounded-xl p-m max-h-[420px] overflow-auto">
                      <pre className="text-foreground text-xs font-mono whitespace-pre-wrap break-words">
                        {JSON.stringify(result.data, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              ) : (
                <div className="border border-dashed border-border rounded-xl p-xl my-auto min-h-[360px] flex items-center justify-center text-center">
                  <p className="text-muted text-sm font-medium">
                    Results appear here after you generate.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card py-m mt-12">
        <div className="max-w-7xl mx-auto px-l text-center">
          <p className="text-muted text-xs font-medium">
            Powered by LangChain & Ollama • Built with Next.js
          </p>
        </div>
      </footer>
    </div>
  );
}
