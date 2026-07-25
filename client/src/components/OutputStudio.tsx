"use client";

import { useState, useMemo, useEffect } from "react";
import { useWorkbenchStore } from "@/store/useWorkbenchStore";
import { GenerateResponse } from "@/app/api";
import { useGenerateDatasetStream } from "@/hooks/useGenerateDatasetStream";
import StreamPanel from "@/components/StreamPanel";
import { Code2, Table, LayoutGrid, Search, Copy, Check, Sparkles, Database, Zap, Sliders } from "lucide-react";

interface OutputStudioProps {
  result: GenerateResponse | null;
  loading: boolean;
  stream: ReturnType<typeof useGenerateDatasetStream>;
  onClearLogs: () => void;
}

export default function OutputStudio({ result, loading, stream, onClearLogs }: OutputStudioProps) {
  const { viewMode, setViewMode, searchFilter, setSearchFilter, activeStudioTab, setActiveStudioTab } =
    useWorkbenchStore();
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (stream.state.status === "streaming" && activeStudioTab === "dataset" && stream.state.samples.length === 0) {
      setActiveStudioTab("stream");
    }
  }, [stream.state.status, stream.state.samples.length, activeStudioTab, setActiveStudioTab]);

  const filteredData = useMemo(() => {
    if (!result?.data) return [];
    if (!searchFilter.trim()) return result.data;
    const query = searchFilter.toLowerCase();
    return result.data.filter((item) =>
      JSON.stringify(item).toLowerCase().includes(query)
    );
  }, [result?.data, searchFilter]);

  const handleCopy = () => {
    if (!result?.data) return;
    navigator.clipboard.writeText(JSON.stringify(result.data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!result?.data) return;
    const jsonStr = JSON.stringify(result.data, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dataset_${result.format_type || "export"}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const totalChars = useMemo(() => {
    if (!result?.data) return 0;
    return JSON.stringify(result.data).length;
  }, [result?.data]);

  return (
    <div className="bg-card border border-border rounded-2xl p-6 shadow-xs flex flex-col lg:h-[640px] w-full overflow-hidden">
      <div className="flex-1 flex flex-col min-h-0 gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex bg-muted p-1 rounded-xl border border-border">
              <button
                type="button"
                onClick={() => setActiveStudioTab("dataset")}
                className={`flex items-center gap-1.5 px-m h-small rounded-lg text-xs font-extrabold transition-all ${
                  activeStudioTab === "dataset"
                    ? "bg-card text-foreground shadow-2xs border border-border"
                    : "text-secondary hover:text-foreground"
                }`}
              >
                <Database className="w-3.5 h-3.5 text-primary" />
                <span>Dataset</span>
                {result?.success && (
                  <span className="bg-primary-light text-primary text-[10px] px-1.5 py-0.2 rounded-full font-mono">
                    {result.data.length}
                  </span>
                )}
              </button>
              <button
                type="button"
                onClick={() => setActiveStudioTab("stream")}
                className={`flex items-center gap-1.5 px-m h-small rounded-lg text-xs font-extrabold transition-all ${
                  activeStudioTab === "stream"
                    ? "bg-card text-foreground shadow-2xs border border-border"
                    : "text-secondary hover:text-foreground"
                }`}
              >
                <Zap className="w-3.5 h-3.5 text-primary" />
                <span>Live Stream</span>
                {stream.state.status === "streaming" && (
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse ml-0.5" />
                )}
              </button>
              <button
                type="button"
                onClick={() => setActiveStudioTab("split")}
                className={`flex items-center gap-1.5 px-m h-small rounded-lg text-xs font-extrabold transition-all ${
                  activeStudioTab === "split"
                    ? "bg-card text-foreground shadow-2xs border border-border"
                    : "text-secondary hover:text-foreground"
                }`}
              >
                <Sliders className="w-3.5 h-3.5 text-primary" />
                <span>Split View</span>
              </button>
            </div>
          </div>

          {result?.success && result.data.length > 0 && activeStudioTab !== "stream" && (
            <div className="flex items-center gap-2">
              <div className="flex bg-muted p-0.5 rounded-lg border border-border">
                <button
                  type="button"
                  onClick={() => setViewMode("json")}
                  aria-label="Raw JSON View"
                  className={`p-1.5 rounded-md text-xs font-bold transition-all ${
                    viewMode === "json"
                      ? "bg-card text-foreground shadow-2xs"
                      : "text-secondary hover:text-foreground"
                  }`}
                >
                  <Code2 className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("table")}
                  aria-label="Table View"
                  className={`p-1.5 rounded-md text-xs font-bold transition-all ${
                    viewMode === "table"
                      ? "bg-card text-foreground shadow-2xs"
                      : "text-secondary hover:text-foreground"
                  }`}
                >
                  <Table className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("cards")}
                  aria-label="Cards View"
                  className={`p-1.5 rounded-md text-xs font-bold transition-all ${
                    viewMode === "cards"
                      ? "bg-card text-foreground shadow-2xs"
                      : "text-secondary hover:text-foreground"
                  }`}
                >
                  <LayoutGrid className="w-4 h-4" />
                </button>
              </div>

              <button
                type="button"
                onClick={handleCopy}
                className="h-small px-m rounded-lg border border-border bg-card text-foreground hover:bg-muted text-xs font-bold flex items-center gap-1.5 transition-all shadow-2xs"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-success" /> : <Copy className="w-3.5 h-3.5 text-primary" />}
                <span>{copied ? "Copied" : "Copy"}</span>
              </button>

              <button
                type="button"
                onClick={handleDownload}
                className="h-small px-m rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-extrabold flex items-center gap-1.5 transition-all shadow-2xs"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                <span>Export JSON</span>
              </button>
            </div>
          )}
        </div>

        {stream.state.status === "streaming" && activeStudioTab === "dataset" && (
          <div className="flex items-center justify-between p-m bg-primary-light border border-primary/30 rounded-xl text-xs font-bold text-primary shrink-0">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 animate-spin" />
              <span>Synthesizing live LLM dataset ({stream.state.samples.length} samples generated so far)...</span>
            </div>
            <button
              type="button"
              onClick={() => setActiveStudioTab("stream")}
              className="text-xs font-extrabold underline hover:text-primary-hover"
            >
              Watch Stream & Traces
            </button>
          </div>
        )}

        {result?.success && result.data.length > 0 && activeStudioTab !== "stream" && (
          <div className="flex items-center justify-between gap-3 shrink-0">
            <div className="relative flex-1">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-secondary" />
              <input
                type="text"
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                placeholder="Search dataset samples..."
                className="w-full h-small pl-9 pr-m bg-card border border-border rounded-lg text-xs text-foreground font-semibold placeholder-secondary focus:outline-none focus:border-primary"
              />
            </div>
            <span className="text-xs text-secondary font-bold font-mono hidden sm:inline">
              {(totalChars / 1024).toFixed(1)} KB
            </span>
          </div>
        )}

        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {activeStudioTab === "stream" && (
            <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
              <StreamPanel state={stream.state} onClearLogs={onClearLogs} />
            </div>
          )}

          {activeStudioTab === "split" && (
            <div className="flex-1 min-h-0 flex flex-col md:flex-row gap-3 overflow-hidden">
              <div className="flex-1 min-w-0 h-1/2 md:h-full flex flex-col overflow-hidden">
                <StreamPanel state={stream.state} onClearLogs={onClearLogs} />
              </div>
              <div className="flex-1 min-w-0 h-1/2 md:h-full overflow-y-auto rounded-xl border border-border bg-card p-m">
                {result?.success && result.data.length > 0 ? (
                  <>
                    {viewMode === "json" && (
                      <pre className="text-foreground text-xs font-mono font-semibold whitespace-pre-wrap break-words">
                        {JSON.stringify(filteredData, null, 2)}
                      </pre>
                    )}

                    {viewMode === "table" && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs text-foreground">
                          <thead className="border-b border-border bg-muted text-secondary uppercase tracking-wider font-extrabold">
                            <tr>
                              <th className="p-s">#</th>
                              {Object.keys(filteredData[0] || {}).map((key) => (
                                <th key={key} className="p-s">
                                  {key}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {filteredData.map((row, idx) => (
                              <tr key={idx} className="hover:bg-muted/50">
                                <td className="p-s text-secondary font-bold font-mono">{idx + 1}</td>
                                {Object.keys(filteredData[0] || {}).map((key) => (
                                  <td key={key} className="p-s max-w-xs truncate font-mono text-[11px] font-semibold">
                                    {typeof row[key] === "object"
                                      ? JSON.stringify(row[key])
                                      : String(row[key])}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {viewMode === "cards" && (
                      <div className="grid gap-3">
                        {filteredData.map((item, idx) => (
                          <div
                            key={idx}
                            className="bg-card border border-border rounded-xl p-m space-y-2 shadow-2xs"
                          >
                            <div className="flex items-center justify-between text-xs font-extrabold text-secondary border-b border-border pb-1">
                              <span>Sample #{idx + 1}</span>
                              <span className="uppercase text-[10px] font-mono text-primary font-bold">
                                {result.format_type}
                              </span>
                            </div>
                            <pre className="text-foreground text-xs font-mono font-semibold whitespace-pre-wrap break-words">
                              {JSON.stringify(item, null, 2)}
                            </pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-secondary text-xs font-bold text-center py-l">
                    No samples generated yet.
                  </div>
                )}
              </div>
            </div>
          )}

          {activeStudioTab === "dataset" && (
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              {loading && (!result || result.data.length === 0) ? (
                <div className="border-2 border-dashed border-border rounded-xl p-xl flex flex-col items-center justify-center text-center space-y-3 bg-muted/40 flex-1 min-h-0">
                  <div className="w-12 h-12 rounded-2xl bg-primary-light flex items-center justify-center text-primary animate-pulse">
                    <Sparkles className="w-6 h-6" />
                  </div>
                  <p className="text-foreground font-extrabold text-sm">
                    Synthesizing dataset with LLM...
                  </p>
                  <p className="text-secondary font-bold text-xs max-w-xs">
                    Formatting and verifying JSON output samples.
                  </p>
                </div>
              ) : result?.success ? (
                <div className="flex-1 min-h-0 overflow-y-auto rounded-xl border border-border bg-card p-m">
                  {viewMode === "json" && (
                    <pre className="text-foreground text-xs font-mono font-semibold whitespace-pre-wrap break-words">
                      {JSON.stringify(filteredData, null, 2)}
                    </pre>
                  )}

                  {viewMode === "table" && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs text-foreground">
                        <thead className="border-b border-border bg-muted text-secondary uppercase tracking-wider font-extrabold">
                          <tr>
                            <th className="p-s">#</th>
                            {Object.keys(filteredData[0] || {}).map((key) => (
                              <th key={key} className="p-s">
                                {key}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {filteredData.map((row, idx) => (
                            <tr key={idx} className="hover:bg-muted/50">
                              <td className="p-s text-secondary font-bold font-mono">{idx + 1}</td>
                              {Object.keys(filteredData[0] || {}).map((key) => (
                                <td key={key} className="p-s max-w-xs truncate font-mono text-[11px] font-semibold">
                                  {typeof row[key] === "object"
                                    ? JSON.stringify(row[key])
                                    : String(row[key])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {viewMode === "cards" && (
                    <div className="grid gap-3">
                      {filteredData.map((item, idx) => (
                        <div
                          key={idx}
                          className="bg-card border border-border rounded-xl p-m space-y-2 shadow-2xs"
                        >
                          <div className="flex items-center justify-between text-xs font-extrabold text-secondary border-b border-border pb-1">
                            <span>Sample #{idx + 1}</span>
                            <span className="uppercase text-[10px] font-mono text-primary font-bold">
                              {result.format_type}
                            </span>
                          </div>
                          <pre className="text-foreground text-xs font-mono font-semibold whitespace-pre-wrap break-words">
                            {JSON.stringify(item, null, 2)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="border-2 border-dashed border-border/80 rounded-2xl p-8 flex flex-col items-center justify-center text-center space-y-4 flex-1 bg-muted/20 min-h-0">
                  <div className="w-14 h-14 rounded-2xl bg-card border border-border flex items-center justify-center text-primary shadow-xs">
                    <Code2 className="w-7 h-7" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-foreground font-black text-base tracking-tight">
                      Ready to Synthesize
                    </p>
                    <p className="text-secondary font-semibold text-xs max-w-sm leading-relaxed">
                      Upload text or paste content on the left, then click Generate Dataset to view live JSON samples here.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
