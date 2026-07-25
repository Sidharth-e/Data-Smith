"use client";

import { useEffect, useRef, useState } from "react";
import { StreamState } from "@/hooks/useGenerateDatasetStream";
import {
  Zap,
  Cpu,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Layers,
  X,
  Check,
} from "lucide-react";

interface StreamPanelProps {
  state: StreamState;
  onClearLogs: () => void;
}

const kindStyles: Record<string, string> = {
  info: "text-secondary",
  thinking: "text-primary font-semibold",
  token: "text-foreground",
  batch: "text-success font-bold",
  warning: "text-warning font-bold",
  error: "text-error font-bold",
};

export default function StreamPanel({ state, onClearLogs }: StreamPanelProps) {
  const logsRef = useRef<HTMLDivElement>(null);
  const [activeView, setActiveView] = useState<"batches" | "logs" | "unified">("unified");
  const [showThinking, setShowThinking] = useState<Record<number, boolean>>({});
  const [collapsedBatches, setCollapsedBatches] = useState<Record<number, boolean>>({});
  const [copiedBatchIndex, setCopiedBatchIndex] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [state.logs.length, activeView]);

  useEffect(() => {
    if (state.logs.length === 0) {
      setElapsedSeconds(0);
      return;
    }

    const startTs = state.logs[0].ts;

    if (state.status === "streaming") {
      const interval = setInterval(() => {
        setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startTs) / 1000)));
      }, 1000);
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startTs) / 1000)));
      return () => clearInterval(interval);
    } else {
      const endTs = state.logs[state.logs.length - 1]?.ts || Date.now();
      setElapsedSeconds(Math.max(0, Math.floor((endTs - startTs) / 1000)));
    }
  }, [state.status, state.logs]);

  const progressPct =
    state.progressTotal > 0
      ? Math.min(100, Math.round((state.progressDone / state.progressTotal) * 100))
      : 0;

  const streaming = state.status === "streaming";

  const formatElapsed = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  const handleCopyContent = (content: string, index: number) => {
    navigator.clipboard.writeText(content);
    setCopiedBatchIndex(index);
    setTimeout(() => setCopiedBatchIndex(null), 2000);
  };

  const toggleBatchCollapse = (index: number) => {
    setCollapsedBatches((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <div className="bg-card border border-border rounded-2xl shadow-xs flex flex-col h-full overflow-hidden">
      <div className="border-b border-border bg-card shrink-0">
        <div className="flex items-center justify-between p-m gap-2 border-b border-border/50">
          <div className="flex items-center gap-1.5 shrink-0">
            <Zap className="w-4 h-4 text-primary shrink-0" />
            <h3 className="text-sm font-extrabold text-foreground tracking-tight whitespace-nowrap">
              Live Stream
            </h3>
            {streaming && (
              <span className="flex items-center gap-1 text-[10px] font-bold text-primary bg-primary-light border border-primary/30 px-2 py-0.5 rounded-full shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                LIVE
              </span>
            )}
            {state.status === "error" && (
              <span className="flex items-center gap-1 text-[10px] font-bold text-error bg-error/10 border border-error/30 px-2 py-0.5 rounded-full shrink-0">
                <AlertCircle className="w-3 h-3" />
                ERROR
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-muted p-0.5 rounded-lg border border-border">
              <button
                type="button"
                onClick={() => setActiveView("unified")}
                className={`px-2 py-0.5 rounded text-[10px] font-extrabold transition-all ${
                  activeView === "unified"
                    ? "bg-card text-foreground shadow-2xs"
                    : "text-secondary hover:text-foreground"
                }`}
              >
                Unified
              </button>
              <button
                type="button"
                onClick={() => setActiveView("batches")}
                className={`px-2 py-0.5 rounded text-[10px] font-extrabold transition-all ${
                  activeView === "batches"
                    ? "bg-card text-foreground shadow-2xs"
                    : "text-secondary hover:text-foreground"
                }`}
              >
                Batches ({state.batches.length})
              </button>
              <button
                type="button"
                onClick={() => setActiveView("logs")}
                className={`px-2 py-0.5 rounded text-[10px] font-extrabold transition-all ${
                  activeView === "logs"
                    ? "bg-card text-foreground shadow-2xs"
                    : "text-secondary hover:text-foreground"
                }`}
              >
                Logs ({state.logs.length})
              </button>
            </div>

            <button
              type="button"
              onClick={onClearLogs}
              className="text-secondary hover:text-foreground p-1 rounded-md hover:bg-muted transition-colors"
              title="Clear logs"
              aria-label="Clear logs"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="px-m py-1.5 bg-muted/30 flex items-center justify-between text-[11px] font-mono font-bold text-secondary gap-2">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-foreground">
              <Layers className="w-3 h-3 text-primary" />
              <span className="text-secondary uppercase text-[9px]">Batches:</span>
              <span>{state.progressDone}/{state.progressTotal || state.numBatches}</span>
            </span>
            <span className="flex items-center gap-1 text-foreground">
              <Cpu className="w-3 h-3 text-primary" />
              <span className="text-secondary uppercase text-[9px]">Samples:</span>
              <span>{state.samples.length}/{state.numSamples || 0}</span>
            </span>
          </div>
          <span className="flex items-center gap-1 text-foreground">
            <span className="text-secondary uppercase text-[9px]">Progress:</span>
            <span className="text-primary font-extrabold">{progressPct}%</span>
          </span>
        </div>
      </div>

      <div className="h-1 bg-muted border-b border-border shrink-0">
        <div
          className={`h-full transition-all duration-300 ${
            state.status === "error" ? "bg-error" : "bg-primary"
          }`}
          style={{ width: `${progressPct}%` }}
        />
      </div>

      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {(activeView === "batches" || activeView === "unified") && (
          <div className={`overflow-y-auto p-m space-y-3 min-h-0 ${activeView === "unified" ? "flex-1 border-b border-border" : "flex-1"}`}>
            {state.batches.length === 0 && state.status === "idle" && (
              <div className="text-secondary text-xs font-bold text-center py-xl">
                No active generation. Click <span className="text-primary">Generate</span> to start streaming.
              </div>
            )}

            {state.batches.map((b) => {
              const open = showThinking[b.index];
              const isCollapsed = collapsedBatches[b.index] ?? false;

              return (
                <div
                  key={b.index}
                  className="border border-border rounded-xl bg-card shadow-2xs overflow-hidden"
                >
                  <div className="flex items-center justify-between px-m py-s text-xs font-bold bg-muted/40 border-b border-border select-none">
                    <button
                      type="button"
                      onClick={() => toggleBatchCollapse(b.index)}
                      className="flex items-center gap-2 text-left flex-1 min-w-0 hover:opacity-80 transition-opacity"
                    >
                      {isCollapsed ? (
                        <svg className="w-4 h-4 text-secondary shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4 text-secondary shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                        </svg>
                      )}
                      {b.done ? (
                        <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
                      ) : (
                        <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0" />
                      )}
                      <span className="text-foreground font-extrabold truncate">
                        Batch #{b.index + 1}
                      </span>
                      {b.done && (
                        <span className="bg-success/10 text-success border border-success/30 px-2 py-0.5 rounded-full text-[10px] font-mono shrink-0">
                          {b.samples.length} samples
                        </span>
                      )}
                    </button>

                    <div className="flex items-center gap-2 shrink-0">
                      {b.content && (
                        <button
                          type="button"
                          onClick={() => handleCopyContent(b.content, b.index)}
                          className="p-1 rounded text-secondary hover:text-foreground hover:bg-muted transition-colors"
                          title="Copy content"
                        >
                          {copiedBatchIndex === b.index ? (
                            <Check className="w-3 h-3 text-success" />
                          ) : (
                            <svg className="w-3 h-3 text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                            </svg>
                          )}
                        </button>
                      )}

                      {b.thinking && !isCollapsed && (
                        <button
                          type="button"
                          onClick={() =>
                            setShowThinking((s) => ({ ...s, [b.index]: !s[b.index] }))
                          }
                          className="flex items-center gap-1 text-primary hover:underline text-[10px] font-extrabold"
                        >
                          <Cpu className="w-3 h-3" />
                          <span>{open ? "Hide Reasoning" : "Show Reasoning"}</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {!isCollapsed && (
                    <>
                      {b.thinking && open && (
                        <div className="p-m text-[11px] font-mono text-primary whitespace-pre-wrap break-words border-b border-border bg-primary/5">
                          {b.thinking}
                        </div>
                      )}

                      {b.content && (
                        <div className="p-m text-[11px] font-mono text-foreground whitespace-pre-wrap break-words bg-card">
                          {b.content.slice(-1000)}
                          {!b.done && (
                            <span className="inline-block w-1.5 h-3.5 bg-primary animate-pulse ml-1 align-middle" />
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })}

            {state.errorMessage && (
              <div className="flex items-start gap-2 p-m bg-error/10 border border-error/30 rounded-xl text-error text-xs font-bold">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{state.errorMessage}</span>
              </div>
            )}
          </div>
        )}

        {(activeView === "logs" || activeView === "unified") && (
          <div
            className={`bg-muted/20 overflow-y-auto flex flex-col font-mono text-xs ${
              activeView === "unified" ? "h-36 shrink-0 min-h-0" : "flex-1 min-h-0"
            }`}
            ref={logsRef}
          >
            <div className="px-m py-1.5 border-b border-border/50 bg-muted/40 text-secondary text-[10px] font-bold uppercase tracking-wider flex items-center justify-between sticky top-0 bg-muted/90 backdrop-blur-xs">
              <div className="flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-primary shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 17l6-6-6-6M12 19h8" />
                </svg>
                <span className="text-foreground font-extrabold text-xs">Log Timeline</span>
              </div>
              <div className="flex items-center gap-1.5 text-secondary text-[11px] font-mono font-bold">
                <svg className="w-3 h-3 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="9" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 3" />
                </svg>
                <span>{formatElapsed(elapsedSeconds)}</span>
              </div>
            </div>

            <div className="p-2 space-y-1 flex-1 overflow-y-auto">
              {state.logs.length === 0 ? (
                <div className="text-secondary text-[11px] italic p-2">
                  No log entries recorded.
                </div>
              ) : (
                state.logs.map((log) => (
                  <div
                    key={log.id}
                    className={`text-[11px] flex items-start gap-2 ${
                      kindStyles[log.kind] || "text-secondary"
                    }`}
                  >
                    <span className="text-secondary/50 shrink-0 select-none">
                      {new Date(log.ts).toLocaleTimeString()}
                    </span>
                    <span className="text-secondary/60 shrink-0 select-none">›</span>
                    <span className="break-words flex-1">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}