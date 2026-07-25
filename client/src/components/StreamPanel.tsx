"use client";

import { useEffect, useRef, useState } from "react";
import { StreamState } from "@/hooks/useGenerateDatasetStream";
import {
  Zap,
  Cpu,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Code2,
  Layers,
  X,
} from "lucide-react";

interface StreamPanelProps {
  state: StreamState;
  onClearLogs: () => void;
}

const kindStyles: Record<string, string> = {
  info: "text-secondary",
  thinking: "text-primary",
  token: "text-foreground",
  batch: "text-success",
  warning: "text-warning",
  error: "text-error",
};

export default function StreamPanel({ state, onClearLogs }: StreamPanelProps) {
  const logsRef = useRef<HTMLDivElement>(null);
  const [showThinking, setShowThinking] = useState<Record<number, boolean>>({});

  // Auto-scroll the log to the latest entry while streaming.
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [state.logs.length]);

  const progressPct =
    state.progressTotal > 0
      ? Math.min(100, Math.round((state.progressDone / state.progressTotal) * 100))
      : 0;

  const streaming = state.status === "streaming";

  return (
    <div className="bg-card border border-border rounded-2xl shadow-xs flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border p-m">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-extrabold text-foreground">
            Live Generation Stream
          </h3>
          {streaming && (
            <span className="flex items-center gap-1 text-[10px] font-bold text-primary bg-primary-light border border-primary/30 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              LIVE
            </span>
          )}
          {state.status === "done" && (
            <span className="flex items-center gap-1 text-[10px] font-bold text-success bg-success/10 border border-success/30 px-2 py-0.5 rounded-full">
              <CheckCircle2 className="w-3 h-3" />
              DONE
            </span>
          )}
          {state.status === "error" && (
            <span className="flex items-center gap-1 text-[10px] font-bold text-error bg-error/10 border border-error/30 px-2 py-0.5 rounded-full">
              <AlertCircle className="w-3 h-3" />
              ERROR
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onClearLogs}
          className="text-secondary hover:text-foreground p-1 rounded transition-colors"
          aria-label="Clear logs"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 p-m border-b border-border bg-muted/30">
        <Stat
          icon={<Layers className="w-3.5 h-3.5" />}
          label="Batches"
          value={`${state.progressDone}/${state.progressTotal || state.numBatches}`}
        />
        <Stat
          icon={<Cpu className="w-3.5 h-3.5" />}
          label="Samples"
          value={`${state.samples.length}/${state.numSamples || 0}`}
        />
        <Stat
          icon={<Zap className="w-3.5 h-3.5" />}
          label="Progress"
          value={`${progressPct}%`}
        />
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-muted border-b border-border">
        <div
          className={`h-full transition-all duration-300 ${
            state.status === "error" ? "bg-error" : "bg-primary"
          }`}
          style={{ width: `${progressPct}%` }}
        />
      </div>

      {/* Batch cards (thinking + content per batch) */}
      <div className="flex-1 overflow-auto p-m space-y-2 min-h-0">
        {state.batches.length === 0 && state.status === "idle" && (
          <div className="text-secondary text-xs font-bold text-center py-xl">
            No active generation. Click <span className="text-primary">Generate</span> to start streaming.
          </div>
        )}

        {state.batches.map((b) => {
          const open = showThinking[b.index];
          return (
            <div
              key={b.index}
              className="border border-border rounded-xl bg-muted/30 overflow-hidden"
            >
              <div className="flex items-center justify-between px-m py-s text-xs font-bold">
                <div className="flex items-center gap-2">
                  {b.done ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                  ) : (
                    <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />
                  )}
                  <span className="text-foreground">
                    Batch #{b.index + 1}
                  </span>
                  {b.done && (
                    <span className="text-success text-[10px]">
                      {b.samples.length} samples
                    </span>
                  )}
                </div>
                {b.thinking && (
                  <button
                    type="button"
                    onClick={() =>
                      setShowThinking((s) => ({ ...s, [b.index]: !s[b.index] }))
                    }
                    className="flex items-center gap-1 text-primary hover:underline text-[10px]"
                  >
                    <Cpu className="w-3 h-3" />
                    {open ? "Hide thinking" : "Show thinking"}
                    <Code2
                      className={`w-3 h-3 transition-transform ${
                        open ? "rotate-90" : ""
                      }`}
                    />
                  </button>
                )}
              </div>

              {b.thinking && open && (
                <div className="px-m pb-s text-[11px] font-mono text-primary whitespace-pre-wrap break-words border-t border-border bg-primary/5">
                  {b.thinking}
                </div>
              )}

              {b.content && (
                <div className="px-m py-s text-[11px] font-mono text-foreground/80 whitespace-pre-wrap break-words border-t border-border max-h-40 overflow-auto">
                  {b.content.slice(-800)}
                  {!b.done && (
                    <span className="inline-block w-1.5 h-3 bg-primary animate-pulse ml-0.5 align-middle" />
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* Error / warning display */}
        {state.errorMessage && (
          <div className="flex items-start gap-2 p-m bg-error/10 border border-error/30 rounded-xl text-error text-xs font-bold">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{state.errorMessage}</span>
          </div>
        )}
      </div>

      {/* Log tail */}
      {state.logs.length > 0 && (
        <div className="border-t border-border bg-muted/20 max-h-32 overflow-auto" ref={logsRef}>
          <div className="p-s space-y-0.5">
            {state.logs.map((log) => (
              <div
                key={log.id}
                className={`text-[10px] font-mono ${kindStyles[log.kind] || "text-secondary"}`}
              >
                <span className="text-secondary/60">
                  {new Date(log.ts).toLocaleTimeString()} ›{" "}
                </span>
                {log.message}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-1 text-secondary text-[10px] font-bold uppercase tracking-wider">
        {icon}
        {label}
      </div>
      <div className="text-foreground text-sm font-extrabold font-mono">
        {value}
      </div>
    </div>
  );
}