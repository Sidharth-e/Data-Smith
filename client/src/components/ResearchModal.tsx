"use client";

import { useState, useEffect, useRef } from "react";
import {
  Search,
  X,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  FileText,
  Layers,
  ShieldCheck,
  Code2,
  FileCode2,
} from "lucide-react";
import { useResearchStream } from "@/hooks/useResearchStream";
import { ResearchAgentName } from "@/app/api";

interface ResearchModalProps {
  open: boolean;
  onClose: () => void;
  onApprove: (document: string, topic: string) => void;
}

const AGENT_META: Record<
  ResearchAgentName,
  { label: string; icon: typeof Search; blurb: string }
> = {
  planner: {
    label: "Planner",
    icon: Search,
    blurb: "Breaks the topic into research questions",
  },
  researcher: {
    label: "Researcher",
    icon: FileText,
    blurb: "Runs web search for each question",
  },
  gap: {
    label: "Gap",
    icon: Layers,
    blurb: "Finds angles still missing",
  },
  quality: {
    label: "Quality",
    icon: ShieldCheck,
    blurb: "Drops duplicates & low-signal snippets",
  },
  writer: {
    label: "Writer",
    icon: Code2,
    blurb: "Synthesizes a markdown source document",
  },
};

export default function ResearchModal({ open, onClose, onApprove }: ResearchModalProps) {
  const stream = useResearchStream();
  const [topic, setTopic] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const docRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      stream.reset();
      setTopic("");
      setTimeout(() => inputRef.current?.focus(), 50);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (docRef.current) {
      docRef.current.scrollTop = docRef.current.scrollHeight;
    }
  }, [stream.state.document]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        handleClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const handleStart = async () => {
    if (!topic.trim()) return;
    try {
      await stream.mutateAsync(topic);
    } catch {
      // surfaced via stream.state
    }
  };

  const handleClose = () => {
    if (stream.isPending) stream.abort();
    stream.reset();
    setTopic("");
    onClose();
  };

  const handleApprove = () => {
    if (!stream.state.document) return;
    onApprove(stream.state.document, stream.state.topic);
    stream.reset();
    setTopic("");
    onClose();
  };

  const handleDiscard = () => {
    stream.reset();
    setTopic("");
  };

  const { state } = stream;
  const hasDoc = state.document.length > 0;
  const canApprove = state.status === "done" && hasDoc;

  return (
    <div
      className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-m"
      onClick={handleClose}
    >
      <div
        className="w-full max-w-4xl max-h-[90vh] bg-card border-2 border-border rounded-2xl shadow-xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-l py-m border-b border-border bg-muted/30 shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-primary-light text-primary flex items-center justify-center border border-primary/30">
              <Search className="w-4.5 h-4.5" />
            </div>
            <div>
              <h2 className="text-sm font-black text-foreground tracking-tight">
                Research & Generate Source
              </h2>
              <p className="text-[11px] text-secondary font-bold">
                Multi-agent pipeline · planner → researcher → gap → quality → writer
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleClose}
            aria-label="Close"
            className="p-1.5 rounded-lg text-secondary hover:text-foreground hover:bg-muted cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Topic input (only when idle / not started) */}
        {state.status === "idle" && (
          <div className="p-l space-y-3 border-b border-border shrink-0">
            <label className="text-xs font-extrabold uppercase tracking-wider text-secondary block">
              Topic to research
            </label>
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && topic.trim()) handleStart();
                }}
                placeholder="e.g. Retrieval-Augmented Generation for code assistants"
                className="flex-1 h-11 px-m bg-card border border-border rounded-xl text-sm text-foreground font-semibold placeholder-secondary focus:outline-none focus:border-primary"
              />
              <button
                type="button"
                onClick={handleStart}
                disabled={!topic.trim()}
                className="h-11 px-l bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-extrabold text-sm rounded-xl transition-all flex items-center gap-2 cursor-pointer"
              >
                <Sparkles className="w-4 h-4" />
                <span>Start</span>
              </button>
            </div>
            <p className="text-[11px] text-secondary font-bold">
              The agents will research the web (DuckDuckGo) and synthesize a source
              document you can approve and feed into dataset generation.
            </p>
          </div>
        )}

        {/* Body: agents + live view */}
        {(state.status !== "idle" || hasDoc) && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            {/* Agent progress strip */}
            <div className="px-l py-m border-b border-border bg-muted/20 shrink-0">
              <div className="grid grid-cols-5 gap-2">
                {state.agents.map((a) => {
                  const meta = AGENT_META[a.name];
                  const Icon = meta.icon;
                  const running = a.status === "running";
                  const done = a.status === "done";
                  return (
                    <div
                      key={a.name}
                      className={`rounded-xl border p-s transition-all ${
                        running
                          ? "border-primary bg-primary-light"
                          : done
                            ? "border-success/40 bg-success/10"
                            : "border-border bg-card"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        {running ? (
                          <Loader2 className="w-3.5 h-3.5 text-primary animate-spin shrink-0" />
                        ) : done ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-success shrink-0" />
                        ) : (
                          <Icon className="w-3.5 h-3.5 text-secondary shrink-0" />
                        )}
                        <span
                          className={`text-[11px] font-extrabold truncate ${
                            running
                              ? "text-primary"
                              : done
                                ? "text-success"
                                : "text-secondary"
                          }`}
                        >
                          {meta.label}
                        </span>
                      </div>
                      <p className="text-[10px] text-secondary font-semibold mt-0.5 leading-tight hidden sm:block">
                        {meta.blurb}
                      </p>
                      {a.message && (
                        <p className="text-[10px] text-foreground/70 font-bold mt-1 truncate">
                          {a.message}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Live content */}
            <div className="flex-1 min-h-0 overflow-auto p-l space-y-3" ref={docRef}>
              {/* Plan */}
              {state.plan.length > 0 && (
                <Section title="Research Plan" icon={<Search className="w-3.5 h-3.5" />}>
                  <ol className="space-y-1 list-decimal list-inside text-xs font-semibold text-foreground">
                    {state.plan.map((q, i) => (
                      <li key={i}>{q}</li>
                    ))}
                  </ol>
                </Section>
              )}

              {/* Snippets */}
              {state.snippets.length > 0 && (
                <Section
                  title={`Collected Snippets (${state.snippets.length})`}
                  icon={<FileText className="w-3.5 h-3.5" />}
                >
                  <div className="space-y-1.5">
                    {state.snippets.map((s, i) => (
                      <div
                        key={i}
                        className="text-[11px] border border-border rounded-lg p-s bg-card"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-extrabold text-foreground truncate">
                            {s.title || "(no title)"}
                          </span>
                          {s.url && (
                            <a
                              href={s.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-primary text-[10px] hover:underline shrink-0"
                            >
                              source ↗
                            </a>
                          )}
                        </div>
                        {s.snippet && (
                          <p className="text-secondary mt-0.5 line-clamp-2">
                            {s.snippet}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </Section>
              )}

              {/* Gap questions */}
              {state.gapQuestions.length > 0 && (
                <Section
                  title="Gap Follow-ups"
                  icon={<Layers className="w-3.5 h-3.5" />}
                >
                  <ul className="space-y-1 list-disc list-inside text-xs font-semibold text-foreground">
                    {state.gapQuestions.map((q, i) => (
                      <li key={i}>{q}</li>
                    ))}
                  </ul>
                </Section>
              )}

              {/* Quality count */}
              {state.qualityCount > 0 && (
                <Section
                  title="Quality Filter"
                  icon={<ShieldCheck className="w-3.5 h-3.5" />}
                >
                  <p className="text-xs font-semibold text-foreground">
                    Kept <span className="text-success font-extrabold">{state.qualityCount}</span>{" "}
                    high-signal snippets after dedup & relevance filtering.
                  </p>
                </Section>
              )}

              {/* Writer document */}
              {hasDoc && (
                <Section
                  title="Synthesized Source Document"
                  icon={<Code2 className="w-3.5 h-3.5" />}
                >
                  <pre className="text-[11px] font-mono text-foreground/90 whitespace-pre-wrap break-words border border-border rounded-lg p-m bg-muted/30 max-h-72 overflow-auto">
                    {state.document}
                  </pre>
                </Section>
              )}

              {/* Live log tail */}
              {state.logs.length > 0 && (
                <div className="border-t border-border pt-2 mt-2">
                  <p className="text-[10px] font-extrabold uppercase tracking-wider text-secondary mb-1">
                    Activity
                  </p>
                  <div className="space-y-0.5 max-h-32 overflow-auto">
                    {state.logs.map((log) => (
                      <div
                        key={log.id}
                        className="text-[10px] font-mono text-secondary"
                      >
                        <span className="text-secondary/60">
                          {new Date(log.ts).toLocaleTimeString()} ›{" "}
                        </span>
                        {log.text}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Error */}
              {state.status === "error" && state.errorMessage && (
                <div className="flex items-start gap-2 p-m bg-error/10 border border-error/30 rounded-xl text-error text-xs font-bold">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{state.errorMessage}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer actions */}
        {state.status !== "idle" && (
          <div className="flex items-center justify-between gap-2 px-l py-m border-t border-border bg-muted/30 shrink-0">
            <div className="flex items-center gap-2 text-[11px] font-bold text-secondary">
              {stream.isPending ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
                  <span>Agents working…</span>
                </>
              ) : state.status === "done" ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                  <span>Document ready ({state.document.length} chars)</span>
                </>
              ) : state.status === "error" ? (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-error" />
                  <span>Research failed</span>
                </>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              {stream.isPending && (
                <button
                  type="button"
                  onClick={stream.abort}
                  className="h-9 px-m rounded-lg bg-muted hover:bg-border text-foreground text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
                >
                  <X className="w-3.5 h-3.5 text-error" />
                  <span>Stop</span>
                </button>
              )}
              {!stream.isPending && state.status !== "done" && (
                <button
                  type="button"
                  onClick={handleDiscard}
                  className="h-9 px-m rounded-lg bg-muted hover:bg-border text-foreground text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
                >
                  <FileCode2 className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
              )}
              <button
                type="button"
                onClick={handleApprove}
                disabled={!canApprove}
                className="h-9 px-l rounded-lg bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Approve & Use for Dataset</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-[11px] font-extrabold uppercase tracking-wider text-primary">
        {icon}
        <span>{title}</span>
      </div>
      {children}
    </div>
  );
}