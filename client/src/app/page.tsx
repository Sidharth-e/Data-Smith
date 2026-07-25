"use client";

import { useState, useEffect } from "react";
import { useWorkbenchStore } from "@/store/useWorkbenchStore";
import { GenerateResponse } from "./api";
import Navbar from "@/components/Navbar";
import InputWorkbench from "@/components/InputWorkbench";
import OutputStudio from "@/components/OutputStudio";
import CommandPalette from "@/components/CommandPalette";
import { Sparkles, Terminal, FileCode2, ShieldCheck, Zap } from "lucide-react";

export default function Home() {
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const { isDarkMode } = useWorkbenchStore();

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  const handleGenerateStart = () => {
    setIsGenerating(true);
  };

  const handleGenerateSuccess = (res: GenerateResponse) => {
    setResult(res);
    setIsGenerating(false);
  };

  const handleGenerateError = (errorMsg: string) => {
    setResult({
      success: false,
      format_type: "alpaca",
      data: [],
      message: errorMsg,
    });
    setIsGenerating(false);
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col transition-colors duration-200">
      <Navbar />

      <main className="max-w-7xl mx-auto px-l py-l space-y-8 flex-1 w-full">
        <section className="bg-card border-2 border-primary/20 rounded-2xl p-l md:p-xl shadow-xs relative overflow-hidden">
          <div className="grid md:grid-cols-3 gap-6 items-center relative z-10">
            <div className="md:col-span-2 space-y-4">
              <div className="inline-flex items-center gap-2 bg-primary-light text-primary text-xs font-extrabold px-m py-s rounded-full border border-primary/30 uppercase tracking-wider shadow-2xs">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Next-Gen Fine-Tuning Synthesis Studio</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-black text-foreground tracking-tight leading-tight">
                Synthesize Fine-Tuning Datasets in Seconds
              </h2>
              <p className="text-secondary text-sm md:text-base font-semibold leading-relaxed max-w-2xl">
                Transform raw text or document uploads into instruction-aligned Alpaca, OpenAI Chat, or Completion JSON samples for training LLMs.
              </p>
              <div className="flex flex-wrap items-center gap-4 pt-2 text-xs font-bold text-secondary">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-success" />
                  Deterministic Parsing
                </span>
                <span className="flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-warning" />
                  LangChain & Ollama Backend
                </span>
                <span className="flex items-center gap-1.5">
                  <FileCode2 className="w-4 h-4 text-primary" />
                  Instant JSON Export
                </span>
              </div>
            </div>

            <div className="bg-muted border border-border rounded-2xl p-l shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-secondary uppercase tracking-wider">
                  Schema Accuracy
                </span>
                <span className="text-xs font-extrabold text-success">100% Validated</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl md:text-4xl font-black text-foreground">
                  JSONL
                </span>
                <span className="text-xs text-secondary font-extrabold">ready schema</span>
              </div>
              <div className="w-full bg-card border border-border h-2 rounded-full overflow-hidden">
                <div className="bg-primary h-full rounded-full w-[92%] transition-all duration-500" />
              </div>
              <div className="flex items-center justify-between text-[11px] text-secondary font-bold pt-1">
                <span>Alpaca</span>
                <span>OpenAI Chat</span>
                <span>Completion</span>
              </div>
            </div>
          </div>
        </section>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          <InputWorkbench
            onGenerateStart={handleGenerateStart}
            onGenerateSuccess={handleGenerateSuccess}
            onGenerateError={handleGenerateError}
          />
          <OutputStudio result={result} loading={isGenerating} />
        </div>
      </main>

      <footer className="border-t border-border bg-card py-m mt-12">
        <div className="max-w-7xl mx-auto px-l flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-secondary font-bold">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-primary" />
            <span className="text-foreground">Data Smith Workbench v1.0</span>
          </div>
          <p>Powered by LangChain, Ollama & Next.js App Router</p>
        </div>
      </footer>

      <CommandPalette />
    </div>
  );
}
