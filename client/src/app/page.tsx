"use client";

import { useState, useEffect } from "react";
import { useWorkbenchStore } from "@/store/useWorkbenchStore";
import { GenerateResponse } from "./api";
import Navbar from "@/components/Navbar";
import InputWorkbench from "@/components/InputWorkbench";
import OutputStudio from "@/components/OutputStudio";
import CommandPalette from "@/components/CommandPalette";
import { Terminal } from "lucide-react";

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
          <p>Powered by LangChain Agents</p>
        </div>
      </footer>

      <CommandPalette />
    </div>
  );
}
