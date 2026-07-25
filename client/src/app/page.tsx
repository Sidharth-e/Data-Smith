"use client";

import { useEffect, useMemo } from "react";
import { useWorkbenchStore } from "@/store/useWorkbenchStore";
import Navbar from "@/components/Navbar";
import InputWorkbench from "@/components/InputWorkbench";
import OutputStudio from "@/components/OutputStudio";
import StreamPanel from "@/components/StreamPanel";
import CommandPalette from "@/components/CommandPalette";
import { Terminal } from "lucide-react";
import { useGenerateDatasetStream } from "@/hooks/useGenerateDatasetStream";

export default function Home() {
  const { isDarkMode } = useWorkbenchStore();
  const stream = useGenerateDatasetStream();

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  // Derive the OutputStudio result directly from the streaming state — no effect needed.
  const result = useMemo(() => {
    if (stream.state.status === "done" && stream.state.samples.length > 0) {
      return {
        success: true as const,
        format_type: stream.state.formatType || "alpaca",
        data: stream.state.samples,
        message: `Generated ${stream.state.samples.length} samples in ${stream.state.formatType} format`,
      };
    }
    if (stream.state.status === "error" && stream.state.errorMessage) {
      return {
        success: false as const,
        format_type: stream.state.formatType || "alpaca",
        data: [] as Record<string, unknown>[],
        message: stream.state.errorMessage,
      };
    }
    return null;
  }, [
    stream.state.status,
    stream.state.samples,
    stream.state.errorMessage,
    stream.state.formatType,
  ]);

  const handleGenerateStart = () => {
    // Reset handled by the streaming hook; nothing to do here.
  };

  const handleClearLogs = () => {
    stream.reset();
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col transition-colors duration-200">
      <Navbar />

      <main className="max-w-7xl mx-auto px-l py-l space-y-8 flex-1 w-full">
        <div className="grid lg:grid-cols-2 gap-8 items-start">
          <InputWorkbench
            onGenerateStart={handleGenerateStart}
            stream={stream}
          />
          <OutputStudio result={result} loading={stream.isPending} />
        </div>

        {(stream.state.status !== "idle" || stream.state.logs.length > 0) && (
          <div className="h-[420px]">
            <StreamPanel state={stream.state} onClearLogs={handleClearLogs} />
          </div>
        )}
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