"use client";

import { useEffect, useMemo } from "react";
import { useWorkbenchStore } from "@/store/useWorkbenchStore";
import Navbar from "@/components/Navbar";
import InputWorkbench from "@/components/InputWorkbench";
import OutputStudio from "@/components/OutputStudio";
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
    // Fresh start
  };

  const handleClearLogs = () => {
    stream.reset();
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col transition-colors duration-200">
      <Navbar />

      <main className="max-w-[1600px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 flex flex-col">
        <div className="grid lg:grid-cols-12 gap-6 flex-1">
          <div className="lg:col-span-5 xl:col-span-4 flex flex-col">
            <InputWorkbench
              onGenerateStart={handleGenerateStart}
              stream={stream}
            />
          </div>
          <div className="lg:col-span-7 xl:col-span-8 flex flex-col">
            <OutputStudio
              result={result}
              loading={stream.isPending}
              stream={stream}
              onClearLogs={handleClearLogs}
            />
          </div>
        </div>
      </main>

      <footer className="border-t border-border bg-card py-4 mt-8">
        <div className="max-w-[1600px] w-full mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-secondary font-bold">
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