"use client";

import { useWorkbenchStore } from "@/store/useWorkbenchStore";
import { useApiHealth } from "@/hooks/useGenerateDataset";
import Logo from "@/components/Logo";
import { Command, Moon, Sun, CheckCircle2, AlertCircle, Search } from "lucide-react";

export default function Navbar() {
  const { isDarkMode, toggleDarkMode, toggleCommandPalette, toggleResearchModal } =
    useWorkbenchStore();
  const { data: health, isError } = useApiHealth();

  return (
    <header className="border-b border-border bg-card sticky top-0 z-40 backdrop-blur-md shadow-2xs">
      <div className="max-w-[1600px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary text-white flex items-center justify-center shadow-xs">
            <Logo size={24} className="text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-black text-foreground tracking-tight">
                Data Smith
              </h1>
              <span className="bg-primary-light text-primary text-[10px] font-extrabold px-2 py-0.5 rounded-full border border-primary/30 uppercase tracking-widest">
                Studio
              </span>
            </div>
            <p className="text-xs text-secondary font-bold">
              LLM Dataset Synthesis & Workbench
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 px-m py-1.5 bg-muted border border-border rounded-xl text-xs font-bold">
            {isError ? (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-error" />
                <span className="text-error font-extrabold">API Disconnected</span>
              </>
            ) : health ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                <span className="text-foreground font-extrabold">{health.model || "Ollama"}</span>
                <span className="text-secondary">•</span>
                <span className="text-secondary font-bold capitalize">{health.provider || "Local"}</span>
              </>
            ) : (
              <span className="text-secondary font-bold">Connecting...</span>
            )}
          </div>

          <button
            type="button"
            onClick={toggleResearchModal}
            aria-label="Open Research modal"
            className="flex items-center gap-2 px-m h-medium rounded-xl border border-border bg-card hover:bg-muted text-xs font-bold text-foreground transition-all shadow-2xs cursor-pointer"
          >
            <Search className="w-3.5 h-3.5 text-primary" />
            <span className="hidden md:inline">Research</span>
          </button>

          <button
            type="button"
            onClick={toggleCommandPalette}
            aria-label="Open Command Palette"
            className="flex items-center gap-2 px-m h-medium rounded-xl border border-border bg-card hover:bg-muted text-xs font-bold text-foreground transition-all shadow-2xs cursor-pointer"
          >
            <Command className="w-3.5 h-3.5 text-primary" />
            <span className="hidden md:inline">Command Palette</span>
            <kbd className="hidden md:inline-block px-1.5 py-0.5 text-[10px] font-mono bg-muted border border-border rounded text-secondary font-extrabold">
              ⌘K
            </kbd>
          </button>

          <button
            type="button"
            onClick={toggleDarkMode}
            aria-label="Toggle theme mode"
            className="w-10 h-10 rounded-xl border border-border bg-card text-foreground hover:bg-muted flex items-center justify-center transition-all shadow-2xs cursor-pointer"
          >
            {isDarkMode ? <Sun className="w-4 h-4 text-warning" /> : <Moon className="w-4 h-4 text-primary" />}
          </button>
        </div>
      </div>
    </header>
  );
}
