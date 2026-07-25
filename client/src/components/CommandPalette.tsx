"use client";

import { useEffect } from "react";
import { useWorkbenchStore, FormatType, ViewMode } from "@/store/useWorkbenchStore";
import { Command, Search, Layers, Sliders, Moon, Sun, X } from "lucide-react";

export default function CommandPalette() {
  const {
    isCommandPaletteOpen,
    setCommandPaletteOpen,
    setFormatType,
    setNumSamples,
    setViewMode,
    toggleDarkMode,
    isDarkMode,
  } = useWorkbenchStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandPaletteOpen(!isCommandPaletteOpen);
      }
      if (e.key === "Escape" && isCommandPaletteOpen) {
        setCommandPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isCommandPaletteOpen, setCommandPaletteOpen]);

  if (!isCommandPaletteOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-start justify-center pt-20 p-m"
      onClick={() => setCommandPaletteOpen(false)}
    >
      <div
        className="w-full max-w-lg bg-card border-2 border-border rounded-2xl shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center px-m py-l border-b border-border gap-2 bg-muted/30">
          <Search className="w-4 h-4 text-primary shrink-0" />
          <input
            type="text"
            autoFocus
            placeholder="Type a command or format..."
            aria-label="Command palette input"
            className="w-full bg-transparent text-sm font-bold text-foreground placeholder-secondary focus:outline-none"
          />
          <button
            type="button"
            onClick={() => setCommandPaletteOpen(false)}
            aria-label="Close Command Palette"
            className="p-1 rounded-lg text-secondary hover:text-foreground hover:bg-muted"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-m space-y-4 max-h-[360px] overflow-auto text-xs">
          <div>
            <span className="text-secondary font-black uppercase tracking-wider block px-s pb-s">
              Output Format
            </span>
            <div className="space-y-1">
              {(["alpaca", "chat", "completion"] as FormatType[]).map((fmt) => (
                <button
                  key={fmt}
                  type="button"
                  onClick={() => {
                    setFormatType(fmt);
                    setCommandPaletteOpen(false);
                  }}
                  className="w-full text-left px-m py-s rounded-lg hover:bg-muted text-foreground flex items-center justify-between font-extrabold"
                >
                  <div className="flex items-center gap-2">
                    <Layers className="w-3.5 h-3.5 text-primary" />
                    <span className="capitalize">{fmt} Format</span>
                  </div>
                  <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-muted border border-border rounded text-secondary font-bold">
                    Format
                  </kbd>
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="text-secondary font-black uppercase tracking-wider block px-s pb-s">
              Sample Count Presets
            </span>
            <div className="grid grid-cols-3 gap-1">
              {[5, 20, 50, 100, 500, 1000].map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => {
                    setNumSamples(n);
                    setCommandPaletteOpen(false);
                  }}
                  className="px-m py-s rounded-lg hover:bg-muted text-foreground flex items-center justify-between font-extrabold border border-border"
                >
                  <div className="flex items-center gap-1">
                    <Sliders className="w-3.5 h-3.5 text-primary" />
                    <span>{n}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="text-secondary font-black uppercase tracking-wider block px-s pb-s">
              Output View Mode
            </span>
            <div className="space-y-1">
              {(["json", "table", "cards"] as ViewMode[]).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => {
                    setViewMode(v);
                    setCommandPaletteOpen(false);
                  }}
                  className="w-full text-left px-m py-s rounded-lg hover:bg-muted text-foreground flex items-center justify-between font-extrabold"
                >
                  <span className="capitalize">{v} View</span>
                  <span className="text-secondary text-[10px] font-bold">Switch Inspector</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="text-secondary font-black uppercase tracking-wider block px-s pb-s">
              Appearance
            </span>
            <button
              type="button"
              onClick={() => {
                toggleDarkMode();
                setCommandPaletteOpen(false);
              }}
              className="w-full text-left px-m py-s rounded-lg hover:bg-muted text-foreground flex items-center justify-between font-extrabold"
            >
              <div className="flex items-center gap-2">
                {isDarkMode ? <Sun className="w-3.5 h-3.5 text-warning" /> : <Moon className="w-3.5 h-3.5 text-primary" />}
                <span>Toggle Light/Dark Theme</span>
              </div>
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-muted border border-border rounded text-secondary font-bold">
                Theme
              </kbd>
            </button>
          </div>
        </div>

        <div className="border-t border-border px-m py-s bg-muted text-[10px] text-secondary font-bold flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Command className="w-3 h-3 text-primary" />
            <span>Data Smith Palette</span>
          </div>
          <span>Press ESC to close</span>
        </div>
      </div>
    </div>
  );
}
