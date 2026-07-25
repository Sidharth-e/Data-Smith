import { create } from "zustand";

export type FormatType = "alpaca" | "chat" | "completion";
export type ViewMode = "json" | "table" | "cards";
export type InputMode = "file" | "text";

export type StudioTab = "dataset" | "stream" | "split";

interface WorkbenchState {
  inputMode: InputMode;
  formatType: FormatType;
  numSamples: number;
  file: File | null;
  textInput: string;
  viewMode: ViewMode;
  searchFilter: string;
  activeStudioTab: StudioTab;
  isCommandPaletteOpen: boolean;
  isDarkMode: boolean;

  setInputMode: (mode: InputMode) => void;
  setFormatType: (format: FormatType) => void;
  setNumSamples: (num: number) => void;
  setFile: (file: File | null) => void;
  setTextInput: (text: string) => void;
  setViewMode: (mode: ViewMode) => void;
  setSearchFilter: (query: string) => void;
  setActiveStudioTab: (tab: StudioTab) => void;
  setCommandPaletteOpen: (open: boolean) => void;
  toggleCommandPalette: () => void;
  setDarkMode: (dark: boolean) => void;
  toggleDarkMode: () => void;
  resetInputs: () => void;
}

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  inputMode: "file",
  formatType: "alpaca",
  numSamples: 5,
  file: null,
  textInput: "",
  viewMode: "json",
  searchFilter: "",
  activeStudioTab: "dataset",
  isCommandPaletteOpen: false,
  isDarkMode: false,

  setInputMode: (mode) => set({ inputMode: mode }),
  setFormatType: (format) => set({ formatType: format }),
  setNumSamples: (num) => set({ numSamples: Math.min(Math.max(num, 1), 1000) }),
  setFile: (file) => set({ file, inputMode: "file" }),
  setTextInput: (text) => set({ textInput: text }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setSearchFilter: (query) => set({ searchFilter: query }),
  setActiveStudioTab: (tab) => set({ activeStudioTab: tab }),
  setCommandPaletteOpen: (open) => set({ isCommandPaletteOpen: open }),
  toggleCommandPalette: () =>
    set((state) => ({ isCommandPaletteOpen: !state.isCommandPaletteOpen })),
  setDarkMode: (dark) => set({ isDarkMode: dark }),
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
  resetInputs: () =>
    set({
      file: null,
      textInput: "",
      numSamples: 5,
      searchFilter: "",
    }),
}));
