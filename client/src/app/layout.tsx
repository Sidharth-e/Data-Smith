import type { Metadata } from "next";
import "./globals.css";
import QueryProvider from "@/providers/QueryProvider";

export const metadata: Metadata = {
  title: "Data Smith | LLM Fine-Tuning Dataset Generator Studio",
  description: "Synthesize high-quality Alpaca, Chat, and Completion datasets for model fine-tuning with LangChain and Ollama.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-background text-foreground min-h-screen">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
