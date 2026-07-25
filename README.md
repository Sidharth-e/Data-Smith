<div align="center">
  <img src="client/public/icon.svg" alt="Data Smith Logo - Dataset Generation Tool for Fine-Tuning LLMs" width="120" height="120" />
  <h1>Data Smith</h1>
  <p><strong>A powerful dataset generation tool for fine-tuning language models</strong></p>
  <p>Transform raw text into structured training data formats using LangChain agents powered by Ollama and Gemini.</p>

  <p>
    <a href="https://github.com/sidharthe/Data-Smith/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/sidharthe/Data-Smith?style=for-the-badge&color=f04461" /></a>
    <a href="https://github.com/sidharthe/Data-Smith/forks"><img alt="GitHub Forks" src="https://img.shields.io/github/forks/sidharthe/Data-Smith?style=for-the-badge&color=blue" /></a>
    <a href="https://github.com/sidharthe/Data-Smith/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/sidharthe/Data-Smith?style=for-the-badge&color=orange" /></a>
    <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" /></a>
    <img alt="Built with Next.js" src="https://img.shields.io/badge/Next.js-16.0.8-black?style=for-the-badge&logo=next.js" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi" />
    <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-0.2.0-blue?style=for-the-badge" />
  </p>
</div>

---

> **Data Smith** is an open-source dataset generation tool that converts raw text into fine-tuning-ready formats (Alpaca, ChatML, ShareGPT, DPO, Completion) using LLMs. It features deep research capabilities, real-time streaming, and supports multiple providers including local Ollama and Google Gemini.

## Features

- **🧠 Deep Research Agent** - Multi-agent LangGraph pipeline (planner, researcher, writer) that searches the web to synthesize comprehensive source documents.
- **⚡ Real-time Streaming** - Watch datasets generate in real-time with granular progress and thinking state updates.
- **🔌 Multi-Provider Support** - Native support for Local Ollama, Ollama Cloud, and Google Gemini via a simple `config.toml`.
- **🎯 5 Output Formats**:
  - **Alpaca** - Instruction / input / output pairs.
  - **ChatML** - Conversational messages with roles.
  - **ShareGPT** - Multi-turn conversations for conversational AI.
  - **DPO** - Direct Preference Optimization pairs (prompt, chosen, rejected).
  - **Completion** - Raw text continuations.
- **🎨 Modern Workbench UI** - Includes Dark Mode, Command Palette, Vertical Split View layout, subtle grid aesthetics, and multiple view modes (JSON, Table, Cards).
- **♿ 100% Accessible** - Fully WCAG compliant with optimized contrast, semantic HTML, and complete screen reader support.
- **📥 JSON Export** - Download your generated datasets instantly.

## Architecture

```
Data Smith/
├── client/          # Next.js frontend (React 19, Tailwind 4)
│   └── src/
│       ├── app/         # Next.js App Router
│       ├── components/  # React UI Components (OutputStudio, StreamPanel, etc.)
│       └── store/       # Zustand State Management
└── server/          # Python backend (FastAPI)
    ├── main.py      # API Endpoints (Generation & Streaming)
    ├── agent.py     # LangChain Agent for Dataset Generation
    ├── research.py  # LangGraph Agent for Deep Research
    ├── model_factory.py # LLM Provider Factory
    └── config.toml  # Centralized configuration
```

## Prerequisites

- **Node.js 18+** and **pnpm**
- **Python 3.10+** (using `uv` or `pip`)
- **Ollama** (optional, for local models) or **Gemini API Key**

## Quick Start

### 1. Configure the Environment

Create your local environment file in the `server` directory:
```bash
cd server
cp .env.example .env.local
```
*(Edit `.env.local` to add your `GEMINI_API_KEY` or `OLLAMA_CLOUD_API_KEY` if not using local Ollama)*

You can also edit `server/config.toml` to change the default LLM provider from `ollama` to `gemini` or `ollama_cloud`.

### 2. Start Python Server

Using `uv` (recommended):
```bash
cd server
uv run main.py
# Server runs at http://localhost:8000
```

Or using standard `pip`:
```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 3. Start Next.js Client

```bash
cd client
pnpm install
pnpm dev
# Open http://localhost:3000
```

## Usage

1. **Input Data**: Upload a `.txt` file, paste text directly, or use the **Research** feature to synthesize a document from the web.
2. **Configure**: Select your desired output format (Alpaca / ChatML / ShareGPT / DPO / Completion) and adjust the number of samples.
3. **Generate**: Click Generate to stream results in real-time.
4. **Export**: View the output in JSON, Table, or Card formats and download the generated dataset.

## Output Format Examples

### Alpaca
```json
{
  "instruction": "Summarize the key points",
  "input": "The document discusses...",
  "output": "The main points are..."
}
```

### ChatML
```json
{
  "messages": [
    { "role": "system", "content": "You are an expert..." },
    { "role": "user", "content": "What is..." },
    { "role": "assistant", "content": "It is..." }
  ]
}
```

### ShareGPT
```json
{
  "conversations": [
    { "from": "system", "value": "You are an expert..." },
    { "from": "human", "value": "What is..." },
    { "from": "gpt", "value": "It is..." }
  ]
}
```

### DPO
```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I think it's Berlin."
}
```

### Completion
```json
{
  "text": "The concept of neural networks involves..."
}
```

## API Endpoints

| Endpoint                      | Method | Description                                      |
| ----------------------------- | ------ | ------------------------------------------------ |
| `/api/health`                 | GET    | Health check                                     |
| `/api/generate`               | POST   | Generate from file upload                        |
| `/api/generate-text`          | POST   | Generate from text input                         |
| `/api/generate-stream`        | POST   | Stream generation from file upload (SSE)         |
| `/api/generate-text-stream`   | POST   | Stream generation from text input (SSE)          |
| `/api/research-stream`        | POST   | Stream deep research document synthesis (SSE)    |

## Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS 4, Zustand, TanStack Query
- **Backend**: FastAPI, LangChain, LangGraph
- **LLM Support**: Ollama (Local/Cloud), Google Gemini
- **Formats**: Alpaca, ChatML, ShareGPT, DPO, Completion

## License

MIT
