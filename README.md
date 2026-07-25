<div align="center">
  <img src="client/public/icon.svg" alt="Data Smith Logo - Dataset Generation Tool for Fine-Tuning LLMs" width="120" height="120" />
  <h1>Data Smith</h1>
  <p><strong>A powerful dataset generation tool for fine-tuning language models</strong></p>
  <p>Transform raw text into structured training data formats using LangChain agents powered by Ollama.</p>

  <p>
    <a href="https://github.com/sidharthe/Data-Smith/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/sidharthe/Data-Smith?style=for-the-badge&color=f04461" /></a>
    <a href="https://github.com/sidharthe/Data-Smith/forks"><img alt="GitHub Forks" src="https://img.shields.io/github/forks/sidharthe/Data-Smith?style=for-the-badge&color=blue" /></a>
    <a href="https://github.com/sidharthe/Data-Smith/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/sidharthe/Data-Smith?style=for-the-badge&color=orange" /></a>
    <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" /></a>
    <img alt="Built with Next.js" src="https://img.shields.io/badge/Next.js-16-black?style=for-the-badge&logo=next.js" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi" />
    <img alt="Ollama" src="https://img.shields.io/badge/Ollama-Local%20LLM-f04461?style=for-the-badge" />
  </p>
</div>

---

> **Data Smith** is an open-source dataset generation tool that converts raw text into fine-tuning-ready formats (Alpaca, Chat, Completion) using local LLMs via Ollama and LangChain agents — no API costs, fully private.

## Features

- **📄 File Upload** - Upload .txt files with your source content
- **✏️ Text Input** - Paste text directly for quick processing
- **🎯 3 Output Formats**:
  - **Alpaca** - instruction/input/output pairs
  - **Chat** - conversational messages with roles
  - **Completion** - raw text continuations
- **⚡ Local LLM** - Powered by Ollama (no API costs)
- **📥 JSON Export** - Download generated datasets

## Architecture

```
Data Smith/
├── client/          # Next.js frontend (React 19, Tailwind 4)
│   └── src/app/
│       └── page.tsx # Main UI with file upload & preview
└── server/          # Python backend
    ├── main.py      # FastAPI server
    ├── agent.py     # LangChain agent with Ollama
    ├── formats.py   # Pydantic output models
    └── requirements.txt
```

## Prerequisites

- **Node.js 18+** and **pnpm**
- **Python 3.10+**
- **Ollama** with a model installed (e.g., `llama3.2`, `mistral`)

## Quick Start

### 1. Start Ollama

```bash
ollama serve
# In another terminal, pull a model if needed:
ollama pull llama3.2
```

### 2. Start Python Server

```bash
cd server
pip install -r requirements.txt
python main.py
# Server runs at http://localhost:8000
```

### 3. Start Next.js Client

```bash
cd client
pnpm install
pnpm dev
# Open http://localhost:3000
```

## Usage

1. **Upload** a `.txt` file or paste text directly
2. **Select** your desired output format (Alpaca/Chat/Completion)
3. **Adjust** the number of samples (1-20)
4. **Click** Generate Dataset
5. **Download** the JSON output

## Output Format Examples

### Alpaca

```json
{
  "instruction": "Summarize the key points",
  "input": "The document discusses...",
  "output": "The main points are..."
}
```

### Chat

```json
{
  "messages": [
    { "role": "system", "content": "You are an expert..." },
    { "role": "user", "content": "What is..." },
    { "role": "assistant", "content": "It is..." }
  ]
}
```

### Completion

```json
{
  "text": "The concept of neural networks involves..."
}
```

## API Endpoints

| Endpoint             | Method | Description               |
| -------------------- | ------ | ------------------------- |
| `/api/health`        | GET    | Health check              |
| `/api/generate`      | POST   | Generate from file upload |
| `/api/generate-text` | POST   | Generate from text input  |

## Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS 4
- **Backend**: FastAPI, LangChain, Ollama
- **LLM**: Local models via Ollama
- **Models**: Llama3.2, Mistral
- **LangChain**: v0.0.266
- **Output Formats**: Alpaca, Chat, Completion

## License

MIT
