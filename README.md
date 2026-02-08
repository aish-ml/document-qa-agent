# Document Q&A AI Agent

An enterprise-ready AI agent that ingests PDF documents and enables intelligent question-answering using LLM APIs. Built with Python, LangChain, and Chroma vector store.

## Features

- **Multi-modal PDF Extraction** — Extracts text, tables, images, graphs, charts, equations, and structured sections (titles, abstracts, methodology, results, references) from multiple PDFs
- **NLP-Powered Q&A Interface** — Natural language queries with three core capabilities:
  - **Direct content lookup** — "What is the conclusion of Paper X?"
  - **Summarization** — "Summarize the methodology of Paper C"
  - **Evaluation extraction** — "What are the accuracy and F1-score in Paper D?"
- **Multi-LLM Support** — Configurable backends: OpenAI (GPT-4o), Google Gemini, Ollama (local open-source models like Llama 3)
- **Arxiv Integration** (Bonus) — Search and download papers from Arxiv directly into the knowledge base
- **Auto-Ingest on Startup** — PDFs in `documents/` are automatically ingested when the agent starts
- **Arxiv Auto-Ingest** — Search Arxiv and download+ingest top N papers in one step
- **Streamlit Web UI** — Beautiful chat interface with PDF upload, Arxiv search, background ingestion, and runtime LLM configuration
- **Runtime LLM Configuration** — Switch LLM provider, model, and API key from the UI without restarting
- **Background Ingestion** — Non-blocking worker thread processes PDFs while you keep chatting
- **Enterprise Features** — Response caching, rate limiting, conversation memory, context window management, retry logic, thread-safe vector store

## Architecture

```
document-qa-agent/
├── main.py                         # CLI entry point & interactive shell
├── config.py                       # Centralized configuration from .env
├── requirements.txt                # Python dependencies
│
├── ingestion/
│   ├── pdf_extractor.py            # Multi-modal PDF extraction (text/tables/images)
│   └── chunker.py                  # Overlap-aware text chunking with metadata
│
├── knowledge_base/
│   └── vector_store.py             # Chroma vector store with embedding support
│
├── agent/
│   ├── llm_provider.py             # Multi-LLM factory (OpenAI/Gemini/Ollama)
│   ├── tools.py                    # LangChain function-calling tools
│   └── qa_agent.py                 # Main agent with conversation memory
│
├── arxiv_integration/
│   └── arxiv_client.py             # Arxiv search, fetch, and download
│
├── utils/
│   ├── helpers.py                  # Logging & formatting utilities
│   └── ingest_worker.py            # Background ingestion worker & queue
│
├── ui/
│   └── streamlit_app.py            # Streamlit web chat interface
│
├── documents/                      # Place PDF files here for ingestion
└── .env.example                    # Environment variable template
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- An API key for at least one LLM provider (OpenAI, Google Gemini, or local Ollama)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd document-qa-agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key(s):

```env
# Choose your provider
LLM_PROVIDER=gemini

# Set your API key
GEMINI_API_KEY=your-api-key-here

# Or for OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your-api-key-here

# Or for local Ollama (no API key needed)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3
```

### 5. Add PDF Documents

Place your PDF files in the `documents/` directory.

## Usage

### Web UI (Recommended)

The easiest way to use the agent:

```bash
streamlit run ui/streamlit_app.py
```

This opens a browser with:
- **Chat interface** — Ask questions, get answers with source citations
- **PDF upload** — Drag & drop PDFs into the sidebar
- **LLM settings** — Switch provider (OpenAI / Gemini / Ollama), model, and API key at runtime without restarting
- **Auto-ingest** — PDFs in `documents/` are ingested automatically on first load
- **Arxiv search** — Search, browse results, and click "Download & Ingest All" to load papers
- **Background ingestion** — All ingestion runs non-blocking; keep chatting while it works
- **Knowledge base view** — See loaded documents, chunk counts, agent stats

### Interactive CLI Mode

```bash
python main.py
```

This starts an interactive shell where you can:

- `/ingest` — Ingest all PDFs from `documents/` folder
- `/ingest path/to/paper.pdf` — Ingest a specific PDF
- `/sources` — List loaded documents
- `/stats` — Show agent statistics
- `/arxiv <query>` — Search Arxiv for papers
- `/arxiv-ingest <query>` — Search Arxiv & auto-ingest top N papers
- `/download <arxiv_id>` — Download & ingest a specific Arxiv paper
- `/clear` — Clear conversation history
- `/verbose` — Toggle verbose mode (shows tool calls)
- `/quit` — Exit

Then ask questions naturally:

```
You: What is the conclusion of the transformer paper?
You: Summarize the methodology used in Paper B
You: What accuracy and F1-score are reported?
You: Compare the results across all papers
```

### Command-Line Mode

```bash
# Ingest documents
python main.py ingest

# Ask a single question
python main.py ask "What are the key findings?"

# Use specific provider
python main.py --provider openai --model gpt-4o

# Auto-ingest on startup (also enabled by default via .env)
python main.py --auto-ingest

# Disable auto-ingest
python main.py --no-auto-ingest

# Verbose logging
python main.py -v
```

## Enterprise Features

| Feature | Description |
|---------|-------------|
| **Context Window Management** | Top-k retrieval limits context to most relevant chunks |
| **Response Caching** | In-memory LRU cache avoids duplicate API calls |
| **Rate Limiting** | Token-bucket rate limiter prevents API throttling |
| **Retry Logic** | Exponential backoff on transient failures |
| **Conversation Memory** | Maintains multi-turn context (last 10 exchanges) |
| **Structured Metadata** | Source attribution with document name, page, section |
| **Multi-modal Extraction** | Tables as Markdown, images saved + base64 encoded |

## Example Queries

| Query Type | Example |
|------------|---------|
| Direct Lookup | "What is the abstract of the attention paper?" |
| Summarization | "Summarize the methodology of Paper C" |
| Metrics Extraction | "What accuracy and F1-score are reported in Paper D?" |
| Cross-document | "Compare the evaluation approaches of all papers" |
| Arxiv Search | "Find papers about vision transformers for medical imaging" |

## Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-4o, GPT-4o-mini, etc. | Requires API key |
| **Google Gemini** | gemini-2.0-flash, gemini-pro | Requires API key |
| **Ollama** | Llama 3, Mistral, Phi-3, etc. | Free, runs locally |

## License

MIT
