# Lit2Lab

> **Convert any research paper PDF into a validated, executable Jupyter Notebook — automatically.**

Lit2Lab is an end-to-end AI agent that reads a research paper, searches the web for real implementations, asks you how you want to handle data, generates a complete Python script, executes it, auto-fixes any errors, and delivers a clean `.ipynb` notebook — all from a single command.

---

## Demo Flow

```
python main.py paper.pdf
```

```
╔══════════════════════════════════════════╗
║   Research Paper → Notebook Agent    ║
╠══════════════════════════════════════════╣
║  PDF:    Attention_Is_All_You_Need.pdf  ║
║  Model:  qwen3-coder:480b-cloud         ║
╚══════════════════════════════════════════╝

[1/6] Parsing PDF...           39,742 chars extracted
[2/6] Extracting paper info... Attention Is All You Need

Paper Summary
┌──────────────┬─────────────────────────────────────────┐
│ Title        │ Attention Is All You Need               │
│ Objective    │ Transformer architecture for seq2seq    │
│ Methods      │ Multi-head attention, positional enc... │
│ Dependencies │ torch, numpy, matplotlib                │
│ Datasets     │ WMT 2014 EN-DE (~4.5GB)                 │
└──────────────┴─────────────────────────────────────────┘

Web Search — Planned queries:
  1. Attention Is All You Need github implementation
  2. Transformer pytorch implementation tutorial
  3. WMT 2014 dataset download python
→ Proceed? [Y/n]

Dataset Selection
  [1] Original dataset     — real data, best results
  [2] Sample / subset      — small portion, faster
  [3] Dummy / synthetic    — fake data, just test code  ← default
  [4] Custom path          — I have it locally
→ Your choice [3]:

Execution Review
  CPU: Intel Core i9  |  RAM: 32GB  |  GPU: RTX 4090
  [Script preview shown here...]
  [1] Run it now  [2] Change timeout  [3] Skip
→ Your choice [1]:

Script executed in 2m 14s

╔══════════════════════════════════════════╗
║            SESSION COMPLETE             ║
╠══════════════════════════════════════════╣
║  Paper:       Attention Is All You Need ║
║  Status:      Clean (0 errors)       ║
║  Dataset:     dummy                     ║
║  Fix rounds:  1                         ║
║  Time:        3m 42s                    ║
║  Memory:      4 entries, 2 successful   ║
║  Script:      outputs/Attention...py    ║
║  Notebook:    outputs/Attention...ipynb ║
╚══════════════════════════════════════════╝
```

---

## Features

### Intelligent Agent Pipeline
- Parses any research paper PDF and extracts title, methods, datasets, dependencies, hyperparameters, and evaluation metrics
- Uses LangGraph for structured, stateful agent orchestration
- Runs on local LLMs via Ollama — no OpenAI required

### Web Search Integration
- Searches for official GitHub implementations, dataset links, and known issues
- Uses **Tavily** (best quality) with **DuckDuckGo** as free fallback
- User can add custom search terms or skip entirely

### Human-in-the-Loop (5 Checkpoints)

| Checkpoint | When | What It Asks |
|---|---|---|
| **1. Web Search** | After extraction | Approve queries, add custom terms |
| **2. Dataset** | Before generation | Original / Sample / Dummy / Custom path |
| **3. Execution** | Before running | Review script, check hardware, set timeout |
| **4. Post Execution** | After success | Finish / Re-run / Tweak hyperparameters |
| **5. Error Review** | On failure | Auto-fix / Abort |

### Self-Healing Fix Loop
- Executes the generated script and captures full tracebacks
- Sends errors back to LLM with full context for fixing
- Retries up to `MAX_FIX_ATTEMPTS` times (default: 3)
- User reviews each error and chooses to auto-fix or abort

### Error Memory System
- Stores every `(error → fix)` pair to `memory/error_memory.json`
- Before each fix attempt, retrieves the top-5 most similar past fixes
- Injects them as few-shot examples into the fix prompt
- **Agent gets smarter with every paper it processes**

### Hybrid .py → .ipynb Approach
- Agent internally works with `.py` files (clean tracebacks, fast execution)
- Converts to `.ipynb` at the end using smart section detection
- Preserves full indentation and code structure
- User gets both formats in `outputs/`

### Auto Environment Management
- Creates a fresh `.agent_venv` using `uv` on first run
- Bootstraps `pip` inside the venv (uv does not bundle it by default)
- Registers the venv as a Jupyter kernel automatically
- Falls back to `stdlib venv` → `system python3` if uv is unavailable

---

## Directory Structure

```
Lit2Lab/
│
├── main.py                              # Entry point
├── requirements.txt                     # Python dependencies
├── .env                                 # API keys and config
│
├── agent/
│   ├── state.py                         # AgentState — shared state across all nodes
│   ├── graph.py                         # LangGraph graph — nodes + edges + routing
│   └── nodes/
│       ├── parse_paper.py               # PDF → raw text (PyMuPDF)
│       ├── extract_info.py              # LLM extracts structured paper info
│       ├── hitl_web_search.py           # HITL 1 — web search approval
│       ├── web_search.py                # Tavily/DuckDuckGo queries
│       ├── hitl_dataset.py              # HITL 2 — dataset choice
│       ├── generate_script.py           # LLM generates .py script
│       ├── hitl_execution.py            # HITL 3 — execution approval
│       ├── execute_script.py            # Runs script, captures errors
│       ├── hitl_error_review.py         # HITL 5 — error review on failure
│       ├── fix_script.py                # LLM fixes errors (with memory)
│       ├── hitl_post_execution.py       # HITL 4 — post execution options
│       └── save_output.py               # Saves .py + .ipynb + error report
│
├── tools/
│   ├── pdf_parser.py                    # PyMuPDF wrapper
│   ├── script_builder.py                # Save/load .py files
│   ├── script_runner.py                 # Run script, capture output
│   ├── py_to_notebook.py                # Convert .py → .ipynb
│   ├── kernel_detector.py               # uv venv setup + kernel registration
│   ├── error_memory.py                  # Error memory store + retrieval
│   ├── tavily_search.py                 # Web search (Tavily + DuckDuckGo)
│   ├── hardware_check.py                # CPU/RAM/GPU detection
│   └── rich_ui.py                       # All Rich terminal UI helpers
│
├── prompts/
│   ├── extract_info.txt                 # Prompt: paper → structured JSON
│   ├── generate_script.txt              # Prompt: info + web + dataset → .py
│   └── fix_script.txt                   # Prompt: error + memory → fixed .py
│
├── outputs/                             # Generated scripts and notebooks land here
│   ├── paper_title.py
│   ├── paper_title.ipynb
│   └── paper_title_error_report.txt     # Only created if errors remain
│
└── memory/
    └── error_memory.json                # Persistent fix memory (auto-created)
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/princ0301/Lit2Lab
cd Lit2Lab
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull qwen3-coder:480b-cloud
ollama serve
```

### 4. Configure environment
Copy `.env` and fill in your keys:
```env
# LLM Settings
LLM_MODEL=qwen3-coder:480b-cloud
LLM_TEMPERATURE=0

# Agent Settings
MAX_FIX_ATTEMPTS=3

# Web Search — get free key at https://tavily.com
# Optional: DuckDuckGo is used automatically if not set
TAVILY_API_KEY=your_tavily_api_key_here

# Execution
EXECUTION_TIMEOUT=300
```

---

## Usage

### Basic
```bash
python main.py path/to/paper.pdf
```

### With custom fix attempts
```bash
python main.py path/to/paper.pdf --max-fix-attempts 5
```

### Examples
```bash
python main.py papers/attention_is_all_you_need.pdf
python main.py papers/ResNet.pdf --max-fix-attempts 5
python main.py "D:/Papers/BERT.pdf"
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen3-coder:480b-cloud` | Ollama model to use |
| `LLM_TEMPERATURE` | `0` | LLM temperature (0 = deterministic) |
| `MAX_FIX_ATTEMPTS` | `3` | Max auto-fix iterations per paper |
| `TAVILY_API_KEY` | *(optional)* | Falls back to DuckDuckGo if not set |
| `EXECUTION_TIMEOUT` | `300` | Script execution timeout in seconds |

---

## How Each Feature Works

### Agent Pipeline

```
parse_paper → extract_info → hitl_web_search → web_search → hitl_dataset
    → generate_script → hitl_execution → execute_script
         ├── success → hitl_post_execution → save_output
         └── failure → hitl_error_review
                ├── auto-fix → fix_script → execute_script (loop)
                └── abort   → save_output
```

### Error Memory

Every fix attempt is stored in `memory/error_memory.json`:
```json
{
  "error_type": "ModuleNotFoundError",
  "error_signature": "No module named 'sklearn'",
  "fix_summary": "Changed 'sklearn' to 'scikit-learn' in packages list",
  "was_successful": true,
  "paper_title": "Attention Is All You Need"
}
```

On the next paper, the top-5 most similar past fixes are retrieved by token overlap and injected into the fix prompt as few-shot examples — so the agent never makes the same mistake twice.

### Dataset Choices

| Choice | What happens |
|---|---|
| **Original** | Generates download code for the real dataset with graceful fallback |
| **Sample** | Downloads a small subset (max 1000 rows / 10%) of the original |
| **Dummy** | No downloads — generates synthetic data using `numpy` / `sklearn` |
| **Custom** | Loads from your local path, falls back to synthetic if not found |

### Virtual Environment

On first run:
1. Creates `.agent_venv` using `uv venv` (falls back to `python -m venv`)
2. Bootstraps `pip` via `uv pip install pip`
3. Installs `ipykernel` into the venv
4. Registers it as a Jupyter kernel named `agent_venv`

On subsequent runs, the existing venv is reused instantly.

### .py → .ipynb Conversion

The agent uses `# ##` section comments to split the script into notebook cells:

```python
# ## 1. Install Dependencies    ← markdown cell
import subprocess, sys          ← code cell

# ## 2. Imports                 ← markdown cell
import numpy as np              ← code cell
```

Indented comments inside functions stay in their code cells — they never break the structure.

---

## Output Files

| File | Description |
|---|---|
| `outputs/<title>.py` | Clean, validated Python script |
| `outputs/<title>.ipynb` | Jupyter notebook converted from .py |
| `outputs/<title>_error_report.txt` | Only if errors remain after all fix attempts |
| `memory/error_memory.json` | Persistent memory — grows with every paper |
| `requirements_notebook.txt` | Packages installed for this paper |

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM interface | LangChain + Ollama |
| LLM model | qwen3-coder:480b-cloud |
| PDF parsing | PyMuPDF |
| Script execution | subprocess |
| Notebook creation | nbformat |
| Web search | Tavily / DuckDuckGo |
| Terminal UI | Rich |
| Virtual env | uv |

---

## Troubleshooting

**`No module named pip` in venv**
```bash
rm -rf .agent_venv
python main.py paper.pdf
```

**Ollama not running**
```bash
ollama serve
# then in a new terminal:
python main.py paper.pdf
```

**Script times out**

At the execution checkpoint choose `[2]` to increase timeout, or set `EXECUTION_TIMEOUT=600` in `.env`.

**Web search not working**

Add `TAVILY_API_KEY` to `.env` for best results. Without it, DuckDuckGo is used automatically.

**LLM returns empty response**

Check that `ollama serve` is running and the model is fully pulled with `ollama pull qwen3-coder:480b-cloud`.

---

## Roadmap

- [ ] FastAPI wrapper — expose as REST API endpoint
- [ ] Batch processing — process a folder of PDFs at once
- [ ] Few-shot memory — learn good code patterns from successful runs
- [ ] Paper context in fix loop — smarter fixes using original paper info
- [ ] `--memory-stats` CLI — view what the agent has learned so far
- [ ] Notebook quality checker — auto-clean empty cells and broken markdown

---
 