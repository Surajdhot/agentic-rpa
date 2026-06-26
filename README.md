# 🎼 Conductor

**An AI-powered RPA orchestrator.** Type a plain-English instruction — *"go to
this site, read the heading, and save it to a file"* — and a
[LangGraph](https://langchain-ai.github.io/langgraph/) agent plans the
automation, executes it by calling tools exposed over the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/), and re-plans
when a step fails. The MCP server wraps a [Playwright](https://playwright.dev/)
browser, so the action layer is a standards-based, swappable component.

Everything runs **locally on free tooling**: Google Gemini (free API tier) for
planning and Playwright for browser automation.

---

## How it works

```
plain English ─▶ LangGraph (plan ▸ execute ▸ check ▸ replan) ─▶ MCP client
                                                                    │
                                                          MCP (stdio) protocol
                                                                    │
                                                MCP server ─▶ Playwright (Chromium)
```

- **LangGraph** drives a real `StateGraph` with typed state and a conditional
  router — not a hand-rolled loop.
- **MCP** is the only way the agent acts on the world. The agent never imports
  Playwright; it discovers and calls MCP tools.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

---

## Setup (macOS / Linux)

```bash
# 1. Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv && source .venv/bin/activate
#    (or with uv:  uv venv --python 3.11 .venv && source .venv/bin/activate)

# 2. Install dependencies
pip install -r requirements.txt
#    (or:  uv pip install -r requirements.txt)

# 3. Install the Chromium browser used by Playwright (one-time)
playwright install chromium

# 4. Configure your free Google AI Studio key
cp .env.example .env
#    then edit .env and set GOOGLE_API_KEY=...   (get one at
#    https://aistudio.google.com/apikey)
```

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501), type an
instruction, and click **Run**. You'll see each step's outcome live, inline
screenshots, clearly-marked re-planning events, and a final summary.

## Run with Docker

```bash
cp .env.example .env   # set GOOGLE_API_KEY first
docker compose up --build
```

Then open http://localhost:8501. Files the agent saves appear in `./output`.

## Test

```bash
pytest tests/
```

The suite mocks the LLM and the MCP tools, so it never calls the Gemini API and
never opens a real browser.

---

## Configuration

| Variable         | Default            | Notes                                            |
| ---------------- | ------------------ | ------------------------------------------------ |
| `GOOGLE_API_KEY` | *(required)*       | Free Google AI Studio key.                       |
| `MODEL`          | `gemini-2.0-flash` | Any free Gemini Flash model (e.g. `gemini-2.5-flash`). |
| `LOG_LEVEL`      | `INFO`             | Standard Python logging level.                   |

Other limits (max steps, max replans, output directory) live in
[config.py](config.py).

> **Model note:** the default is `gemini-2.0-flash`, which is available on the
> free tier. If you prefer, set `MODEL=gemini-2.5-flash` in `.env`.

## Safety

The agent can only perform a fixed allow-list of actions, exposed over MCP:
`navigate`, `click`, `type_text`, `read_text`, `screenshot`, `save_to_file`.
There is no shell or arbitrary-code execution. File writes are sandboxed to
`./output` (path traversal is rejected), and every step is logged before it
runs for auditing.
