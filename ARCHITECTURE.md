# Conductor — Architecture

> This is a plain first draft for you to rewrite in your own voice.

## What Conductor does

Conductor turns a plain-English instruction into a working browser automation.
A LangGraph agent plans an ordered list of steps, executes each step by calling
a tool over the Model Context Protocol (MCP), and re-plans when a step fails —
explaining what it did at every stage. Because the actions are exposed by a
standards-based MCP server, the automation backend (Playwright today) can be
swapped without touching the agent.

## The LangGraph state machine

The orchestration is a real `StateGraph` (see [graph.py](graph.py)) with one
typed state object ([state.py](state.py)) flowing through it. There is **no
hand-written loop** — LangGraph itself drives the plan/execute/replan cycle.

The nodes:

- **plan** ([nodes/plan_node.py](nodes/plan_node.py)) — asks the LLM, given the
  goal and the live list of MCP tools, for an ordered list of steps that use
  only allow-listed tools.
- **execute** ([nodes/execute_node.py](nodes/execute_node.py)) — runs the step
  at the current index by calling its MCP tool, and records the outcome.
- **check** ([nodes/check_node.py](nodes/check_node.py)) — a **pure routing
  function** wired as a *conditional edge*. It only reads state and returns the
  name of the next branch; it never mutates state.
- **replan** ([nodes/replan_node.py](nodes/replan_node.py)) — after a failure,
  asks the LLM for a revised plan for the remaining work, keeping the steps that
  already succeeded.
- **finalize** ([graph.py](graph.py)) — assembles the immutable `RunResult`.

How conditional routing drives the loop:

```
        ┌───────────────────────────── execute ◀─────────────┐
        │                                  │                  │
START ─▶ plan ─▶ execute ─▶ (check_node?) ─┤                  │
                                           ├─ "execute" ──────┘  (next step)
                                           ├─ "replan" ──▶ replan ─▶ execute
                                           └─ "finish" ──▶ finalize ─▶ END
```

`check_node` decides each tick:

- last step **failed** and replans remain → `replan`
- last step **failed** and replans exhausted → `finish` (graceful stop)
- all steps done → `finish`
- otherwise → `execute` the next step

State is a `TypedDict` with an `Annotated` reducer on `results`, so each
`execute` appends its `StepResult` instead of overwriting the list. A
`MAX_REPLANS` counter caps the failure-recovery loop.

## The MCP layer

The agent's only way to act on the world is by calling MCP tools. This is a
deliberate boundary:

- **Server** ([mcp_server/server.py](mcp_server/server.py)) — a real MCP server
  built with the official FastMCP SDK. It registers exactly one tool per
  allow-listed action, each with a name, description, and typed arguments so an
  LLM can discover and call it correctly.
- **Backend** ([mcp_server/playwright_backend.py](mcp_server/playwright_backend.py))
  — the *only* file that imports Playwright. Each MCP tool delegates to one
  small backend method.
- **Client** ([mcp_client.py](mcp_client.py)) — spawns the server as a stdio
  subprocess, lists the tools, and forwards calls. The LangGraph nodes receive
  this client through the runnable config.

**Why expose actions over MCP?** It makes the action layer a swappable,
standards-based component. The agent depends only on the MCP *contract* (tool
names, descriptions, schemas), not on Playwright. To run the same agent against
a different automation engine, you would:

1. Write a new MCP server (e.g. a **UiPath** or **Selenium** MCP server) that
   exposes the same tool names and schemas.
2. Point the MCP client at that server.

Nothing in the graph, nodes, or prompts changes — the planner still discovers
tools over MCP and calls them the same way.

## Safety model

- **Fixed allow-list.** The server exposes only `navigate`, `click`,
  `type_text`, `read_text`, `screenshot`, `save_to_file`. The planner is given
  only these tools, and any step referencing a non-allow-listed tool is dropped
  before execution.
- **No arbitrary execution.** There is no shell or code-execution tool.
- **Output sandbox.** `screenshot` and `save_to_file` resolve paths inside
  `./output` and reject anything that escapes it (path traversal).
- **Audit logging.** Every step is logged before it executes — in the executor
  and again inside the MCP server — so a human can review what was done.

## Known limitations

- The planner relies on CSS selectors it infers from the goal; brittle or
  JavaScript-heavy pages may need a replan or fail gracefully.
- One browser page per run; no tabs, downloads, file uploads, or auth flows.
- Re-planning is capped (`MAX_REPLANS`) and does not learn across runs.
- No human-in-the-loop approval before a step runs (only logging).
- The free Gemini tier is rate-limited; large plans may hit backoff.

## What I'd build next

- A human-approval interrupt before risky steps (LangGraph supports interrupts).
- Richer tools (wait-for-selector, extract-table, multi-tab) behind the same
  MCP contract.
- A second MCP backend (Selenium or a desktop/UiPath server) to demonstrate the
  swap in practice.
- Persistent run history and replay using a LangGraph checkpointer.
- Vision-assisted element location to reduce selector brittleness.
