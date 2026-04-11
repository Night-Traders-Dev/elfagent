# ElfAgentPlus Architecture

## Routing hierarchy

router -> ultra-light helpers -> medium reasoning models -> main model

## Included techniques

### AutoResearch-inspired workflow
This project now implements an application-level AutoResearch loop for web research. `WebReasoner` delegates to an iterative planner that:
- proposes an initial search plan
- runs multiple search/evaluation rounds
- measures query-term coverage
- refines follow-up searches when evidence is thin

### TurboQuant
TurboQuant is implemented here as a lightweight runtime compression layer for helper outputs and handoff packets. It trims snippets, limits evidence payloads, and quantizes structured context before the main model sees it.

### Engram / Engrams
Engram-style memory is implemented here as a selective application-memory store. Each turn can persist a compact "engram" gist, and new turns retrieve only the most relevant prior engrams instead of replaying broad chat history.

### Direct Preference Optimization (DPO)
DPO is a post-training alignment method. This codebase does not train or fine-tune models, so DPO is documented as a future tuning path for the router/helper models if you build a preference dataset.

## Practical interpretation

This repository uses these ideas as follows:
- AutoResearch -> iterative search / evaluate / refine helper workflow
- TurboQuant -> helper-context compression and prompt-budget control
- Engram -> persistent selective memory retrieval for relevant prior turns
- DPO -> future post-training alignment path for specialized helpers


## 2026-04-10 latency fixes
- Switched default MAIN_MODEL to qwen2.5-coder:7b for faster general coding turns.
- Added fast path for simple code generation requests.
- Removed artificial final-answer replay delay.
- Disabled console OpenTelemetry exporters unless ENABLE_CONSOLE_OTEL=1.
- Reduced noisy AgentStream/StopEvent logging.
- Added final answer truncation guard.

- Model stack aligned to use case: FIM/autocomplete=qwen2.5-coder:7b, chat/debug/review=qwen3.5:9b, complex refactor=qwen2.5-coder:14b.
