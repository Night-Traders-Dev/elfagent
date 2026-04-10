# ElfAgentPlus Architecture

## Routing hierarchy

router -> ultra-light helpers -> medium reasoning models -> main model

## Included techniques

### AutoResearch-inspired workflow
This project adopts the *workflow philosophy* of AutoResearch by separating planning, experimentation, evaluation, and iterative improvement loops. It does **not** implement Karpathy's training harness itself.

### TurboQuant
TurboQuant is not directly implemented in Python here. It is referenced as a deployment/runtime optimization target for lower-memory inference stacks and future model-serving upgrades.

### Engram / Engrams
Engram-style memory is not implemented as a model architecture here. Instead, this project uses structured handoff packets and persistent chat memory as an application-level approximation for selective memory access.

### Direct Preference Optimization (DPO)
DPO is a post-training alignment method. This codebase does not train or fine-tune models, so DPO is documented as a future tuning path for the router/helper models if you build a preference dataset.

## Practical interpretation

This repository uses these ideas as follows:
- AutoResearch -> iterative agent improvement workflow and modular experimentation
- TurboQuant -> future model serving / KV-cache compression target
- Engram -> memory-oriented selective retrieval mindset
- DPO -> future post-training alignment path for specialized helpers


## 2026-04-10 latency fixes
- Switched default MAIN_MODEL to qwen2.5-coder:7b for faster general coding turns.
- Added fast path for simple code generation requests.
- Removed artificial final-answer replay delay.
- Disabled console OpenTelemetry exporters unless ENABLE_CONSOLE_OTEL=1.
- Reduced noisy AgentStream/StopEvent logging.
- Added final answer truncation guard.

- Model stack aligned to use case: FIM/autocomplete=qwen2.5-coder:7b, chat/debug/review=qwen3.5:9b, complex refactor=qwen2.5-coder:14b.
