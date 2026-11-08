---
name: local-second-opinion
description: Ask a local model (via Proprioceptive Lens) for a second opinion, a private draft, or a cheap sub-task, and read how certain it was. Use when the user asks to check something with their local model, wants a private/offline answer, or wants to compare models.
---

# Local second opinion with Proprioceptive Lens

The `lens` MCP server talks to the user's local models (Ollama, or a Hugging Face model running in Lens deep mode).
Tools:

- `list_local_models`: which models are available. Prefer a chat/instruct model; avoid embedding or vision models.
- `ask_local_model`: send `prompt` (plus optional `model`, `max_tokens`, `use_memory`). You get the answer, its
  **certainty** (mean probability of the chosen words), the number of **uncertain words**, the **hesitations**
  (words chosen with low probability and what the model almost said instead) and, in deep mode, **Behaviour**
  readings from the model's internal state.
- `search_memory`: the user's private history of local-model conversations.

## How to read the result

- **Certainty is not correctness.** A model can be confidently wrong. In Lens's own example, a 4B model named the
  wrong 17th US president at 100% certainty. Never present a local answer as verified because its certainty is high.
- **Hesitations are where to look.** Words with low probability, especially names, numbers and dates, are
  where the model was choosing between options. If an alternative it almost said is plausible, say so.
- **Disagreement is information.** If the local answer differs from yours, show both, point to the
  hesitation words, and say which parts you are and aren't sure of.
- Behaviour readings (deep mode only) describe how the model was writing (formal or casual, asking or stating, and so on).
  They are not a fact check.

## Good uses

- A second opinion on a factual claim, a piece of code or a draft. Report its certainty and hesitations.
- Private or offline work the user doesn't want sent elsewhere: the prompt never leaves their computer.
- Comparing local models: ask the same prompt to two models and compare the certainty and hesitations.

If the tools are missing, the user needs Lens installed: `pip install proprioceptive-lens` (Ollama for local
models: https://ollama.com). Then restart Claude Code.
