# Proprioceptive Lens

**See inside your local model.** A free app for anyone running local AI with [Ollama](https://ollama.com) or Hugging Face models.

![Lens showing a confidently wrong answer: every word at near-100% certainty](https://raw.githubusercontent.com/Loganwins/ProprioceptiveAI/main/docs/demo-confidence.gif)

*qwen3:4b says the 17th US president was Garfield, with 100% certainty. (It was Andrew Johnson.) Token confidence shows the surface; Lens also lets you look underneath.*

- **Confidence you can see.** Every word is tinted by how sure the model was. Hover any word to see what it almost said.
- **Where it hesitated.** A live certainty trace, plus the exact words where the model weighed alternatives.
- **It remembers you.** Private memory of your conversations, on your computer, for any model.
- **Inside the model** (deep mode). A live layer-by-layer activity map of the network while it writes.
- **Behaviour** (optional). Internal-state probes score what the model is doing: writing code or prose, formal or casual
  register, following your instructions, creative or literal, assertive or hedged, repeating itself, asking or stating.
  These probes were validated on the original model, then transferred to each supported model. There they were
  re-tested against length and random-direction controls, and with a live test on the model's own answers.
- **Terminal mode and an API** for wiring Lens into your own stack.
- **Private.** Everything runs locally. Behaviour is opt-in and sends only 16 scrambled numbers per word, never your text.

![Deep mode: layer activity map and live Behaviour scores](https://raw.githubusercontent.com/Loganwins/ProprioceptiveAI/main/docs/demo-deep-behaviour.gif)

## Install and run

```bash
pip install proprioceptive-lens      # no dependencies; Python 3.9+
ollama pull qwen3:4b                 # any chat model works
lens                                 # opens the app in your browser
```

Options: `lens --port 7777 --ollama http://127.0.0.1:11434 --hindsight http://localhost:8888`.

## Deep mode and Behaviour

```bash
pip install "proprioceptive-lens[deep]"
lens --deep Qwen/Qwen2.5-3B-Instruct
```

Already chatting with an Ollama model? Press **See inside** in the app. Lens loads the same model's open weights
and switches you into deep mode, with no restart. Or start Lens with `--deep <model>` as above.

Deep mode runs a Hugging Face model locally and reads its hidden states. It works with Llama, Qwen, Mistral, Gemma,
Phi, Falcon, Granite, OLMo, SmolLM, DeepSeek and other families. Lens tells Ollama users which command matches the
model they already run. Switch on **Behaviour** in the app to see internal-state probe scores. The app lists which
models have a Behaviour pack; so does [docs/MODELS.md](https://github.com/Loganwins/ProprioceptiveAI/blob/main/docs/MODELS.md).

## Lens Deluxe: steer the model from the inside

Behaviour shows what the model is doing. Deluxe lets you change it while it writes. A slider per behaviour
(more or less formal, more or less repetitive, and others) pushes the model's internal state in that direction,
and **Before / after** re-runs your question without steering so you can see exactly what changed.

- Every steering direction is tested on each model before it is offered. It has to change the model's writing more
  than random pushes of the same size do. It also has to keep the text fluent, add no refusals and stay on topic.
  Sliders stop at the strongest setting that passed.
- Nothing steers automatically. You choose, and you can always compare against the unsteered answer.
- Same privacy as Behaviour: your text never leaves your computer.

Start the 7-day free trial from the Steer panel; no card is needed. Enter your email, then paste the activation token
you receive. Deluxe also removes the daily Behaviour limit.

## Proprioceptive Terminal Mode

```bash
lens term                                   # interactive: confidence-coloured output, /why, /memory, /behaviour
lens term -m qwen3:4b "your question"       # one-shot
echo "your question" | lens term --json     # JSON-lines events for scripts and pipelines
```

Events: `memory`, `token` (`text`, `p`, `alternatives`, plus `layers` in deep mode), `behaviour` (score per probe),
`done`. Any OpenAI-compatible client pointed at `http://127.0.0.1:7777/v1` also gets Lens memory.

## Use it from Claude Code, Cursor and other agents

**Claude Code plugin** (MCP server + a skill that teaches Claude when to ask your local model and how to read its certainty):

```
/plugin install proprioceptive-lens --marketplace Loganwins/ProprioceptiveAI
```

(Needs `pip install proprioceptive-lens` first. On older Claude Code: `/plugin marketplace add Loganwins/ProprioceptiveAI`
then `/plugin install proprioceptive-lens@proprioceptive-ai`.)

Or add just the MCP server: `lens mcp` runs Lens as an MCP server.

```bash
claude mcp add lens -- lens mcp            # Claude Code
```

Claude Desktop: add `{"mcpServers": {"lens": {"command": "lens", "args": ["mcp"]}}}` to `claude_desktop_config.json`
(use the full path from `which lens` if Desktop can't find it).

For Cursor or another client, add a stdio server with command `lens` and argument `mcp`. Tools:
- `ask_local_model`: the answer, its overall certainty, and the words the model hesitated on, with what it almost said
  (plus Behaviour scores in deep mode);
- `list_local_models`;
- `search_memory`.

## Share a prompt

`http://127.0.0.1:7777/?ask=Why%20is%20the%20sky%20blue` opens Lens with the prompt ready to send.

## Privacy

Lens listens only on `127.0.0.1` and rejects requests from other websites. Memories live in
`~/.proprioceptive-lens/memory.sqlite3`. You can view or forget them in the app, or delete the file. Behaviour is off
until you turn it on.

## License

MIT for this client. The Behaviour probes and the scoring service are proprietary to Proprioceptive AI, are not part
of this package, and no licence to them (or to any Proprioceptive AI patent) is granted by this licence.
