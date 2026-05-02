# aiwolf-jsai-manyshot

[日本語 README](/README.md)

LLM agent for the AIWolf NLP (Natural Language Division) competition — JSAI 2026 edition.
Specialized fork that primes the LLM with **manyshot scenarios** (real play-log scripts) before each game.

## Highlights

- **Manyshot scenario priming** (`data/sample_games_md/`) — Markdown reference scripts are fed to the LLM as `(HumanMessage, AIMessage summary)` pairs in `llm_message_history`. Pre-warmed cache eliminates INITIALIZE timeouts. -> [doc/scenario_cache.md](doc/scenario_cache.md)
- **Freeform turn-taking** (`agent.freeform`) — tuned for group-chat servers that send `TALK_PHASE_START/END`. Uses a `[PASS]` control token and a per-agent remain_talk_map for natural turn handoff. -> [doc/freeform.md](doc/freeform.md)
- **Narration-split mode** (`prompt.narration_split`) — wraps dialogue in `「...」` and lets the LLM write stage directions outside the quotes. Server only sees the dialogue inside the quotes. -> [doc/narration_split.md](doc/narration_split.md)
- **multi-turn / single-turn modes** — keep the conversation history in LangChain or embed full context per request.
- **Split LangChain** — separate models / histories for the talk-group (talk/whisper) vs action-group (vote/divine/guard/attack).
- **Anthropic prompt-cache auto-injection** — get OpenAI-equivalent automatic caching on Claude with one flag (`anthropic.cache: true`). -> [doc/anthropic_cache.md](doc/anthropic_cache.md)
- **Local profile resolution** (`profile.source: local`) — looks up character names in `data/prompts/profiles.<lang>.yml` and renders the rich profile into identity.
- **Cost tracking** — per-call USD cost written to `log/<game>/cost_summary.{json,md}` in real time. Distinguishes input / cached_input / output / thinking. -> [doc/cost.md](doc/cost.md)

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
# 1) Clone
git clone <repo-url> aiwolf-jsai-manyshot
cd aiwolf-jsai-manyshot

# 2) Create the .env from the template (edit later)
cp config/.env.example config/.env

# 3) Copy configs from the .example templates
cp config/config.main.en.yml.example         config/config.main.en.yml
cp config/config.multi_turn.en.yml.example   config/config.multi_turn.en.yml
cp config/config.single_turn.en.yml.example  config/config.single_turn.en.yml

# 4) Install
uv sync

# 5) Pre-warm scenario cache (consumes LLM API)
#    Cache is written under ./data/scenario_cache/ keyed by scenario.delivery and agent.freeform.
uv run python scripts/prewarm_scenario.py
```

Set the API keys in `config/.env` (`OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` as needed):

```bash
# Run the agent (defaults to ./config/config.main.jp.yml; loads the matching child config by mode)
uv run python src/main.py

# Use the English prompts
uv run python src/main.py -c ./config/config.main.en.yml

# Run multiple configs in parallel (mind your rate limits)
uv run python src/main.py -c './config/*.main.*.yml'
```

## Config layout

Two-tier: **main config + mode-specific child config**.

| File | Purpose |
|---|---|
| `config/config.main.{jp,en}.yml` | mode, web_socket, agent, log, profile, headings |
| `config/config.multi_turn.{jp,en}.yml` | scenario / llm / prompt for multi-turn |
| `config/config.single_turn.{jp,en}.yml` | scenario / llm / prompt for single-turn |

Main `configs:` lists child paths; the matching child is merged at load time (child wins on key conflict).

### Key flags

| Flag | Where | Purpose |
|---|---|---|
| `mode` | main | `multi_turn` / `single_turn` |
| `lang` | main | `jp` / `en` (selects prompts/<lang>/) |
| `headings.enabled` / `headings.style` | main | prepend headings to blocks (`markdown` / `xml`) |
| `profile.source` | main | `server` / `local` |
| `agent.num` | main | 5 / 9 / 13 (village size) |
| `agent.freeform` | main | enable freeform-server-tuned behavior |
| `agent.kill_on_timeout` | main | force-stop the action thread on timeout |
| `scenario.enabled` | mode | load manyshot scripts |
| `scenario.delivery` | mode | `full` (one-shot) / `by_day` (per-day chunk) |
| `scenario.ack_mode` | mode | `llm_summary` / `static` |
| `scenario.use_cache` | mode | use scenario_cache |
| `scenario.on_cache_miss` | mode | `static` / `live` / `error` |
| `scenario.prewarm.{talk,action}` | mode | dedicated prewarm model overrides |
| `llm.type` | mode | `openai` / `google` / `vertexai` / `ollama` / `anthropic` |
| `llm.separate_langchain` | mode | split LangChain by request kind |
| `llm.{talk,action}.{type,model,...}` | mode | per-stream model overrides |
| `anthropic.cache` | mode | enable Claude prompt-cache auto-injection (default true) |
| `anthropic.cache_ttl` | mode | `5m` / `1h` |
| `prompt.narration_split` | mode | wrap dialogue in `「」` and allow stage directions outside |

See [doc/config_reference.md](doc/config_reference.md) for full details.

## Modes: multi-turn / single-turn

| Mode | Characteristic | LLM input |
|---|---|---|
| **multi-turn** | conversation history kept in LangChain | full history per request |
| **single-turn** | no history; full context embedded each time | a single `HumanMessage` per call |

### single-turn details
- `initialize` / `daily_initialize` / `daily_finish` are **NOT sent to the LLM**; the agent stores them as snapshots in `day_events`
- talk / whisper / divine etc. embed `day_events` and the full `talk_history` / `whisper_history` in the prompt body

The scenario_cache mechanism is most useful in multi-turn (the manyshot summary is loaded once and reused). Single-turn pays a per-request prompt size penalty but is simpler and bounds memory growth.

## Prompt blocks

Reusable Jinja2 fragments live under `prompts/jp/` and `prompts/en/`. Switch via `lang` in the main config; reference from prompt templates with `{{ block('<name>') }}`.

| Block | Purpose |
|---|---|
| `identity.jinja` | name / role / profile |
| `history.jinja` | utterance-history loop (driven by `history_source` / `history_start`) |
| `event.jinja` | per-day event listing (`day_events` first, falling back to `info`) |
| `instruction.jinja` | minimal per-request instruction |
| `constraints.jinja` | output format + length caps + freeform `[PASS]` instruction |
| `scenario.jinja` | manyshot script feed body (full delivery) |
| `scenario_daily.jinja` | manyshot script feed (by_day delivery, per-day) |
| `scenario_system.jinja` | SystemMessage prepended before the scenario feed |

`block('<name>')` renders `prompts/<lang>/<name>.jinja` with the caller's context (equivalent to `{% include %}`). When `headings.enabled: true`, a heading is prepended to the body. Heading text is defined in `prompts/<lang>/_labels.yml`.

## scripts/

| Script | Description |
|---|---|
| `scripts/prewarm_scenario.py` | Pre-generate the manyshot summary cache (reads `config.scenario.*`) |
| `scripts/preview_prompt.py` | Read `data/sample_packet.yml` and render every request for jp/en x multi_turn/single_turn into `preview.md` |
| `scripts/render_scenario_cache.py` | Convert `data/scenario_cache/*.json` to readable Markdown under `data/scenario_cache_readable/` |
| `scripts/convert_sample_games.py` | Convert raw `data/sample_games/*.log` (CSV) to Markdown under `data/sample_games_md/` |
| `scripts/migrate_turn_observation.py` | (one-off) inject the "Turn progression and vote declaration" supplement into existing talk-side cache |

```bash
uv run python scripts/prewarm_scenario.py                    # prewarm with default config
uv run python scripts/prewarm_scenario.py --agent-num 9      # prewarm for 9-player (separate dir)
uv run python scripts/prewarm_scenario.py --force            # force-regenerate ignoring existing cache
uv run python scripts/preview_prompt.py                      # regenerate preview.md
uv run python scripts/render_scenario_cache.py               # regenerate readable .md
```

## Directory layout

```
aiwolf-jsai-manyshot/
├── config/                            # configs (.example shipped)
├── data/
│   ├── model_cost/                    # per-provider price CSVs
│   ├── prompts/profiles.{jp,en}.yml   # local profile dictionary
│   ├── sample_games/sample_games_<N>/ # original CSV logs
│   ├── sample_games_md/sample_games_<N>/  # MD-converted (manyshot source)
│   ├── scenario_cache/                # prewarm cache (4 dirs: 5p/5p_freeform/9p/9p_freeform)
│   ├── scenario_cache_readable/       # readable MD versions of the above
│   └── sample_packet.yml              # preview sample
├── doc/                               # per-feature deep-dive docs
├── prompts/
│   ├── jp/  (8 blocks + _labels.yml)
│   └── en/  (8 blocks + _labels.yml)
├── scripts/                           # utilities
├── src/
│   ├── agent/                         # Agent implementation (incl. role-specific)
│   ├── utils/                         # helpers (jinja_env / scenario_cache /
│   │                                  #   anthropic_cache / cost_utils, ...)
│   ├── main.py                        # entry point
│   └── starter.py                     # game-session loop
├── preview.md                         # (generated) prompt preview
├── pyproject.toml
└── README.md
```

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
uv run python scripts/preview_prompt.py   # sanity-check after prompt edits
```

## References

- [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent) — reference implementation
- [aiwolf-nlp-server](https://github.com/aiwolfdial/aiwolf-nlp-server) — game server
- [aiwolf-nlp-common](https://github.com/aiwolfdial/aiwolf-nlp-common) — shared library
