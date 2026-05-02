# Config Reference

全設定項目の一覧。`config/*.yml.example` を起点としつつ、コードでサポートされる全フラグを列挙する。

## 構成

```
config/
├── config.main.{jp,en}.yml.example       # メイン (mode / web_socket / agent / log / profile / headings)
├── config.multi_turn.{jp,en}.yml.example # multi-turn 子 (scenario / llm / prompt)
├── config.single_turn.{jp,en}.yml.example# single-turn 子 (scenario / llm / prompt)
└── .env.example                          # API キー
```

メイン config の `configs:` セクションで子 config パスを指定し、`mode` に応じたものが起動時にマージされる (キー衝突時は子優先)。

## メイン config

```yaml
mode: multi_turn         # multi_turn | single_turn
lang: jp                 # jp | en (prompts/<lang>/ 配下を選択)

headings:
  enabled: true          # 各ブロック冒頭に見出しを付与するか
  style: markdown        # markdown | xml

profile:
  source: local          # server (info.profile を使う) | local (profiles.yml から引く)

configs:
  multi_turn: ./config.multi_turn.jp.yml
  single_turn: ./config.single_turn.jp.yml

web_socket:
  url: ws://127.0.0.1:8080/ws
  token:                 # 将来用 (現在未使用)
  auto_reconnect: false  # 接続切断時の自動再接続

agent:
  num: 5                 # 5 / 9 / 13 (村サイズ)
  team: kanolab          # サーバ側で表示されるチーム名
  kill_on_timeout: true  # アクションタイムアウト時にスレッド強制停止
  freeform: false        # グループチャット仕様サーバ向け挙動

log:
  console_output: true
  file_output: true
  output_dir: ./log
  level: debug

  request:               # request 種別ごとに DEBUG ログ出力するか
    name: false
    initialize: false
    daily_initialize: false
    whisper: true
    talk: true
    daily_finish: false
    divine: true
    guard: true
    vote: true
    attack: true
    finish: false
```

## 子 config (mode 別)

```yaml
scenario:
  enabled: true                # manyshot 台本を読むか
  delivery: full               # full | by_day
  ack_mode: llm_summary        # llm_summary | static
  ack_static_text: "..."       # ack_mode=static のとき積む固定文
  use_cache: true              # scenario_cache を使うか
  on_cache_miss: static        # static | live | error
  prewarm:                     # prewarm 専用モデル指定 (任意)
    talk:
      type: openai
      model: gpt-5.4
    action:
      type: openai
      model: gpt-5.4
  # 上級オプション
  sample_dir: ./data/sample_games_md/sample_games_5  # 既定: agent.num に応じて自動選択
  glob: "*.md"                 # 既定: "*.md"
  cache_dir:                   # 既定: ./data/scenario_cache/sample_games_<num>[_freeform]/

llm:
  type: openai                 # openai | google | vertexai | ollama | anthropic
  sleep_time: 0
  separate_langchain: true     # talk/action で LangChain 分離
  talk:                        # separate_langchain=true 時の talk 系統設定
    type: openai
    model: gpt-5.2
    # temperature / pricing_mode / base_url も上書き可能
  action:
    type: openai
    model: gpt-4o-mini

# プロバイダ別デフォルト (llm.{talk,action} で上書き可能)
openai:
  model: gpt-4o-mini
  temperature: 0.7

google:
  model: gemini-2.0-flash-lite
  temperature: 0.7

vertexai:
  model: gemini-2.0-flash-lite
  temperature: 0.7

ollama:
  model: llama3.1
  temperature: 0.7
  base_url: http://localhost:11434

anthropic:
  model: claude-opus-4-5-20251101
  temperature: 1.0
  cache: true                  # prompt cache 自動注入 (default true)
  cache_ttl: "5m"              # "5m" | "1h"

prompt:
  narration_split: false       # 「」内を発話本文として抽出するモード

  initialize: |-               # 各リクエスト用プロンプトテンプレート
    {{ block('identity') }}
    {{ block('instruction') }}
    ...
  daily_initialize: |- ...
  daily_finish: |- ...
  talk: |- ...
  whisper: |- ...
  divine: |- ...
  guard: |- ...
  vote: |- ...
  attack: |- ...
```

## .env

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
# Vertex AI は Application Default Credentials (ADC) を使う
# gcloud auth application-default login
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## フラグ詳細リンク

| フラグ | 詳細ドキュメント |
|---|---|
| `agent.freeform` | [freeform.md](freeform.md) |
| `prompt.narration_split` | [narration_split.md](narration_split.md) |
| `scenario.*` | [scenario_cache.md](scenario_cache.md) |
| `anthropic.cache` / `cache_ttl` | [anthropic_cache.md](anthropic_cache.md) |
| `data/model_cost/*.csv` | [cost.md](cost.md) |

## インライン上書きキー (`llm.{talk,action}` 直下)

| キー | 用途 |
|---|---|
| `type` | プロバイダ (openai / google / ollama / anthropic / vertexai) |
| `model` | モデル ID |
| `temperature` | 温度 |
| `pricing_mode` | 料金モード (standard / batch 等) |
| `base_url` | ollama 等のエンドポイント |

`api_key` は **書けない** (起動時にエラー)。API キーは `.env` 経由のみ。
