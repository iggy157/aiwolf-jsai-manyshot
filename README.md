# aiwolf-jsai-manyshot

[README in English](/README.en.md)

人狼知能コンテスト (自然言語部門) JSAI 2026 版の LLM エージェント実装。
**お手本台本 (manyshot scenarios)** を LLM の「事前学習」コンテキストに焼き込んでから対局させる方式に特化したフォーク。

## 主な機能

- **Manyshot scenario priming** (`data/sample_games_md/`) — 実プレイログを Markdown で読ませて要約させ, 起動時に `(HumanMessage 台本, AIMessage 要約)` ペアを `llm_message_history` に積む。事前 prewarm でキャッシュするので INITIALIZE タイムアウトを回避できる。 → [doc/scenario_cache.md](doc/scenario_cache.md)
- **Freeform turn-taking** (`agent.freeform`) — サーバが `TALK_PHASE_START/END` を送るグループチャット仕様に最適化。`[PASS]` 制御トークンと残り発話回数マップで自然な交代を実現。 → [doc/freeform.md](doc/freeform.md)
- **Narration-split** (`prompt.narration_split`) — 発話を `「...」` で囲ませてト書きを外側に書かせ, サーバ送信前に `「」` 内側のみ抽出。物語性のある台本になる。 → [doc/narration_split.md](doc/narration_split.md)
- **multi-turn / single-turn モード** — 会話履歴を LangChain で保持するか, 毎回フルコンテキストを埋め込むかを切替。
- **LangChain 分離** — 発話系 (talk/whisper) とアクション系 (vote/divine/guard/attack) で別モデル・別履歴を使える。
- **Anthropic prompt cache 自動注入** — Claude を使うとき OpenAI の自動キャッシュ相当の挙動を 1 行 (`anthropic.cache: true`) で。 → [doc/anthropic_cache.md](doc/anthropic_cache.md)
- **プロフィール解決** (`profile.source: local`) — 台本登場人物名と同名のキャラクターを `data/prompts/profiles.<lang>.yml` から引いて identity に展開。
- **コストトレース** — `log/<game>/cost_summary.{json,md}` をリアルタイム生成。OpenAI / Anthropic / Google の cached_input・thinking まで分離集計。 → [doc/cost.md](doc/cost.md)

## クイックスタート

Python 3.11 以上 + [uv](https://docs.astral.sh/uv/) を推奨。

```bash
# 1) リポジトリ取得
git clone <repo-url> aiwolf-jsai-manyshot
cd aiwolf-jsai-manyshot

# 2) API キー用の .env をテンプレから作成 (編集は後述)
cp config/.env.example config/.env

# 3) config を example からコピー
cp config/config.main.jp.yml.example         config/config.main.jp.yml
cp config/config.multi_turn.jp.yml.example   config/config.multi_turn.jp.yml
cp config/config.single_turn.jp.yml.example  config/config.single_turn.jp.yml

# 4) 依存インストール
uv sync

# 5) 台本キャッシュを事前生成 (LLM API を消費)
#    config の scenario.delivery と agent.freeform に応じた cache が ./data/scenario_cache/ に書かれる
uv run python scripts/prewarm_scenario.py
```

`config/.env` に必要な API キー (`OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` のうち使うもの) を設定後:

```bash
# エージェント起動 (既定で ./config/config.main.jp.yml を読み, モードに応じて子configを自動マージ)
uv run python src/main.py

# 英語プロンプトを使う場合
uv run python src/main.py -c ./config/config.main.en.yml

# 複数 config を並列実行 (rate limit 注意)
uv run python src/main.py -c './config/*.main.*.yml'
```

## 設定ファイル構成

設定は **メイン config + モード別子 config** の 2 層構造。

| ファイル | 役割 |
|---|---|
| `config/config.main.{jp,en}.yml` | mode (`multi_turn` / `single_turn`), web_socket, agent, log, profile, headings |
| `config/config.multi_turn.{jp,en}.yml` | multi-turn 時の scenario / llm / prompt 定義 |
| `config/config.single_turn.{jp,en}.yml` | single-turn 時の scenario / llm / prompt 定義 |

メイン config の `configs:` セクションで子 config のパスを指定。`mode` に応じて対応する子 config がロード時にマージされる (キー衝突時は子 config 優先)。

### 主要フラグ早見表

| フラグ | 場所 | 説明 |
|---|---|---|
| `mode` | main | `multi_turn` / `single_turn` |
| `lang` | main | `jp` / `en` (prompts/<lang>/ 配下を選択) |
| `headings.enabled` / `headings.style` | main | ブロック冒頭の見出し付与 (`markdown` / `xml`) |
| `profile.source` | main | `server` (info.profile を使う) / `local` (profiles.yml から引く) |
| `agent.num` | main | 5 / 9 / 13 (村サイズ) |
| `agent.freeform` | main | グループチャット仕様向け挙動を有効化 |
| `agent.kill_on_timeout` | main | アクションタイムアウト時にスレッドを強制停止 |
| `scenario.enabled` | mode | manyshot 台本を読み込むか |
| `scenario.delivery` | mode | `full` (1回でまとめてfeed) / `by_day` (day別) |
| `scenario.ack_mode` | mode | `llm_summary` (要約 cache) / `static` (固定文) |
| `scenario.use_cache` | mode | scenario_cache を使うか |
| `scenario.on_cache_miss` | mode | `static` / `live` / `error` |
| `scenario.prewarm.{talk,action}` | mode | prewarm 専用モデル指定 |
| `llm.type` | mode | `openai` / `google` / `vertexai` / `ollama` / `anthropic` |
| `llm.separate_langchain` | mode | talk/action 系統で LangChain 分離 |
| `llm.{talk,action}.{type,model,...}` | mode | 系統別モデル指定 |
| `anthropic.cache` | mode | Claude 用 prompt cache 自動注入 (default true) |
| `anthropic.cache_ttl` | mode | `5m` / `1h` |
| `prompt.narration_split` | mode | 発話を `「」` で囲ませて地の文を許可 |

詳細は [doc/config_reference.md](doc/config_reference.md) を参照。

## モード: multi-turn / single-turn

| モード | 特徴 | LLM への入力 |
|---|---|---|
| **multi-turn** | 会話履歴 (`llm_message_history`) を LangChain で保持 | 毎リクエスト, 履歴全体を送信 |
| **single-turn** | 履歴を使わず毎回フルコンテキストを埋め込み | `HumanMessage` 単発のみ |

### single-turn の動作
- `initialize` / `daily_initialize` / `daily_finish` は **LLM に送信せず**, agent 内部 (`day_events`) にスナップショット保持
- talk / whisper / divine 等の各リクエストで, `day_events` とフル `talk_history` / `whisper_history` をプロンプト本文に埋め込み

multi-turn では scenario_cache 機構が効くため, manyshot priming を活かしたい場合は multi-turn 推奨。single-turn は履歴肥大化に強い反面, 1 リクエストの prompt サイズが膨らむ。

## プロンプトブロック

`prompts/jp/` と `prompts/en/` の配下に再利用可能 Jinja2 ブロック群がある。メイン config の `lang: jp` / `lang: en` で参照先を切替。子 config の `prompt.<request>` から `{{ block('<name>') }}` で参照する。

| ブロック | 役割 |
|---|---|
| `identity.jinja` | 名前・役職・プロフィール |
| `history.jinja` | 発言履歴ループ (`history_source` / `history_start` で切替) |
| `event.jinja` | 日次イベント一覧 (`day_events` 優先, 無ければ `info`) |
| `instruction.jinja` | リクエスト別の最低限の指示文 |
| `constraints.jinja` | 出力形式・文字数制限・freeform 用 [PASS] 指示 |
| `scenario.jinja` | manyshot 台本フィードの HumanMessage 本文 (full delivery) |
| `scenario_daily.jinja` | manyshot 台本フィード (by_day delivery, day 別) |
| `scenario_system.jinja` | scenario フィード前に積む SystemMessage |

`block('<name>')` は `prompts/<lang>/<name>.jinja` を呼び出し側コンテキストでレンダした結果を返す (`{% include %}` 相当)。`headings.enabled: true` の場合は本文冒頭に見出しを自動付与。見出しテキストは `prompts/<lang>/_labels.yml` で定義。

## scripts/

| Script | 説明 |
|---|---|
| `scripts/prewarm_scenario.py` | manyshot 台本 → LLM 要約のキャッシュを事前生成 (`config.scenario.*` を読み実行) |
| `scripts/preview_prompt.py` | `data/sample_packet.yml` を読み, jp/en × multi_turn/single_turn の全リクエストをレンダして `preview.md` に出力 |
| `scripts/render_scenario_cache.py` | `data/scenario_cache/*.json` を `data/scenario_cache_readable/*.md` に人間可読変換 |
| `scripts/convert_sample_games.py` | `data/sample_games/*.log` (CSV) を `data/sample_games_md/*.md` に変換 |
| `scripts/migrate_turn_observation.py` | (一時) talk-side cache に "Turn 進行と投票宣言" 補足を append (Phase 1 実験用) |

実行例:

```bash
uv run python scripts/prewarm_scenario.py                    # 既定 config で prewarm
uv run python scripts/prewarm_scenario.py --agent-num 9      # 9人村用に prewarm (5人とは別ディレクトリ)
uv run python scripts/prewarm_scenario.py --force            # 既存 cache を無視して再生成
uv run python scripts/preview_prompt.py                      # preview.md を再生成
uv run python scripts/render_scenario_cache.py               # readable .md を再生成
```

## ディレクトリ構成

```
aiwolf-jsai-manyshot/
├── config/                            # 設定ファイル (.example 同梱)
├── data/
│   ├── model_cost/                    # プロバイダ別料金 CSV
│   ├── prompts/profiles.{jp,en}.yml   # ローカルプロフィール辞書
│   ├── sample_games/sample_games_<N>/ # オリジナル CSV ログ
│   ├── sample_games_md/sample_games_<N>/  # MD 変換版 (manyshot ソース)
│   ├── scenario_cache/                # prewarm キャッシュ (4種: 5p/5p_freeform/9p/9p_freeform)
│   ├── scenario_cache_readable/       # 上記の人間可読 MD 版
│   └── sample_packet.yml              # preview 用サンプル
├── doc/                               # 機能別詳細ドキュメント
├── prompts/
│   ├── jp/  (8 ブロック + _labels.yml)
│   └── en/  (8 ブロック + _labels.yml)
├── scripts/                           # ユーティリティ
├── src/
│   ├── agent/                         # Agent 実装 (役職別含む)
│   ├── utils/                         # 各種ヘルパ (jinja_env / scenario_cache /
│   │                                  #   anthropic_cache / cost_utils 等)
│   ├── main.py                        # エントリポイント
│   └── starter.py                     # ゲームセッションループ
├── preview.md                         # (生成物) プロンプトプレビュー
├── pyproject.toml
└── README.md
```

## 開発

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # 型チェック (strict)
uv run python scripts/preview_prompt.py   # プロンプト変更後の確認
```

## 参考リンク

- [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent) — リファレンス実装
- [aiwolf-nlp-server](https://github.com/aiwolfdial/aiwolf-nlp-server) — ゲームサーバ
- [aiwolf-nlp-common](https://github.com/aiwolfdial/aiwolf-nlp-common) — 共通ライブラリ (パケット定義等)
