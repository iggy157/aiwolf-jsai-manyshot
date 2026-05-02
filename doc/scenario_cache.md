# Manyshot Scenario Cache

このプロジェクトのコア機能。`data/sample_games_md/` の Markdown 台本を LLM の「事前学習」コンテキストに焼き込み、応答を要約させてキャッシュする。

## 仕組み

```
data/sample_games_md/sample_games_5/sample_5_001.md  ┐
data/sample_games_md/sample_games_5/sample_5_002.md  │ 全部読み込み
...                                                  │
data/sample_games_md/sample_games_5/sample_5_010.md  ┘
                              │
                              ▼ scripts/prewarm_scenario.py
                              │
                  ┌──────────────────────────┐
                  │ talk side / action side  │
                  │ 別々に LLM 要約を取得     │
                  └──────────────────────────┘
                              │
                              ▼
                  data/scenario_cache/sample_games_<N>[_freeform]/
                    <hash>.json     # prompt + response 保存
                              │
                              ▼ ゲーム起動時 (Agent._feed_sample_games)
                  llm_message_history に
                  (HumanMessage 台本, AIMessage 要約) ペアで積む
```

## 4 種類のキャッシュディレクトリ

`agent.num` (5/9/13) × `agent.freeform` (false/true) で 4 種類のキャッシュが独立して存在する:

| ディレクトリ | 用途 |
|---|---|
| `data/scenario_cache/sample_games_5/`           | 5人村 / request-response 仕様サーバ |
| `data/scenario_cache/sample_games_5_freeform/`  | 5人村 / freeform (TALK_PHASE_*) 仕様サーバ |
| `data/scenario_cache/sample_games_9/`           | 9人村 / request-response |
| `data/scenario_cache/sample_games_9_freeform/`  | 9人村 / freeform |

`agent.freeform` を切替えるだけで読み込みディレクトリが変わる仕組みになっており、両モード用のキャッシュを並行保持できる。

## 配信モード (`scenario.delivery`)

| 値 | 動作 |
|---|---|
| `full` (既定) | INITIALIZE で全 day まとめて 1 回フィード。1 LLM コール / agent。 |
| `by_day` | INITIALIZE で各 manyshot の Day 0 部分のみフィード。Day 1 以降は `daily_initialize` で当該日章節を継ぎ足す。各日が別キャッシュ。 |

`by_day` のキャッシュキーには day が含まれるため、Day 0/1/2 で別ファイルになる。`prewarm_scenario.py` は `by_day` モードのとき自動で全 day を prewarm する。

## 要約モード (`scenario.ack_mode`)

| 値 | 動作 |
|---|---|
| `llm_summary` (既定) | LLM に台本要約を生成させて AIMessage として履歴に積む。最も学習効果が高い反面 prewarm に LLM 呼び出しが必要。 |
| `static` | 固定の承諾文 (`scenario.ack_static_text`) を AIMessage として積む。API コールなし。 |

実ゲーム時の `scenario.use_cache: true` (既定) でキャッシュ読みのみ → LLM コールゼロ → INITIALIZE タイムアウト回避。

## キャッシュキー

`SHA-256(provider, model_id, lang, target_role, prompt_text, system_text, day)`。

prompt 文字列が 1 文字でも変われば別キーになり、自動的に古いキャッシュが無効化される。逆に同じ prompt なら永続的にヒットする。

## prewarm 専用モデル指定

実ゲームで使うモデル (例: `gpt-5.2`) と prewarm に使うモデル (例: 安価な `gpt-5.4`) を分離できる:

```yaml
scenario:
  prewarm:
    talk:
      type: openai
      model: gpt-5.4    # prewarm はこちらで安く済ませる
    action:
      type: openai
      model: gpt-5.4
llm:
  type: openai
  talk:
    type: openai
    model: gpt-5.2     # 実ゲームはこちら
```

cache key は prewarm 側の `(provider, model)` で計算されるため、runtime と独立。実ゲーム起動時は cache hit が保証される。

## キャッシュミス時の挙動 (`scenario.on_cache_miss`)

| 値 | 動作 |
|---|---|
| `static` (既定, 推奨) | `ack_static_text` にフォールバック。タイムアウト安全。 |
| `live` | 実行時に LLM を呼ぶ。タイムアウトリスクあり (旧挙動)。 |
| `error` | 例外を投げて INITIALIZE を失敗させる (CI/開発用)。 |

## 関連スクリプト

```bash
# 既定 config で prewarm
uv run python scripts/prewarm_scenario.py

# 別 config 指定
uv run python scripts/prewarm_scenario.py -c ./config/config.main.jp.yml

# 既存 cache を無視して再生成
uv run python scripts/prewarm_scenario.py --force

# agent.num を上書き (5人/9人 両方 prewarm したいとき)
uv run python scripts/prewarm_scenario.py --agent-num 9

# 特定 target_role だけ prewarm
uv run python scripts/prewarm_scenario.py --target-role talk

# キャッシュ JSON → 人間可読 MD 変換 (save_cache_entry が自動で生成するが, 一括再生成用)
uv run python scripts/render_scenario_cache.py
```

## キャッシュ readable .md

`data/scenario_cache_readable/sample_games_<N>[_freeform]/<role>_<day_label>__<short_hash>.md` 形式で人間可読版が自動生成される。`save_cache_entry` が JSON 書き出しと同時に MD も書く仕組み。

ファイル名例:
```
talk_full__d214a36a.md      # 5人村 full モード, talk 系
talk_day0__28108c6e.md      # 5人村 by_day モード, Day 0 talk
action_full__6441080f.md    # 5人村 full モード, action 系
```

## 分析プロンプトの観点

`prompts/<lang>/scenario.jinja` で要約させる観点は「**状況 → 議論/判断**」の因果対応にチューニングされている (線形パターン暗記を避ける)。具体的には:

- 盤面状況と議題の対応 (CO 数, 黒判定, 残り人数等から議題が決まる仕組み)
- 観察される議論展開のパターン (主要 2〜3 件, 状況条件付き)
- Day ごとのフェーズ感
- 同じ議題への応答の散らばり方 (役割分散の観察)
- 議論の主流から外れる位置取り (非を認める / 中立 / 整理役)
- Day 内の Turn 進行と投票宣言フェーズ
- (freeform モード時) 次発話者の選ばれ方 / 残り発話回数を踏まえた振る舞い

これらは台本観察として LLM に抽出させ、実ゲーム時の発話・判断の参考に使う。
