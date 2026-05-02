# Freeform Mode (Group-Chat Turn Taking)

サーバが `TALK_PHASE_START` / `TALK_PHASE_END` を送るグループチャット仕様 (= freeform 仕様) に最適化したエージェント挙動。

## 有効化

```yaml
# config/config.main.{jp,en}.yml
agent:
  freeform: true   # default false (旧来挙動: 毎ターン必ず talk + 5sec 固定 sleep)
```

`false` のままなら従来挙動。誤って freeform サーバに繋いでも壊れない (= フォールバック)。

## 何が変わるか

| | freeform=false (既存) | freeform=true |
|---|---|---|
| Cache dir | `sample_games_<num>/` | `sample_games_<num>_freeform/` |
| 分析プロンプト | 通常の talk 観点 | 「次発話者の選ばれ方」「残り発話回数を踏まえた振る舞い」が追加 |
| constraints プロンプト | 文字数制限のみ | `[PASS]` 制御トークン指示 + `remain_talk_map` 表示 |
| handle_talk_phase | sleep(5) で必ず talk | 初回 stagger (0〜3 sec) + `[PASS]` 検出 + 微小ジッタ (4〜6 sec) |
| handle_whisper_phase | 同上 | 同上 |

## 仕組み

### `[PASS]` 制御トークン

LLM が「今は自分のターンでない」と判断したとき `[PASS]` のみを出力する。`handle_talk_phase` は送信せず短い再試行間隔で次サイクルへ進む。

LLM が `[PASS]` を選ぶ条件 (constraints.jinja に記載):
- 直前で他キャラが名指し / 直接質問された → 名指し相手 (= 自分でない) の応答待ち
- 自分が直前のターンで発話済みで, 追加すべき新論点が無い
- Over したキャラへの応答を求められている (誰も応えられない不毛な往復になる)
- 同じ趣旨を繰り返すだけになりそう
- 残り発話回数を温存したい (核心議題まで取っておきたい)

### `remain_talk_map`

各エージェントの本日残り発話回数を毎ターン計算してプロンプトに表示する:

```
本日の各エージェントの残り発話可能回数:
- メイ: 残り 2 回
- シオン: 残り 1 回
- ミヅキ: Over (発話済み・応答不可)
...
```

これにより LLM は「Over してる人に質問しない」「残り少ない人に核心質問を振らない」を判断できる。

`info.talk_history` から本日のエージェント別発話数を集計し, `setting.talk.max_count.per_agent` (典型的に 4) から減算して計算。

### 初回 stagger (0〜3 秒)

5 つのプロセスが同時に `TALK_PHASE_START` を受信して同時に LLM コールを開始すると, 全員が空の `talk_history` を見て独立に「Day 開幕セリフ」を生成 → 並列 monoculture (全員「CO お願い」) が発生する race condition がある。

これを避けるため, `handle_talk_phase` 開始直後に **`agent_name` 基準の決定論的 0〜3 秒 stagger** を入れる。早い人の broadcast が遅い人に届いてから発話判断される。

```python
# Agent._initial_freeform_stagger_seconds()
bucket = abs(hash(self.agent_name)) % 1000
return (bucket / 1000.0) * 3.0
```

各エージェントは別プロセスなので `PYTHONHASHSEED` が異なり、同じ名前でも実行ごとに stagger 値が変わる (= 確率的分散)。5 つの IID 一様分布 [0, 3] サンプル。

### 微小ジッタ

通常の発話間隔も `random.uniform(4.0, 6.0)` 秒のジッタを入れて, 同期発話を徐々にずらす。`[PASS]` 後の再試行間隔は `random.uniform(2.0, 3.5)` 秒。

## Cache 連動

`agent.freeform: true` のとき:
- `resolve_cache_dir` が `sample_games_<num>_freeform/` を返す
- `scenario.jinja` の talk ブランチに「次発話者の選ばれ方」「残り発話回数を踏まえた振る舞い」観点が追加される
- `prewarm_scenario.py` も freeform フラグを踏まえて `_freeform` ディレクトリに書き出す

なので freeform 用キャッシュは別途 prewarm が必要:

```bash
# config を一時的に freeform: true に切替えて prewarm
sed -i 's/^  freeform: false$/  freeform: true/' config/config.main.jp.yml
uv run python scripts/prewarm_scenario.py
sed -i 's/^  freeform: true$/  freeform: false/' config/config.main.jp.yml
```

## 注意

- freeform 仕様サーバが `TALK_PHASE_START` を送らない場合, `handle_talk_phase` 自体が呼ばれないので `agent.freeform: true` でも実質無効 (旧来の request/response パスが走る)
- 初回 stagger を 0 にしたい場合は `_initial_freeform_stagger_seconds()` を上書きするのではなく `agent.freeform: false` を選ぶこと
- `[PASS]` を全員が連続で出すと talk_phase が空回りで終わる可能性があるが、サーバが `remain_count` を消費しないので実害は小さい (TALK_PHASE_END で抜けるだけ)
