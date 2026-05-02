# Anthropic Prompt Cache 自動注入

Claude (Anthropic) は OpenAI と違い **明示的に `cache_control` マーカーを刺さないと prompt cache が効かない**。素で使うと scenario が毎コール全額課金される。これを 1 行で OpenAI 同等の自動キャッシュ挙動にする機能。

## 有効化 (default true)

```yaml
# config/config.{multi_turn,single_turn}.{jp,en}.yml
anthropic:
  model: claude-opus-4-5-20251101
  temperature: 1.0
  cache: true            # default true
  cache_ttl: "5m"        # "5m" (default) or "1h"
```

## 仕組み

`src/utils/anthropic_cache.py` の `apply_cache_control(messages)` を `RunnableLambda` で `ChatAnthropic` の前段に挟む:

```
[user input messages] → apply_cache_control() → ChatAnthropic.invoke()
```

`apply_cache_control` の動作:
1. messages list を **コピー** (元の `llm_message_history` は変更しない)
2. **末尾の HumanMessage を除いた、その前の最後の AIMessage** の最終 text block に `cache_control: {"type": "ephemeral"}` を付ける
3. `content` が `str` の場合は `[{"type": "text", "text": <str>, "cache_control": ...}]` に変換
4. AIMessage が無いケース (initialize 直後等) は cache_control 付与をスキップ

これで「system + scenario + 過去ターン全部 + 最後の AI 応答」までが prompt cache の対象になり、新規 HumanMessage 部分だけが uncached になる。

`llm_builder.py` で:

```python
case "anthropic":
    inner = ChatAnthropic(...)
    if not cache_enabled:
        return inner, meta
    ttl = section.get("cache_ttl", "5m")
    runnable = RunnableLambda(lambda msgs: apply_cache_control(msgs, ttl=ttl)) | inner
    return runnable, meta
```

エージェント側のコードは触らず、透明に動作する。

## TTL

| 値 | cache write 倍率 | 期間 |
|---|---|---|
| `5m` (default) | 1.25× | 5 分 |
| `1h` | 2× | 1 時間 |

1 局が 5 分以内に終わるなら 5m で十分。長引くゲームや、複数局を続けて回すなら 1h を検討。

## Anthropic 制約

- **最大 4 個の `cache_control` breakpoint /リクエスト**: 本実装は 1 個のみ使うので余裕
- **最低トークン数しきい値**: Sonnet/Opus = 1024 token, Haiku = 2048 token。これ未満は cache_control マーカーが silently 無視される。manyshot scenario が常に超えるので問題なし
- **cache_read 価格**: base_input の 10% (90% 割引)。OpenAI cached_input と同水準
- **cache_write 価格**: base_input × 1.25 (5m) or × 2 (1h)。OpenAI は cache_write 無料なのでここで微差が出る

## 経済性 (Sonnet 4.5 想定)

| | gpt-5.2 (full, OpenAI auto-cache) | Sonnet 4.5 (full, 自動 cache_control) | Sonnet 4.5 (cache 無し) |
|---|---:|---:|---:|
| 入力 base | $1.75/M | $3.00/M | $3.00/M |
| キャッシュ read | $0.175/M | $0.30/M | - |
| キャッシュ write 5m | - | $3.75/M | - |
| 出力 | $14/M | $15/M | $15/M |
| **1局コスト目安** | **$0.75** | **~$1.55** | **~$8.5** |

cache 無し→ありで Claude も 1/6 程度に圧縮。OpenAI と同レベルの "85% 削減" を達成。

## 無効化

```yaml
anthropic:
  cache: false   # 旧挙動 (素の ChatAnthropic, 全額課金)
```

## 関連

- 実装: `src/utils/anthropic_cache.py` (`apply_cache_control`)
- ラッパ組み込み: `src/utils/llm_builder.py` の `case "anthropic":`
- 環境変数: `ANTHROPIC_API_KEY` (config/.env)
