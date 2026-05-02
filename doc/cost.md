# コストトレース

LLM 呼び出しごとに `AIMessage.usage_metadata` からトークン使用量を抽出し、`data/model_cost/*.csv` を参照して USD 換算。`log/<game>/cost_summary.{json,md}` をリアルタイム生成する。

## 出力先

`log/<YYYYMMDDHHmmssSSS>/` 配下 (`agent_logger` と同じ命名規則):

```
log/20260418033529578/
  yharada1.log
  yharada2.log
  ...
  cost_summary.json    # 呼び出しごとに上書き (fcntl ロック付き)
  cost_summary.md      # finish 時に生成
```

## 集計対象

トークンは 4 種類に分離:

| 種類 | 説明 |
|---|---|
| `input` | 通常の入力トークン |
| `cached_input` | プロンプトキャッシュから読み込まれたトークン (10% 割引で課金) |
| `output` | 通常の出力トークン |
| `thinking` | reasoning / extended thinking トークン (出力価格で課金) |

サポートする抽出元:
- OpenAI reasoning (`output_token_details.reasoning`)
- Anthropic extended thinking (`output_token_details.reasoning`)
- Google cached content (`input_token_details.cache_read`)
- 旧形式 `response_metadata.token_usage` も best-effort で取得

## 料金表

`data/model_cost/openai.csv` / `anthropic.csv` / `google.csv` の 3 ファイル。

```csv
provider,product_group,model_name,model_id,status,pricing_mode,context_band,...,input_price_usd,cached_input_price_usd,output_price_usd,...
openai,chat,GPT-5.2,gpt-5.2,active,token,400k,...,1.75,0.175,14.00,...
```

価格は **1M tokens あたり USD**。

新モデル追加時は対応する CSV に行を追加するだけ。`pricing_mode` (standard / batch / その他) を config の `<provider>.pricing_mode` で切替可能 (既定 `standard`)。

## モデル別の挙動

- **OpenAI**: 1 model_id につき 1 行。pricing_mode 列は `token` だが内部で `standard` に正規化される
- **Anthropic**: standard / batch の 2 モードを別行に保存
- **Google**: model × pricing_mode × prompt_size_band で複数行、`all` または短い band を優先採用
- **Ollama**: 無料計上 (`_FREE_PROVIDERS` 指定)
- **未登録モデル**: 警告ログ + `unknown_pricing` フラグ付与, cost = 0

## ファイル形式

`cost_summary.json`:
```json
{
  "game_id": "01KQD...",
  "mode": "multi_turn",
  "updated_at": "2026-05-02T17:25:08+00:00",
  "total": {
    "input_tokens": 343214,
    "cached_input_tokens": 3868416,
    "output_tokens": 5531,
    "thinking_tokens": 0,
    "cost_usd": 0.781,
    "call_count": 133,
    "unknown_pricing_calls": 0
  },
  "by_model": { "OpenAI/gpt-5.2/standard": {...}, ... },
  "by_agent":  { "yharada1": {...}, ... },
  "by_agent_model": { "yharada1": { "OpenAI/gpt-5.2/standard": {...} }, ... },
  "records": [ {ts, agent, request, ...}, ... ]
}
```

`cost_summary.md` は finish 時に renderer が生成する人間可読版。

## 設計上のポイント

- **fcntl ロック**: 5 つの multiprocessing 子プロセスが同時に書き込むので各 write を排他
- **request 毎の records 配列**: 全 LLM コールが時系列で残る → デバッグ用
- **集計は in-process 累積**: 毎 request で全件再計算ではなく既存 dict を increment

## 関連

- `src/utils/cost_utils.py`: 料金 CSV 読込, トークン抽出, USD 換算
- `src/utils/cost_logger.py`: JSON 書込 / MD レンダリング
- `data/model_cost/openai.csv` / `anthropic.csv` / `google.csv`: 料金テーブル
