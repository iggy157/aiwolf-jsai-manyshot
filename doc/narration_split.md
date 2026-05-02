# Narration-Split Mode

LLM に発話本文を `「...」` で囲ませてその外側に「ト書き・地の文」を許可するモード。サーバ送信前に `「」` 内側だけ抽出するので、サーバから見ると通常通りの 1 発話、内部的には物語性のある脚本として記録される。

## 有効化

```yaml
# config/config.{multi_turn,single_turn}.{jp,en}.yml
prompt:
  narration_split: true   # default false
```

## 出力形式

LLM はこのような形式で出力する:

```
（少し息を整えて、ミヅキの方を見る）
「占い結果はサクラさん、白でした」
（深く息を吐く）
「対抗が居ないので、このまま進めましょう」
```

`extract_dialogue_quotes()` がサーバ送信前に `「」` 内側のみを抽出して連結:

```
占い結果はサクラさん、白でした 対抗が居ないので、このまま進めましょう
```

これがサーバに送信される。

## 文字数制限の扱い

文字数制限 (`setting.talk.max_length`) は **`「」` 内側の合計のみ** に適用されることを LLM に明示する。ト書き部分は API トークンとしてのみ消費 (送信ゼロ)。

constraints.jinja で:

```
発話本文は必ず ``「...」`` で囲んで出力してください。``「」`` の外側にはキャラクターの仕草・視線・間合いなどのト書きを書いてよく (例: ``（少し間を置いて）``), サーバには ``「」`` 内側の文字列だけが連結されて送信されます。上記の文字数制限は ``「」`` 内側の合計に対してのみ適用され, ト書き部分は字数にカウントされません。
```

## エッジケース

| ケース | 挙動 |
|---|---|
| `「」` が 0 個 | 全文をそのまま発話扱い (旧モード fallback) |
| `「」` 複数 | 半角スペースで連結 |
| `「」` 内に改行 | 改行をスペース化 (1 発話 = 1 行) |
| `「` で始まり `」` で閉じない | `「` から末尾までを救出 (LLM が末尾 `」` を落とした場合の救済) |
| ネスト `「彼は『お願い』と言った」` | 外側 1 段だけ抽出 (non-greedy regex) |
| 空文字 | 空文字を返す |

## Cache への影響

`narration_split: true` にすると `scenario_system.jinja` と `constraints.jinja` の本文が変わるため scenario_cache のキー (= prompt 文字列ハッシュ) が変わる。

→ **既存の (narration_split=false の) cache は使えなくなる**ので prewarm 再実行が必要。または `scenario.on_cache_miss: static` で fallback 動作させる。

両モードを並行使いしたい場合は cache を 2 セット持つ (それぞれ別ハッシュで保存される)。

## 関数

```python
from utils.text_postprocess import extract_dialogue_quotes

extract_dialogue_quotes('（少し間を置いて）\n「占い結果は白でした」')
# => '占い結果は白でした'

extract_dialogue_quotes('シオンが怪しい')   # 「」無し
# => 'シオンが怪しい'  (fallback)

extract_dialogue_quotes('私は…「待ってください、それは')  # 閉じない
# => '待ってください、それは'  (rescue)
```

## いつ有効か

- 物語性のある対局ログを残したいとき
- LLM のキャラクタ表現が「セリフのみ」で平坦になる傾向を打破したいとき (内省・仕草を出させると芝居が立体的になる)

副作用として API トークン消費がやや増える (ト書き部分の output コスト) が, OpenAI/Claude の output 価格で 1 発話あたり数十〜数百トークン程度。1 局で $0.05〜0.20 増える程度。

## 関連

- 実装: `src/utils/text_postprocess.py` (`extract_dialogue_quotes`)
- 呼び出し: `src/agent/agent.py` の `_postprocess_utterance()` から `talk()` / `whisper()` 戻り値直前で
- プロンプト: `prompts/{jp,en}/scenario_system.jinja` + `constraints.jinja` で `narration_split` フラグ参照
