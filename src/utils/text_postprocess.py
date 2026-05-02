"""Lightweight post-processing for LLM-generated text before sending to the server.

LLM 出力をサーバ送信前に軽く整形するための小さなユーティリティ群.
純粋関数のみで, エージェント状態に触れない.
"""

from __future__ import annotations

import re

# ``@`` と名前の間で許容する区切り文字 (半角スペース, タブ, 全角スペース).
_AT_SEPARATORS = (" ", "\t", "　")

# narration_split モードで「」内の発話本文を抽出する正規表現.
# - 開き「と閉じ」を non-greedy で対応付ける.
# - 「」内に改行が入りうるので DOTALL 相当が必要だが, 後段でスペース化するため
#   ``[\s\S]`` で表現して flag 不要にする.
# - non-greedy なので外側ネストはそのまま 1 段目だけ抽出される
#   (例: ``「彼は『お願い』と言った」`` → ``彼は『お願い』と言った``).
_DIALOGUE_QUOTE_RE = re.compile(r"「([\s\S]*?)」")
# 抽出した本文に含まれうる改行 (CR/LF) を半角スペース 1 個に潰すための正規表現.
_NEWLINE_RUN_RE = re.compile(r"[\r\n]+")
# 不正に閉じない「で始まり末尾まで続くケースを救う fallback パターン.
_UNCLOSED_DIALOGUE_RE = re.compile(r"「([\s\S]+)$")


def extract_dialogue_quotes(text: str) -> str:
    """Extract Japanese-quoted dialogue (``「...」``) and join into a single utterance.

    LLM が narration_split モードで出力した「ト書き混じりテキスト」から,
    ``「`` と ``」`` で囲まれた発話本文だけを取り出して 1 つの発話文字列に
    潰して返す. 主な挙動:

    - ``「...」`` を non-greedy にマッチさせ, 全件抽出して半角スペースで連結.
    - 抽出した本文中の改行は半角スペース 1 個に置換 (1 発話 = 1 行に潰す).
    - ``「`` が 1 個も見つからなければ全文をそのまま返す (旧モード fallback).
    - ``「`` で始まり ``」`` で閉じない壊れた出力は, ``「`` から末尾までを救出する.

    Args:
        text (str): LLM 出力の生テキスト / Raw LLM response.

    Returns:
        str: ``「」`` 内の発話本文を抽出して連結した文字列 / Extracted dialogue.
    """
    if not text:
        return text
    matches = _DIALOGUE_QUOTE_RE.findall(text)
    if matches:
        cleaned = (_NEWLINE_RUN_RE.sub(" ", m).strip() for m in matches)
        return " ".join(s for s in cleaned if s)
    # 「が 1 つも閉じなかった場合の救済 (LLM が末尾 ``」`` を落としたケース).
    unclosed = _UNCLOSED_DIALOGUE_RE.search(text)
    if unclosed:
        return _NEWLINE_RUN_RE.sub(" ", unclosed.group(1)).strip()
    # 「」が一切無い → 全文を発話扱い (旧モード fallback).
    return text


def prepend_at_if_missing(text: str, name: str) -> str:
    """Prepend ``@`` to occurrences of ``name`` that are not already prefixed.

    ``name`` の出現箇所のうち直前 (空白をスキップした非空白文字) が ``@`` でないものに
    ``@`` を付与して返す. 既に ``@`` / ``@ `` / ``@　`` のように前置されていれば触らない.

    Args:
        text (str): Source text / 入力テキスト.
        name (str): Character name to guard / 付与対象のキャラクター名.

    Returns:
        str: Text with ``@`` prepended where missing / ``@`` を補ったテキスト.
    """
    if not name or not text:
        return text
    result: list[str] = []
    i = 0
    name_len = len(name)
    text_len = len(text)
    while i < text_len:
        if text.startswith(name, i):
            # Walk back through whitespace to find the immediate non-whitespace predecessor.
            j = i - 1
            while j >= 0 and text[j] in _AT_SEPARATORS:
                j -= 1
            if j >= 0 and text[j] == "@":
                # Already tagged — leave as-is.
                result.append(text[i : i + name_len])
            else:
                result.append("@")
                result.append(name)
            i += name_len
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def enforce_at_prefix_for_names(text: str, names: list[str]) -> str:
    """Apply :func:`prepend_at_if_missing` for each name, longest-first.

    複数のキャラクター名を長い順に処理することで, 部分一致による二重付与を避ける.
    例: "ミナ" と "ミナコ" が両方名前の場合, "ミナコ" を先に処理してから "ミナ" を処理する.

    Args:
        text (str): Source text / 入力テキスト.
        names (list[str]): Character names to enforce / 対象のキャラクター名群.

    Returns:
        str: Transformed text / 変換後テキスト.
    """
    if not text or not names:
        return text
    ordered = sorted({n for n in names if n}, key=len, reverse=True)
    out = text
    for name in ordered:
        out = prepend_at_if_missing(out, name)
    return out
