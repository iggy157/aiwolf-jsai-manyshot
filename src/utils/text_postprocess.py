"""Lightweight post-processing for LLM-generated text before sending to the server.

LLM 出力をサーバ送信前に軽く整形するための小さなユーティリティ群.
純粋関数のみで, エージェント状態に触れない.
"""

from __future__ import annotations

# ``@`` と名前の間で許容する区切り文字 (半角スペース, タブ, 全角スペース).
_AT_SEPARATORS = (" ", "\t", "\u3000")


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
