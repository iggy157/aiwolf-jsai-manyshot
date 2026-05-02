"""Anthropic prompt-cache injection helper.

Anthropic prompt cache の自動注入ユーティリティ.

Anthropic API は ``cache_control: {"type": "ephemeral"}`` を content block に
明示的に付与しないと prompt cache を効かせられない. OpenAI の自動キャッシュと
同等の体験を得るため, LLM 呼び出し直前に「末尾の HumanMessage を除いた
最後の AIMessage」の最終 text block へ cache_control を注入する.

これにより ``system + scenario + 過去ターン全部 + 最後の AI 応答`` までが
prompt cache の prefix として再利用される. 末尾の HumanMessage (今回の質問) は
毎ターン変わるためキャッシュ対象外とする.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

_VALID_TTLS: tuple[str, ...] = ("5m", "1h")


def _build_cache_control(ttl: str) -> dict[str, str]:
    """Return the ``cache_control`` payload for the given TTL.

    指定された TTL に対応する ``cache_control`` 辞書を返す.

    Args:
        ttl (str): "5m" または "1h".

    Returns:
        dict[str, str]: Anthropic API 用の cache_control 辞書.
    """
    if ttl == "5m":
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": "1h"}


def _clone_ai_with_cached_content(
    msg: AIMessage,
    cache_control: dict[str, str],
) -> AIMessage:
    """Return a new ``AIMessage`` whose last text block carries ``cache_control``.

    最後の text block に ``cache_control`` を付与した新しい ``AIMessage`` を返す.
    元の ``msg`` および ``msg.content`` (list の場合) は in-place 変更しない.

    Args:
        msg (AIMessage): 元の AIMessage.
        cache_control (dict[str, str]): 付与する cache_control 辞書.

    Returns:
        AIMessage: cache_control を含む新しい AIMessage.
    """
    content: str | list[str | dict[str, Any]] = cast(
        "str | list[str | dict[str, Any]]",
        msg.content,  # pyright: ignore[reportUnknownMemberType]
    )
    new_content: list[str | dict[str, Any]]
    if isinstance(content, str):
        new_content = [{"type": "text", "text": content, "cache_control": cache_control}]
    else:
        new_content = [dict(block) if isinstance(block, dict) else block for block in content]
        last_text_idx: int | None = None
        for idx in range(len(new_content) - 1, -1, -1):
            block = new_content[idx]
            if isinstance(block, dict) and block.get("type") == "text":
                last_text_idx = idx
                break
        if last_text_idx is None:
            new_content.append({"type": "text", "text": "", "cache_control": cache_control})
        else:
            existing = new_content[last_text_idx]
            assert isinstance(existing, dict)  # noqa: S101  探索ループで保証済み.
            target: dict[str, Any] = dict(existing)
            target["cache_control"] = cache_control
            new_content[last_text_idx] = target
    additional_kwargs: dict[str, Any] = dict(
        cast("dict[str, Any]", msg.additional_kwargs),  # pyright: ignore[reportUnknownMemberType]
    )
    response_metadata: dict[str, Any] = dict(
        cast("dict[str, Any]", msg.response_metadata),  # pyright: ignore[reportUnknownMemberType]
    )
    return AIMessage(
        content=new_content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
        name=msg.name,
        id=msg.id,
    )


def apply_cache_control(
    messages: list[BaseMessage],
    ttl: str = "5m",
) -> list[BaseMessage]:
    """Return a copy of ``messages`` with ``cache_control`` attached for prompt caching.

    末尾の HumanMessage を除いた直前の最後の AIMessage に ``cache_control`` を
    付与した新しいメッセージリストを返す. 元のリストおよび要素は変更しない.

    cache 対象 AIMessage が見つからない (履歴が短い / AIMessage が一つもない) 場合は
    元のリストの浅いコピーをそのまま返す. Anthropic 側で 1024 token 未満の prefix は
    無視されるため, この場合に cache_control を付けても効果がない.

    Args:
        messages (list[BaseMessage]): LLM に渡す予定のメッセージ列.
        ttl (str): キャッシュ TTL. "5m" (default) または "1h" のみ受け付ける.

    Returns:
        list[BaseMessage]: cache_control を注入した新しいリスト.

    Raises:
        ValueError: ``ttl`` が "5m" / "1h" 以外の場合.
    """
    if ttl not in _VALID_TTLS:
        msg = f"ttl must be one of {_VALID_TTLS}, got {ttl!r}"
        raise ValueError(msg)
    if not messages:
        return list(messages)

    search_end = len(messages) - 1 if isinstance(messages[-1], HumanMessage) else len(messages)
    target_idx: int | None = None
    target_ai: AIMessage | None = None
    for idx in range(search_end - 1, -1, -1):
        candidate = messages[idx]
        if isinstance(candidate, AIMessage):
            target_idx = idx
            target_ai = candidate
            break
    if target_idx is None or target_ai is None:
        return list(messages)

    new_messages: list[BaseMessage] = list(messages)
    cache_control = _build_cache_control(ttl)
    new_messages[target_idx] = _clone_ai_with_cached_content(target_ai, cache_control)
    return new_messages
