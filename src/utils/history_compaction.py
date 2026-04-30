"""History compaction helpers.

daily_finish(Day N) 完了時に Day N-2 の Human/AI ペア群を構造化サマリに置換し,
llm_message_history を軽量化するためのユーティリティ.

保持ウィンドウ:
    [Raw]        initialize prefix (scenario 台本 + ack) … day=-1 タグで常に保持
    [Compressed] Day 0 .. Day N-2                        … summary pair に置換
    [Raw]        Day N-1 (直近完了日) と Day N (進行中)   … そのまま raw

メッセージは ``additional_kwargs`` に ``manyshot_day`` / ``manyshot_phase`` を持ち,
``day`` で圧縮対象を特定する. scenario prefix / INITIALIZE は day=-1 で付与し,
圧縮対象から自動的に除外される.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Talk

# ``additional_kwargs`` に付与するメタキー.
MSG_KEY_DAY = "manyshot_day"
MSG_KEY_PHASE = "manyshot_phase"

# 圧縮対象外を示す day タグ. scenario prefix / INITIALIZE に付与する.
DAY_PERSISTENT = -1

# phase タグの固定値.
PHASE_SCENARIO = "scenario"
PHASE_INITIALIZE = "initialize"
PHASE_COMPACT_SUMMARY = "_compact_summary"

# フォールバック要約で逐語記述する際の上限文字数 (これを超えたら末尾を省略する).
_FALLBACK_UTTERANCE_LIMIT = 60


def tag_message(msg: BaseMessage, day: int, phase: str) -> BaseMessage:
    """Attach ``day`` / ``phase`` metadata to a LangChain message.

    ``additional_kwargs`` に履歴圧縮用のタグを書き込む.

    Args:
        msg (BaseMessage): Message to tag / タグを付与するメッセージ.
        day (int): Day number (``-1`` for scenario / initialize prefix).
        phase (str): Phase key (``"daily_initialize"`` / ``"talk"`` / ...).

    Returns:
        BaseMessage: Same message with ``additional_kwargs`` updated in place.
    """
    msg.additional_kwargs[MSG_KEY_DAY] = day
    msg.additional_kwargs[MSG_KEY_PHASE] = phase
    return msg


def find_day_range(history: list[BaseMessage], day: int) -> tuple[int, int] | None:
    """Find the inclusive index range of messages tagged with ``day``.

    履歴中で ``day`` タグが一致するメッセージ群の ``[start, end]`` を返す.

    Args:
        history (list[BaseMessage]): Message history to search / 走査対象の履歴.
        day (int): Day number to match / 一致させる day 値.

    Returns:
        tuple[int, int] | None: ``(start, end)`` 両端含む index, or ``None``.
    """
    start: int | None = None
    end: int | None = None
    for i, msg in enumerate(history):
        if msg.additional_kwargs.get(MSG_KEY_DAY) == day:
            if start is None:
                start = i
            end = i
    if start is None or end is None:
        return None
    return start, end


def extract_day_text(history: list[BaseMessage], start: int, end: int) -> str:
    """Serialize messages in ``[start, end]`` into a plain text block.

    サマリ LLM に渡すため, 対象範囲のメッセージを平文ブロックに直列化する.
    役割マーカーは ``HUMAN`` / ``AI`` のような英単語ではなく, 日本語ラベル
    ``[指示]`` / ``[自分の応答]`` を用いる. これは英語ラベルが要約 LLM から
    「"AI" という名前のエージェント」と誤解され, `CO状況: {AI: 占い師}` のような
    幻覚を招いていた問題への対処 (日本語の agent 名との混同を避ける).

    Args:
        history (list[BaseMessage]): Message history / 履歴本体.
        start (int): Start index inclusive / 開始 index.
        end (int): End index inclusive / 終了 index.

    Returns:
        str: Concatenated text representation / 連結したテキスト.
    """
    lines: list[str] = []
    for i in range(start, end + 1):
        msg = history[i]
        role = "指示" if isinstance(msg, HumanMessage) else "自分の応答"
        phase = msg.additional_kwargs.get(MSG_KEY_PHASE, "?")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"[{role} phase={phase}]\n{content}")
    return "\n\n".join(lines)


def replace_with_summary(
    history: list[BaseMessage],
    start: int,
    end: int,
    day: int,
    summary_text: str,
) -> None:
    """Replace messages in ``[start, end]`` with a Human+AI summary pair.

    対象範囲を Human (要約指示ダミー) + AI (要約本文) の 1 ペアに置換する
    (対話ターンの交互性を維持するため 2 メッセージで置換する).

    Args:
        history (list[BaseMessage]): Message history (mutated in place) / 変更対象.
        start (int): Start index inclusive / 開始 index.
        end (int): End index inclusive / 終了 index.
        day (int): Day number being compressed / 圧縮対象の day.
        summary_text (str): Summary body to place in AIMessage / 要約本文.
    """
    prompt = HumanMessage(content=f"Day {day} の対局記録を要約してください.")
    tag_message(prompt, day, PHASE_COMPACT_SUMMARY)
    summary = AIMessage(content=summary_text)
    tag_message(summary, day, PHASE_COMPACT_SUMMARY)
    history[start : end + 1] = [prompt, summary]


def _trim_utterance(text: str) -> str:
    """Trim an utterance to ``_FALLBACK_UTTERANCE_LIMIT`` chars with ellipsis.

    フォールバック要約で逐語を載せる際の簡易省略.
    """
    if len(text) <= _FALLBACK_UTTERANCE_LIMIT:
        return text
    return text[: _FALLBACK_UTTERANCE_LIMIT - 3] + "..."


def _fallback_event_lines(
    day: int,
    day_events: list[dict[str, Any]],
    self_agent_name: str,
) -> list[str]:
    """Produce structured lines from the last matching entry in ``day_events``.

    該当 day の最新 day_events エントリから投票/処刑/襲撃/占い/霊媒の行を組み立てる.
    """
    events = [e for e in day_events if e.get("day") == day]
    if not events:
        return []
    ev = events[-1]
    lines: list[str] = []
    medium = ev.get("medium_result")
    if medium is not None:
        target = getattr(medium, "target", "?")
        result = getattr(medium, "result", "?")
        lines.append(f"公開霊媒結果: [({target}, {result}, {self_agent_name})]")
    divine = ev.get("divine_result")
    if divine is not None:
        target = getattr(divine, "target", "?")
        result = getattr(divine, "result", "?")
        lines.append(f"公開占い結果: [({target}, {result}, {self_agent_name})]")
    votes = ev.get("vote_list")
    if votes:
        vote_map = {
            str(getattr(v, "agent", "?")): str(getattr(v, "target", "?")) for v in votes
        }
        lines.append(f"投票: {vote_map}")
    executed = ev.get("executed_agent")
    if executed:
        lines.append(f"処刑: {executed}")
    attacked = ev.get("attacked_agent")
    if attacked:
        lines.append(f"襲撃: {attacked}")
    return lines


def _fallback_talk_lines(
    day: int,
    talk_history: list[Talk],
    self_agent_name: str,
) -> list[str]:
    """Produce "主張タイムライン" lines from the day's talk history.

    talk_history のうち該当 day 分を話者別に集約して主張タイムラインを作る.
    自分を先頭, 他エージェントを agent 名順で出力する.
    """
    day_talks = [
        t for t in talk_history
        if t.day == day and not getattr(t, "skip", False) and not getattr(t, "over", False)
    ]
    if not day_talks:
        return []
    by_agent: dict[str, list[str]] = {}
    for t in day_talks:
        by_agent.setdefault(t.agent, []).append(t.text)
    ordered: list[tuple[str, list[str]]] = []
    if self_agent_name in by_agent:
        ordered.append((self_agent_name, by_agent.pop(self_agent_name)))
    ordered.extend(sorted(by_agent.items()))
    lines: list[str] = ["主張タイムライン:"]
    for agent, utterances in ordered:
        lines.append("  自分:" if agent == self_agent_name else f"  @{agent}:")
        lines.extend(
            f"    - day {day}: 「{_trim_utterance(utterance)}」"
            for utterance in utterances
        )
    return lines


def _fallback_whisper_lines(day: int, whisper_history: list[Talk]) -> list[str]:
    """Produce whisper-section lines for werewolf-side fallback summary.

    該当 day の whisper を箇条書きで追加する (人狼サイドのみ意味を持つ).
    """
    day_whispers = [
        w for w in whisper_history
        if w.day == day and not getattr(w, "skip", False) and not getattr(w, "over", False)
    ]
    if not day_whispers:
        return []
    lines = ["whisper (自陣営):"]
    lines.extend(f"  - @{w.agent}: 「{_trim_utterance(w.text)}」" for w in day_whispers)
    return lines


def build_fallback_summary(
    day: int,
    day_events: list[dict[str, Any]],
    talk_history: list[Talk],
    whisper_history: list[Talk],
    self_agent_name: str,
) -> str:
    """Build a deterministic fallback summary without calling an LLM.

    要約 LLM 呼び出しが失敗した際に, 手元のデータ構造から決定論的に組み立てる
    バックアップ要約. 推測は含まず, 観測値のみを並べる.

    Args:
        day (int): Day to summarize / 要約対象の day.
        day_events (list[dict[str, Any]]): ``Agent.day_events`` のリスト.
        talk_history (list[Talk]): Full talk history / 発話履歴.
        whisper_history (list[Talk]): Full whisper history / 囁き履歴.
        self_agent_name (str): Self agent name / 自分のエージェント名.

    Returns:
        str: Plain text structured summary / 構造化されたプレーンテキスト要約.
    """
    lines: list[str] = [f"day: {day} (fallback template — LLM summary unavailable)"]
    lines.extend(_fallback_event_lines(day, day_events, self_agent_name))
    lines.extend(_fallback_talk_lines(day, talk_history, self_agent_name))
    lines.extend(_fallback_whisper_lines(day, whisper_history))
    return "\n".join(lines)
