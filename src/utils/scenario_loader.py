"""Load reference game scripts (markdown) and build the scenario prompt body.

お手本対局の Markdown スクリプトを読み込み, 初期化時にLLMへ渡すプロンプト本文を組み立てる.

scenario.delivery=by_day の場合は ``## N日目`` 見出しで本文を章節分割し, Day N の章節
だけを抜き出して渡す API も提供する.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# `## 0日目` `## 1日目` … 形式の見出しを行頭で検出. 数字の幅は 1 桁以上を許容.
_DAY_HEADER_RE = re.compile(r"^## (\d+)日目\s*$", re.MULTILINE)


# 村サイズが「小さい村」(役職減・人狼1人) と判定される閾値. 5人村のみ該当.
_SMALL_VILLAGE_AGENT_NUM = 5


def derive_mechanics_flags(agent_num: int | None) -> dict[str, bool]:
    """Return per-game-mechanics presence flags for the given agent count.

    村サイズから「このゲームに存在する仕組み」を表す真偽値辞書を返す.
    scenario テンプレートが「囁き」「護衛」「霊媒」などの語を出すかどうかを,
    マジックナンバーを散らさずに分岐させるためのフラグ.

    現在の挙動:
        - 5人村: 役職構成は WEREWOLF x1 / POSSESSED x1 / SEER x1 / VILLAGER x2.
          人狼が 1 人なので **囁き (whisper) は発生せず**, 騎士・霊媒も登場しない.
        - 9人村以上: 人狼 2+ 体, 騎士・霊媒も登場するのが標準. 全フラグ True.
        - ``agent_num`` が None や 5 以外の不明値: 9 人村相当 (全 True) でフォールバック.
          知らない構成を想定するより, 全機構をカバーした表現の方が安全.

    Args:
        agent_num: 1ゲームの参加エージェント数 (config.agent.num).

    Returns:
        ``has_whisper`` / ``has_bodyguard`` / ``has_medium`` のフラグ辞書.
    """
    is_small_village = agent_num == _SMALL_VILLAGE_AGENT_NUM
    return {
        "has_whisper": not is_small_village,
        "has_bodyguard": not is_small_village,
        "has_medium": not is_small_village,
    }


def resolve_sample_dir(
    scenario_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    project_root: Path,
) -> Path:
    """Resolve the directory holding scenario markdown for the current agent count.

    起動エージェント数に応じた台本フォルダを解決する.

    解決ルール:
        1. ``scenario.sample_dir`` が config に明示指定されていればそれを尊重 (絶対化のみ).
        2. 未指定なら ``./data/sample_games_md/sample_games_<agent.num>`` を採用.
    """
    explicit = scenario_cfg.get("sample_dir")
    if explicit:
        path = Path(str(explicit))
    else:
        agent_num = int(agent_cfg.get("num", 5))
        path = Path("./data/sample_games_md") / f"sample_games_{agent_num}"
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def resolve_prewarm_identity(
    role: str,
    scenario_cfg: dict[str, Any],
) -> tuple[str, str] | None:
    """Return ``(provider, model_id)`` for cache key when prewarm override exists, else None.

    ``scenario.prewarm.<role>`` に ``type`` と ``model`` の両方が指定されていれば
    その組を返す. 一方でも欠けていれば ``None`` を返し, 呼び出し側で runtime 設定
    (``llm.<role>``) にフォールバックさせる.

    Returns:
        (provider, model_id) tuple if both keys present in ``scenario.prewarm.<role>``;
        otherwise ``None``.
    """
    prewarm = scenario_cfg.get("prewarm") or {}
    role_override = prewarm.get(role) or {}
    type_ = role_override.get("type")
    model = role_override.get("model")
    if type_ and model:
        return str(type_), str(model)
    return None


def is_freeform_enabled(agent_cfg: dict[str, Any]) -> bool:
    """Return whether ``agent.freeform`` is enabled.

    config の ``agent.freeform`` を真偽値として読む. 未指定なら False.
    True のときはキャッシュ・プロンプト・ランタイムが freeform 仕様 (グループチャット
    方式に最適化された次発話者判断・残り発話回数考慮の挙動) に切り替わる.
    """
    return bool(agent_cfg.get("freeform", False))


def resolve_cache_dir(
    scenario_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    project_root: Path,
) -> Path:
    """Resolve the scenario prewarm-cache directory for the current agent count.

    起動エージェント数と freeform フラグに応じたキャッシュフォルダを解決する.

    解決ルール:
        1. ``scenario.cache_dir`` が明示指定されていればそれを尊重 (絶対化のみ).
        2. 未指定なら ``./data/scenario_cache/sample_games_<agent.num>``
           (``agent.freeform: true`` のときは ``sample_games_<num>_freeform``) を採用.
    """
    explicit = scenario_cfg.get("cache_dir")
    if explicit:
        path = Path(str(explicit))
    else:
        agent_num = int(agent_cfg.get("num", 5))
        suffix = "_freeform" if is_freeform_enabled(agent_cfg) else ""
        path = Path("./data/scenario_cache") / f"sample_games_{agent_num}{suffix}"
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def load_scenario_bodies(
    sample_dir: Path,
    glob: str | list[str] = "*.md",
) -> list[str]:
    """Load and return the body text of every matched scenario markdown file, sorted by name.

    指定ディレクトリから glob に一致する Markdown を読み込み, 名前順にソートして本文リストを返す.
    ``glob`` には単一パターンまたはパターンのリストを渡せる. リストの場合は各パターンの
    マッチ結果をマージし, パスの重複は除外する.

    Args:
        sample_dir (Path): Directory that holds the rendered script files / 台本Markdownのディレクトリ
        glob (str | list[str]): Filename glob(s) / 対象ファイルの glob (単一または複数)

    Returns:
        list[str]: Body text per scenario / 台本ごとの本文
    """
    if not sample_dir.exists():
        return []
    patterns: list[str] = [glob] if isinstance(glob, str) else list(glob)
    seen: set[Path] = set()
    matched: list[Path] = []
    for pattern in patterns:
        for path in sample_dir.glob(pattern):
            if path not in seen:
                seen.add(path)
                matched.append(path)
    matched.sort()
    paths: Iterable[Path] = matched
    bodies: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            bodies.append(text)
    return bodies


def split_body_by_day(body: str) -> dict[int, str]:
    """Split a manyshot body into ``{day: chunk}`` keyed by day number.

    `## N日目` 見出し行を境界に Markdown 本文を章節分割する.
    返却される chunk は見出し行を含み, 末尾の空白は除去される.

    Args:
        body (str): 元の Markdown 全文.

    Returns:
        dict[int, str]: day 番号 (int) → 当該日章節 (見出し含む) の dict.
            見出しが 1 件も無ければ空 dict.
    """
    matches = list(_DAY_HEADER_RE.finditer(body))
    if not matches:
        return {}
    chunks: dict[int, str] = {}
    for i, m in enumerate(matches):
        day = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunks[day] = body[start:end].rstrip()
    return chunks


def extract_preamble(body: str) -> str:
    """Return everything before the first ``## N日目`` header (typically ``## 役職配置``).

    Args:
        body (str): 元の Markdown 全文.

    Returns:
        str: 1 件目の day 見出しの直前までの文字列. day 見出しが無ければ全文.
    """
    m = _DAY_HEADER_RE.search(body)
    if m is None:
        return body.rstrip()
    return body[: m.start()].rstrip()


def load_scenario_bodies_by_day(
    sample_dir: Path,
    glob: str | list[str],
    day: int,
    *,
    include_preamble: bool = False,
) -> list[str]:
    """Return per-script chunks for the specified day.

    各 manyshot から指定 day の章節だけを抜き出して返す. ``include_preamble=True``
    のときは各 manyshot の preamble (``## 役職配置`` ブロック等) を当該 chunk の
    冒頭に prepend する (Day 0 を渡すときに役職構成を併せて見せたいときに使う).

    Args:
        sample_dir (Path): manyshot Markdown フォルダ.
        glob (str | list[str]): ファイル glob.
        day (int): 抜き出す day 番号 (0, 1, 2, ...).
        include_preamble (bool): True のとき preamble を chunk 先頭に追加. 既定 False.

    Returns:
        list[str]: 抜粋 chunk のリスト (該当 day を含まない manyshot はスキップ).
    """
    full_bodies = load_scenario_bodies(sample_dir, glob)
    chunks: list[str] = []
    for body in full_bodies:
        days = split_body_by_day(body)
        chunk = days.get(day)
        if chunk is None:
            continue
        if include_preamble:
            preamble = extract_preamble(body)
            if preamble:
                chunk = f"{preamble}\n\n{chunk}"
        chunks.append(chunk)
    return chunks


def discover_available_days(
    sample_dir: Path,
    glob: str | list[str],
) -> list[int]:
    """Return the sorted union of day numbers that appear in any manyshot under sample_dir.

    全 manyshot を走査し, 出現した day 番号の和集合を昇順で返す. by_day モードで
    prewarm すべき day を自動列挙するときに使う.

    Args:
        sample_dir (Path): manyshot Markdown フォルダ.
        glob (str | list[str]): ファイル glob.

    Returns:
        list[int]: 昇順 day 番号 (空なら空 list).
    """
    full_bodies = load_scenario_bodies(sample_dir, glob)
    days: set[int] = set()
    for body in full_bodies:
        days.update(split_body_by_day(body).keys())
    return sorted(days)
