"""Load reference game scripts (markdown) and build the scenario prompt body.

お手本対局の Markdown スクリプトを読み込み, 初期化時にLLMへ渡すプロンプト本文を組み立てる.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


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


def resolve_cache_dir(
    scenario_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    project_root: Path,
) -> Path:
    """Resolve the scenario prewarm-cache directory for the current agent count.

    起動エージェント数に応じたキャッシュフォルダを解決する.

    解決ルール:
        1. ``scenario.cache_dir`` が明示指定されていればそれを尊重 (絶対化のみ).
        2. 未指定なら ``./data/scenario_cache/sample_games_<agent.num>`` を採用.
    """
    explicit = scenario_cfg.get("cache_dir")
    if explicit:
        path = Path(str(explicit))
    else:
        agent_num = int(agent_cfg.get("num", 5))
        path = Path("./data/scenario_cache") / f"sample_games_{agent_num}"
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
