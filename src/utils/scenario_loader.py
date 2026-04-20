"""Load reference game scripts (markdown) and build the scenario prompt body.

お手本対局の Markdown スクリプトを読み込み, 初期化時にLLMへ渡すプロンプト本文を組み立てる.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def load_scenario_bodies(
    sample_dir: Path,
    glob: str | list[str] = "sample_9_*.md",
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
