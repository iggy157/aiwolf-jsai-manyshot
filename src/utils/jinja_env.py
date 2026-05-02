"""Shared Jinja2 environment factory for prompt rendering.

prompt rendering 用 Jinja2 Environment の共有ファクトリ.
Agent (runtime) / prewarm_scenario.py / preview_prompt.py の 3 箇所で使われ,
全部で同じ ``block()`` 関数 (見出し付与つき) ・ラベル読み込み挙動を共有する.

prompts/<lang>/ をルートとする Environment を返し, グローバル関数 ``block(name)`` を
登録する. ``block(name)`` は ``<name>.jinja`` を現在のコンテキストでレンダし,
``config.headings.enabled`` が True なら ``markdown`` / ``xml`` 形式の見出しを前置する.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, pass_context

_PROMPTS_ROOT = Path(__file__).parent.joinpath("./../../prompts").resolve()
# Jinja2 Environment cache keyed by language code (jp / en).
_JINJA_ENVS: dict[str, Environment] = {}

# 見出しスタイル定義. (prefix, suffix, has_close_tag) の3要素タプル.
# has_close_tag=True のとき本文末尾に </name> を自動付与する.
# 標準エージェントでは最低限の2種類 (markdown / xml) のみを提供する.
_HEADING_STYLES: dict[str, tuple[str, str, bool]] = {
    "markdown": ("### ", "", False),
    "xml": ("<", ">", True),
}


def _load_labels(blocks_dir: Path) -> dict[str, str]:
    """Load heading label dictionary from ``prompts/<lang>/_labels.yml``.

    prompts/<lang>/_labels.yml から見出しラベル辞書を読み込む.
    ファイルが無い場合は空辞書を返し, ``block()`` は ``name`` そのものをフォールバックとして使う.
    """
    labels_path = blocks_dir / "_labels.yml"
    if not labels_path.exists():
        return {}
    with labels_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in raw.items()}


def get_jinja_env(lang: str, *, prompts_root: Path | None = None) -> Environment:
    """Return (and cache) a Jinja2 Environment rooted at ``prompts/<lang>/``.

    prompts/<lang>/ をルートとする Jinja2 Environment を返す (キャッシュ有り).
    言語別ディレクトリが無ければ prompts/ 直下にフォールバックする.
    Environment には ``block()`` グローバル関数が登録され, ブロック名 (jinja ファイルの stem)
    から本文レンダ + 見出し付与を 1 関数で行えるようになる.

    Args:
        lang: 言語コード (``jp`` / ``en``).
        prompts_root: テスト等で別ディレクトリを使いたいときの上書き. None なら
            ``./../../prompts`` (= リポジトリルート/prompts) を採用.

    Returns:
        Jinja2 Environment with ``block`` registered as a global.
    """
    cache_key = lang if prompts_root is None else f"{lang}@{prompts_root}"
    if cache_key in _JINJA_ENVS:
        return _JINJA_ENVS[cache_key]
    root = prompts_root if prompts_root is not None else _PROMPTS_ROOT
    lang_dir = root / lang
    blocks_dir = lang_dir if lang_dir.exists() else root
    env = Environment(
        loader=FileSystemLoader(str(blocks_dir)),
        # プロンプトは LLM に渡すプレーンテキストで HTML ではないため autoescape は無効.
        # 特に xml スタイル見出し ("<history>" など) が HTML エンティティに変換されるのを防ぐ.
        autoescape=False,  # noqa: S701
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=False,
    )
    labels = _load_labels(blocks_dir)

    @pass_context
    def block(ctx: Any, name: str) -> str:  # noqa: ANN401
        """Render ``<name>.jinja`` and optionally prepend a heading.

        ``<name>.jinja`` を呼び出し側のコンテキストでレンダし, ``config.headings`` の設定に応じて
        見出しを前置 (必要なら XML 閉じタグも付加) して返す.
        """
        template = env.get_template(f"{name}.jinja")
        body = template.render(ctx.get_all()).strip()
        headings_cfg = ctx.get("headings") or {}
        if not headings_cfg.get("enabled", False):
            return body
        style = str(headings_cfg.get("style", "markdown"))
        prefix, suffix, has_close = _HEADING_STYLES.get(
            style,
            _HEADING_STYLES["markdown"],
        )
        label = labels.get(name, name)
        head = f"{prefix}{label}{suffix}"
        if has_close:
            return f"{head}\n{body}\n</{label}>"
        return f"{head}\n{body}"

    env.globals["block"] = block
    _JINJA_ENVS[cache_key] = env
    return env
