"""Scenario (sample-game) feed cache.

scenario.jinja をレンダリングしたプロンプトを LLM に投げ, 返された要約を
``(prompt, response)`` ペアとしてローカルキャッシュに保存する.
ゲーム起動時の Agent._feed_sample_games は、このキャッシュを読み込むだけで
LLM 呼び出しを省略し, 初期化タイムアウト問題を根絶する.

キャッシュキーは ``(provider, model_id, lang, target_role, prompt_text)`` の SHA-256.
プロンプトが一文字でも変われば (台本追加 / 文言変更 / target_role 違い) キーが変わるため,
キャッシュの自動無効化も兼ねている.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Markdown レンダリング時にメタデータ表へ出すキー (順序保持).
_META_KEYS: tuple[str, ...] = (
    "provider",
    "model_id",
    "lang",
    "target_role",
    "day",
    "prompt_hash",
    "created_at",
)
# 既定の readable ルート名. ``data/scenario_cache/...`` を
# ``data/scenario_cache_readable/...`` に対応付けるための path 置換キー.
_CACHE_DIR_NAME = "scenario_cache"
_READABLE_DIR_NAME = "scenario_cache_readable"


def compute_cache_key(  # noqa: PLR0913
    provider: str,
    model_id: str,
    lang: str,
    target_role: str,
    prompt_text: str,
    *,
    system_text: str = "",
    day: int | None = None,
) -> str:
    """Compute a stable SHA-256 cache key from the render inputs.

    レンダリング入力に対する安定な SHA-256 キャッシュキーを計算する.

    互換ルール:
        - ``system_text`` が空文字列, かつ ``day`` が None のときは旧仕様 (system / day 無し)
          と同じハッシュを返す. 既存キャッシュ互換を維持.
        - ``day`` を指定すると Day 別キャッシュとして別キーになる (by_day モード用).
        - ``system_text`` を指定するとキャッシュキーに含まれる. 空文字なら旧挙動.
    """
    digest = hashlib.sha256()
    for part in (provider, model_id, lang, target_role):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    if day is not None:
        digest.update(b"DAY:")
        digest.update(str(day).encode("utf-8"))
        digest.update(b"\0")
    if system_text:
        digest.update(b"SYSTEM:")
        digest.update(system_text.encode("utf-8"))
        digest.update(b"\0")
    digest.update(prompt_text.encode("utf-8"))
    return digest.hexdigest()


@dataclass
class CacheEntry:
    """In-memory representation of a cached scenario feed response.

    キャッシュに保存された台本フィード応答のメモリ上表現.
    """

    provider: str
    model_id: str
    lang: str
    target_role: str
    prompt_hash: str
    prompt: str
    response: str
    created_at: str
    system_text: str = ""
    day: int | None = None  # by_day モードのときのみ. None なら full モード.

    def to_dict(self) -> dict[str, str | int | None]:
        """Serialize to a dict suitable for JSON / JSON 出力用 dict に変換."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "lang": self.lang,
            "target_role": self.target_role,
            "prompt_hash": self.prompt_hash,
            "system_text": self.system_text,
            "day": self.day,
            "prompt": self.prompt,
            "response": self.response,
            "created_at": self.created_at,
        }


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def resolve_readable_dir(cache_dir: Path) -> Path | None:
    """Mirror ``data/scenario_cache/<sub>`` to ``data/scenario_cache_readable/<sub>``.

    キャッシュディレクトリのパス中に ``scenario_cache`` セグメントがあれば,
    それを ``scenario_cache_readable`` に差し替えた Path を返す. 見つからなければ
    None (= readable 出力対象外).
    """
    parts = list(cache_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == _CACHE_DIR_NAME:
            parts[i] = _READABLE_DIR_NAME
            return Path(*parts)
    return None


def _day_label(data: Mapping[str, Any]) -> str:
    """Return ``"day<N>"`` if ``day`` is an int, else ``"full"``.

    by_day モードかどうかを 1 単語で表すラベル.
    """
    day = data.get("day")
    if isinstance(day, int):
        return f"day{day}"
    return "full"


def readable_md_filename(data: Mapping[str, Any]) -> str:
    """Return the human-friendly ``.md`` filename for a cache entry.

    パターン: ``<role>_<day_label>__<short_hash>.md``
    例: ``talk_day0__28108c6e.md`` / ``action_full__64410804.md``

    short_hash (先頭 8 文字) を残すのは provider/model 違いで同 role/day の
    キャッシュが複数生成されたときの衝突防止 (実用上は同 cache_dir 内で
    ほぼ同一 provider/model なので衝突しないが, defensive).
    """
    target_role = str(data.get("target_role") or "default")
    safe_role = re.sub(r"[^A-Za-z0-9._-]+", "_", target_role) or "default"
    prompt_hash = str(data.get("prompt_hash") or "")
    short_hash = prompt_hash[:8] or "nohash"
    return f"{safe_role}_{_day_label(data)}__{short_hash}.md"


def render_entry_markdown(data: Mapping[str, Any]) -> str:
    """Render a cache entry dict to human-readable Markdown.

    キャッシュエントリ (CacheEntry.to_dict() または読み込んだ JSON) を
    人間向けの Markdown 文字列に変換する. ``save_cache_entry`` の自動 readable
    生成と ``scripts/render_scenario_cache.py`` の両方から共用される.
    """
    target_role = str(data.get("target_role") or "default")
    provider = str(data.get("provider") or "?")
    model_id = str(data.get("model_id") or "?")
    day = data.get("day")
    day_title = f"Day {day}" if isinstance(day, int) else "Full"

    lines: list[str] = []
    lines.append(
        f"# scenario_cache: {target_role.upper()} · {day_title} · {provider}/{model_id}",
    )
    lines.append("")
    lines.append("## メタデータ")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|-----|-------|")
    for key in _META_KEYS:
        value = data.get(key, "")
        lines.append(f"| {key} | {value if value is not None else ''} |")
    lines.append("")

    system_text = str(data.get("system_text") or "")
    prompt = str(data.get("prompt") or "")
    response = str(data.get("response") or "")

    def _append_section(title: str, body: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- 文字数: {len(body):,}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
        lines.append("---")
        lines.append("")

    if system_text:
        _append_section("system", system_text)
    _append_section("prompt", prompt)
    _append_section("response", response)
    return "\n".join(lines)


def _write_readable_md(cache_dir: Path, data: Mapping[str, Any]) -> Path | None:
    """Best-effort write of a readable Markdown alongside the JSON cache.

    ``cache_dir`` から ``resolve_readable_dir`` で対応する readable ディレクトリを求め,
    ``readable_md_filename(data)`` の名前で書き出す. 失敗してもログのみで例外は
    伝播させない (.json 側の書き込みは成功している前提なので, readable 失敗で
    runtime を壊さない).
    """
    readable_dir = resolve_readable_dir(cache_dir)
    if readable_dir is None:
        return None
    try:
        readable_dir.mkdir(parents=True, exist_ok=True)
        out_path = readable_dir / readable_md_filename(data)
        out_path.write_text(render_entry_markdown(data), encoding="utf-8")
    except OSError:
        logger.exception("Failed to write readable scenario cache markdown")
        return None
    return out_path


def load_cached_response(  # noqa: PLR0913
    cache_dir: Path,
    provider: str,
    model_id: str,
    lang: str,
    target_role: str,
    prompt_text: str,
    *,
    system_text: str = "",
    day: int | None = None,
) -> str | None:
    """Return the cached response text for the given inputs, or None on miss.

    キャッシュに該当エントリがあれば応答テキストを返し, 無ければ None を返す.
    ``system_text`` を指定するとキャッシュキーに含まれる. 空文字なら旧挙動と互換.
    ``day`` を指定すると Day 別キャッシュとして別キーになる (by_day モード用).
    """
    if not cache_dir.exists():
        return None
    key = compute_cache_key(
        provider,
        model_id,
        lang,
        target_role,
        prompt_text,
        system_text=system_text,
        day=day,
    )
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    response = data.get("response")
    if not isinstance(response, str):
        return None
    return response


def save_cache_entry(  # noqa: PLR0913
    cache_dir: Path,
    provider: str,
    model_id: str,
    lang: str,
    target_role: str,
    prompt_text: str,
    response_text: str,
    *,
    system_text: str = "",
    day: int | None = None,
) -> Path:
    """Persist a (prompt, response) pair under cache_dir keyed by SHA-256.

    (prompt, response) を cache_dir に SHA-256 キーで保存し, ファイルパスを返す.
    ``system_text`` を渡した場合はキャッシュキー計算とエントリ本文の双方に反映される.
    ``day`` を渡すと Day 別キャッシュとして別キーになり, エントリ本文にも記録される.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = compute_cache_key(
        provider,
        model_id,
        lang,
        target_role,
        prompt_text,
        system_text=system_text,
        day=day,
    )
    entry = CacheEntry(
        provider=provider,
        model_id=model_id,
        lang=lang,
        target_role=target_role,
        prompt_hash=key,
        prompt=prompt_text,
        response=response_text,
        created_at=datetime.now(UTC).isoformat(),
        system_text=system_text,
        day=day,
    )
    path = _cache_path(cache_dir, key)
    entry_dict = entry.to_dict()
    with path.open("w", encoding="utf-8") as f:
        json.dump(entry_dict, f, ensure_ascii=False, indent=2)
    _write_readable_md(cache_dir, entry_dict)
    return path
