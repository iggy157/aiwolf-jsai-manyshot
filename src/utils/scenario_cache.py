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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
        provider, model_id, lang, target_role, prompt_text,
        system_text=system_text, day=day,
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
        provider, model_id, lang, target_role, prompt_text,
        system_text=system_text, day=day,
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
    return path
