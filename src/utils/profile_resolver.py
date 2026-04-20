"""Load and resolve character profiles from data/prompts/profiles.<lang>.yml.

data/prompts/profiles.<lang>.yml からキャラクタープロフィールを読み込み, エージェント名で解決する.

サーバが custom_profile 有効で稼働しているとき, パケットの info.agent には
キャラクター名 (例: "ミナト" / "Minato") がそのまま入る. この名前をキーに
ローカルの詳細プロフィール YAML を参照し, identity.jinja でリッチに描画する.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DATA_ROOT = Path(__file__).parent.joinpath("./../../data/prompts").resolve()

# lang 単位でキャッシュ. 値は (name -> profile dict, profile_encoding dict) のタプル.
_PROFILE_CACHE: dict[str, tuple[dict[str, dict[str, Any]], dict[str, str]]] = {}


def load_profile_data(lang: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return (profiles_by_name, profile_encoding) for the given language.

    指定言語の (name->profile dict, profile_encoding) を返す. 一度読んだ結果は
    プロセス内でキャッシュする. ファイルが存在しない場合は空辞書を返す.

    Args:
        lang (str): 言語コード (jp / en).

    Returns:
        tuple[dict[str, dict[str, Any]], dict[str, str]]:
            - name -> profile dict のマップ (YAML の profiles リストを name で引ける形に変換)
            - profile_encoding (field -> ラベル) の辞書
    """
    if lang in _PROFILE_CACHE:
        return _PROFILE_CACHE[lang]

    path = _DATA_ROOT / f"profiles.{lang}.yml"
    if not path.exists():
        _PROFILE_CACHE[lang] = ({}, {})
        return _PROFILE_CACHE[lang]

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profiles_list = raw.get("profiles") or []
    by_name: dict[str, dict[str, Any]] = {}
    for entry in profiles_list:
        if isinstance(entry, dict) and "name" in entry:
            by_name[str(entry["name"])] = entry

    encoding_raw = raw.get("profile_encoding") or {}
    encoding = {str(k): str(v) for k, v in encoding_raw.items()}

    _PROFILE_CACHE[lang] = (by_name, encoding)
    return _PROFILE_CACHE[lang]


def resolve_profile(lang: str, agent_name: str | None) -> dict[str, Any] | None:
    """Look up a profile by agent name for the given language.

    指定言語のプロフィール一覧から, agent_name に一致するエントリを返す.
    見つからない, あるいは agent_name が None の場合は None を返す (呼び出し側は
    None の場合サーバ由来の info.profile 文字列にフォールバックする想定).

    Args:
        lang (str): 言語コード (jp / en).
        agent_name (str | None): パケットの info.agent.

    Returns:
        dict[str, Any] | None: マッチしたプロフィール dict, もしくは None.
    """
    if not agent_name:
        return None
    by_name, _ = load_profile_data(lang)
    return by_name.get(agent_name)
