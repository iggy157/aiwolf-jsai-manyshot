"""LLM model builder shared between Agent and offline scripts (prewarm, etc).

Agent / オフラインスクリプト (prewarm 等) の両方から使える LLM モデル生成ユーティリティ.
config の provider セクション (openai / google / vertexai / ollama / anthropic) と
ロール別 overrides (llm.talk / llm.action) を受け取り, LangChain の BaseChatModel インスタンスと
料金計算用メタ情報を返す.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# llm.* / llm.talk.* / llm.action.* の中で「provider セクションを上書きする」設定.
# これ以外のキーはロール制御用 (type, sleep_time, separate_langchain, talk, action) か
# 明示的に禁止するキー (api_key) のどちらか.
LLM_OVERRIDE_KEYS: tuple[str, ...] = ("model", "temperature", "pricing_mode", "base_url")


def extract_llm_overrides(
    role_cfg: dict[str, Any],
    *,
    role_name: str,
) -> dict[str, Any]:
    """Extract override fields from an ``llm.*`` / ``llm.talk.*`` / ``llm.action.*`` block.

    config の該当ブロックから <provider> セクションを上書きする項目を抜き出す.
    セキュリティ上の事故防止のため, api_key がここに置かれたら明示的にエラーにする.

    Args:
        role_cfg (dict[str, Any]): llm.* / llm.talk.* / llm.action.* のブロック.
        role_name (str): エラーメッセージで使うロール名 ("talk" / "action" / "").

    Returns:
        dict[str, Any]: provider セクションを上書きする項目のみ抽出した辞書.

    Raises:
        ValueError: api_key が含まれている場合.
    """
    if "api_key" in role_cfg:
        msg = (
            f"api_key must not be set in llm.{role_name}; "
            "use environment variables (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY) instead."
        )
        raise ValueError(msg)
    return {k: role_cfg[k] for k in LLM_OVERRIDE_KEYS if k in role_cfg}


def build_llm_model(
    provider: str,
    provider_section: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> tuple[BaseChatModel, dict[str, str]]:
    """Create an LLM model instance + cost metadata.

    Args:
        provider (str): プロバイダタイプ (openai / google / vertexai / ollama / anthropic).
        provider_section (dict[str, Any]): config のトップレベル <provider>: ブロック.
        overrides (dict[str, Any] | None): llm.*.<provider override keys> で上書きする項目.

    Returns:
        tuple[BaseChatModel, dict[str, str]]: (LLM インスタンス, メタ辞書).
            メタは {provider_key, model_id, pricing_mode} を持つ.
    """
    section: dict[str, Any] = {**(provider_section or {}), **(overrides or {})}
    pricing_mode = str(section.get("pricing_mode", "standard"))
    model_id = str(section.get("model", ""))
    meta = {"provider_key": provider, "model_id": model_id, "pricing_mode": pricing_mode}
    match provider:
        case "openai":
            return (
                ChatOpenAI(
                    model=model_id,
                    temperature=float(section["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                ),
                meta,
            )
        case "google":
            return (
                ChatGoogleGenerativeAI(
                    model=model_id,
                    temperature=float(section["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                ),
                meta,
            )
        case "vertexai":
            return (
                ChatGoogleGenerativeAI(
                    model=model_id,
                    temperature=float(section["temperature"]),
                    vertexai=True,
                ),
                meta,
            )
        case "ollama":
            return (
                ChatOllama(
                    model=model_id,
                    temperature=float(section["temperature"]),
                    base_url=str(section["base_url"]),
                ),
                meta,
            )
        case "anthropic":
            return (
                ChatAnthropic(
                    model_name=model_id,
                    temperature=float(section["temperature"]),
                    timeout=None,
                    stop=None,
                    api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
                ),
                meta,
            )
        case _:
            raise ValueError(provider, "Unknown LLM type")
