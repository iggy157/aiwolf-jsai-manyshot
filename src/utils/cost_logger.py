"""Shared cost-summary file writer (JSON realtime / Markdown finalize).

コストサマリファイルの書き込みユーティリティ.

複数プロセス (multiprocessing で spawn された各エージェント) が同じ game_id フォルダ
下の cost_summary.json を fcntl でロックして read-modify-write する.
最終的な cost_summary.md は finish 時に生成する.

ディレクトリレイアウトは `utils.agent_logger` と揃え, ULID の timestamp から導出した
`YYYYMMDDHHmmssSSS` フォルダ配下に `<agent>.log` と並べて配置する.
"""

from __future__ import annotations

import fcntl
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ulid import ULID

if TYPE_CHECKING:
    from utils.cost_utils import CostRecord

logger = logging.getLogger(__name__)

JSON_FILENAME = "cost_summary.json"
MD_FILENAME = "cost_summary.md"


def resolve_game_log_dir(config: dict[str, Any], game_id: str) -> Path:
    """Resolve the per-game log directory using the same rule as AgentLogger.

    AgentLogger と同じ規則でゲーム別のログディレクトリを解決する.

    Args:
        config (dict[str, Any]): Config dict / 設定
        game_id (str): ULID game id / ゲームID

    Returns:
        Path: Resolved (and created) directory / 解決済みディレクトリ
    """
    ulid = ULID.from_str(game_id)
    tz = datetime.now(UTC).astimezone().tzinfo
    folder = datetime.fromtimestamp(ulid.timestamp, tz=tz).strftime("%Y%m%d%H%M%S%f")[:-3]
    output_dir = Path(str(config["log"]["output_dir"])) / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _empty_agg() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cost_usd": 0.0,
        "call_count": 0,
        "unknown_pricing_calls": 0,
    }


def _apply(agg: dict[str, Any], record: CostRecord) -> None:
    agg["input_tokens"] += record.input_tokens
    agg["cached_input_tokens"] += record.cached_input_tokens
    agg["output_tokens"] += record.output_tokens
    agg["thinking_tokens"] += record.thinking_tokens
    agg["cost_usd"] += record.cost_usd
    agg["call_count"] += 1
    if record.unknown_pricing:
        agg["unknown_pricing_calls"] += 1


def _update_data(  # noqa: PLR0913
    data: dict[str, Any],
    agent_name: str,
    record: CostRecord,
    request_key: str,
    game_id: str,
    mode: str,
) -> None:
    data["game_id"] = game_id
    data["mode"] = mode
    data["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    _apply(data.setdefault("total", _empty_agg()), record)

    model_key = f"{record.provider}/{record.model_id}/{record.pricing_mode}"
    by_model = data.setdefault("by_model", {})
    _apply(by_model.setdefault(model_key, _empty_agg()), record)

    by_agent = data.setdefault("by_agent", {})
    _apply(by_agent.setdefault(agent_name, _empty_agg()), record)

    by_agent_model = data.setdefault("by_agent_model", {})
    agent_bucket = by_agent_model.setdefault(agent_name, {})
    _apply(agent_bucket.setdefault(model_key, _empty_agg()), record)

    data.setdefault("records", []).append(
        {
            "ts": data["updated_at"],
            "agent": agent_name,
            "request": request_key,
            "label": record.details.get("label") if record.details else None,
            "provider": record.provider,
            "model_id": record.model_id,
            "pricing_mode": record.pricing_mode,
            "input_tokens": record.input_tokens,
            "cached_input_tokens": record.cached_input_tokens,
            "output_tokens": record.output_tokens,
            "thinking_tokens": record.thinking_tokens,
            "cost_usd": record.cost_usd,
            "unknown_pricing": record.unknown_pricing,
        },
    )


def append_cost_record(  # noqa: PLR0913
    cost_dir: Path,
    agent_name: str,
    record: CostRecord,
    request_key: str,
    game_id: str,
    mode: str,
) -> None:
    """Atomically merge a single CostRecord into `cost_summary.json`.

    1件の CostRecord をロック付きで cost_summary.json にマージする.

    Args:
        cost_dir (Path): Game-specific log directory / ゲーム別ログディレクトリ
        agent_name (str): Agent name / エージェント名
        record (CostRecord): Cost record to merge / マージ対象
        request_key (str): Request key (talk/divine/...) / リクエストキー
        game_id (str): Game id (ULID) / ゲームID
        mode (str): multi_turn / single_turn
    """
    json_path = cost_dir / JSON_FILENAME
    cost_dir.mkdir(parents=True, exist_ok=True)
    # Ensure the file exists so "r+" can open it. "a+" is unusable here because
    # O_APPEND forces writes to the end of file even after truncate().
    json_path.touch(exist_ok=True)
    with json_path.open("r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            data: dict[str, Any] = json.loads(content) if content.strip() else {}
            _update_data(data, agent_name, record, request_key, game_id, mode)
            f.seek(0)
            f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def render_markdown(cost_dir: Path) -> None:
    """Read cost_summary.json and (re)write a human-readable cost_summary.md.

    cost_summary.json を読み, 人間可読な cost_summary.md を (再)生成する.
    finish 時に呼び出されることを想定.
    """
    json_path = cost_dir / JSON_FILENAME
    md_path = cost_dir / MD_FILENAME
    if not json_path.exists():
        return
    with json_path.open("r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            if not content.strip():
                return
            data = json.loads(content)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    total = data.get("total") or _empty_agg()
    lines = [
        "# Cost Summary",
        "",
        f"- Game ID: `{data.get('game_id', '')}`",
        f"- Mode: `{data.get('mode', '')}`",
        f"- Updated: `{data.get('updated_at', '')}`",
        "",
        "## Total",
        "",
        f"- Calls: **{total.get('call_count', 0)}** (unknown pricing: {total.get('unknown_pricing_calls', 0)})",
        f"- Tokens: input=**{total.get('input_tokens', 0)}** / cached={total.get('cached_input_tokens', 0)} / "
        f"output={total.get('output_tokens', 0)} / thinking={total.get('thinking_tokens', 0)}",
        f"- Cost: **${total.get('cost_usd', 0.0):.6f} USD**",
        "",
        "## By Model",
        "",
        "| Model | Calls | Input | Cached | Output | Thinking | Cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, agg in sorted(data.get("by_model", {}).items()):
        lines.append(
            f"| `{key}` | {agg['call_count']} | {agg['input_tokens']} | {agg['cached_input_tokens']} | "
            f"{agg['output_tokens']} | {agg['thinking_tokens']} | {agg['cost_usd']:.6f} |",
        )
    lines += [
        "",
        "## By Agent",
        "",
        "| Agent | Calls | Input | Cached | Output | Thinking | Cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for agent, agg in sorted(data.get("by_agent", {}).items()):
        lines.append(
            f"| `{agent}` | {agg['call_count']} | {agg['input_tokens']} | {agg['cached_input_tokens']} | "
            f"{agg['output_tokens']} | {agg['thinking_tokens']} | {agg['cost_usd']:.6f} |",
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
