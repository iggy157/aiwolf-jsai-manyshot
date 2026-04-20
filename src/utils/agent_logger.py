"""Module defining agent logging functionality.

エージェントのログを出力するクラスを定義するモジュール.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ulid import ULID

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Request


class AgentLogger:
    """A class for handling agent logging.

    エージェントのログを出力するクラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
    ) -> None:
        """Initialize the agent logger.

        エージェントのログを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary containing logging settings / ログ設定を含む設定辞書
            name (str): Name of the agent for logging / ログ用のエージェント名
            game_id (str): Game ID for log file organization / ログファイル整理用のゲームID
        """
        self.config = config
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            logging.getLevelNamesMapping()[str(self.config["log"]["level"]).upper()],
        )
        if bool(self.config["log"]["console_output"]):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if bool(self.config["log"]["file_output"]):
            ulid: ULID = ULID.from_str(game_id)
            tz = datetime.now(UTC).astimezone().tzinfo
            output_dir = (
                Path(
                    str(self.config["log"]["output_dir"]),
                )
                / datetime.fromtimestamp(ulid.timestamp, tz=tz).strftime(
                    "%Y%m%d%H%M%S%f",
                )[:-3]
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            handler = logging.FileHandler(
                output_dir / f"{self.name}.log",
                mode="w",
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def packet(self, req: Request | None, res: str | None) -> None:
        """Log packet information.

        パケットのログを出力.

        Args:
            req (Request | None): Request packet to log / ログ出力するリクエストパケット
            res (str | None): Response string to log / ログ出力するレスポンス文字列
        """
        if not req:
            return
        if req.lower() not in self.config["log"]["request"]:
            return
        if not bool(self.config["log"]["request"][req.lower()]):
            return
        if not res:
            self.logger.info([str(req)])
        else:
            self.logger.info([str(req), res])
