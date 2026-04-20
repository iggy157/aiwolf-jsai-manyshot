"""Module that defines the Medium agent class.

霊媒師のエージェントクラスを定義するモジュール.
"""

from __future__ import annotations

from typing import Any

from aiwolf_nlp_common.packet import Role

from agent.agent import Agent


class Medium(Agent):
    """Medium agent class.

    霊媒師のエージェントクラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,  # noqa: ARG002
    ) -> None:
        """Initialize the medium agent.

        霊媒師のエージェントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role (ignored, always set to MEDIUM) / 役職(無視され、常にMEDIUMに設定)
        """
        super().__init__(config, name, game_id, Role.MEDIUM)

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return super().talk()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return super().vote()
