"""Module providing utility functions for agent operations.

エージェント関連のユーティリティ関数を提供するモジュール.
"""

from typing import Any

from aiwolf_nlp_common.packet import Packet, Role

from agent.agent import Agent
from agent.bodyguard import Bodyguard
from agent.medium import Medium
from agent.possessed import Possessed
from agent.seer import Seer
from agent.villager import Villager
from agent.werewolf import Werewolf

ROLE_TO_AGENT_CLS: dict[Role, type[Agent]] = {
    Role.WEREWOLF: Werewolf,
    Role.POSSESSED: Possessed,
    Role.SEER: Seer,
    Role.BODYGUARD: Bodyguard,
    Role.VILLAGER: Villager,
    Role.MEDIUM: Medium,
}


@staticmethod
def init_agent_from_packet(
    config: dict[str, Any],
    name: str,
    packet: Packet,
) -> Agent:
    """Initialize an agent class corresponding to the role.

    役職に対応するエージェントクラスを初期化する.

    Args:
        config (dict[str, Any]): Configuration dictionary for the agent / エージェントの設定辞書
        name (str): Name of the agent / エージェントの名前
        packet (Packet): Packet containing game information / ゲーム情報を含むパケット

    Returns:
        Agent: Initialized agent instance for the specified role /
                指定された役職用に初期化されたエージェントインスタンス


    Raises:
        ValueError: If packet info or role is not found / パケット情報またはロールが見つからない場合
    """
    if not packet.info:
        raise ValueError(packet.info, "Info not found")
    role = packet.info.role_map.get(packet.info.agent)
    if not role:
        raise ValueError(packet.info, "Role not found")
    return ROLE_TO_AGENT_CLS[role](
        config=config,
        name=name,
        game_id=packet.info.game_id,
        role=role,
    )
