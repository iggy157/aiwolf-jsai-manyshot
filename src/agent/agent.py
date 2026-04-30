"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, pass_context
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.cost_logger import append_cost_record, render_markdown, resolve_game_log_dir
from utils.cost_utils import CostRecord, PricingRow, build_record, load_pricing_table
from utils.llm_builder import build_llm_model, extract_llm_overrides
from utils.profile_resolver import load_profile_data, resolve_profile
from utils.scenario_cache import load_cached_response, save_cache_entry
from utils.scenario_loader import (
    load_scenario_bodies,
    resolve_cache_dir,
    resolve_prewarm_identity,
    resolve_sample_dir,
)
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")

_TALK_REQUESTS = {Request.TALK, Request.WHISPER}
_ACTION_REQUESTS = {Request.VOTE, Request.DIVINE, Request.GUARD, Request.ATTACK}
_SHARED_REQUESTS = {Request.INITIALIZE, Request.DAILY_INITIALIZE, Request.DAILY_FINISH}

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
    """Load heading label dictionary from prompts/<lang>/_labels.yml.

    prompts/<lang>/_labels.yml から見出しラベル辞書を読み込む.
    ファイルが無い場合は空辞書を返し, `block()` は name そのものをフォールバックとして使う.
    """
    labels_path = blocks_dir / "_labels.yml"
    if not labels_path.exists():
        return {}
    with labels_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in raw.items()}


def _get_jinja_env(lang: str) -> Environment:
    """Return (and cache) a Jinja2 Environment rooted at prompts/<lang>/.

    prompts/<lang>/ をルートとする Jinja2 Environment を返す (キャッシュ有り).
    言語別ディレクトリが無ければ prompts/ 直下にフォールバックする.
    Environment には block() グローバル関数が登録され, ブロック名 (jinjaファイルの stem)
    から本文レンダ + 見出し付与を1関数で行えるようになる.
    """
    if lang not in _JINJA_ENVS:
        lang_dir = _PROMPTS_ROOT / lang
        blocks_dir = lang_dir if lang_dir.exists() else _PROMPTS_ROOT
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
            """Render <name>.jinja and optionally prepend a heading.

            <name>.jinja を呼び出し側のコンテキストでレンダし, config.headings の設定に応じて
            見出しを前置 (必要なら XML 閉じタグも付加) して返す.
            """
            template = env.get_template(f"{name}.jinja")
            body = template.render(ctx.get_all()).strip()
            headings_cfg = ctx.get("headings") or {}
            if not headings_cfg.get("enabled", False):
                return body
            style = str(headings_cfg.get("style", "markdown"))
            prefix, suffix, has_close = _HEADING_STYLES.get(
                style, _HEADING_STYLES["markdown"],
            )
            label = labels.get(name, name)
            head = f"{prefix}{label}{suffix}"
            if has_close:
                return f"{head}\n{body}\n</{label}>"
            return f"{head}\n{body}"

        env.globals["block"] = block
        _JINJA_ENVS[lang] = env
    return _JINJA_ENVS[lang]

_PRICING_ROOT = Path(__file__).parent.joinpath("./../../data/model_cost").resolve()
# プロセス内で一度だけ料金テーブルをロードして共有する.
_PRICING_TABLE: dict[tuple[str, str, str], PricingRow] = (
    load_pricing_table(_PRICING_ROOT) if _PRICING_ROOT.exists() else {}
)


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role
        # グループチャット方式
        self.in_talk_phase = False
        self.in_whisper_phase = False

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []
        self.llm_model_talk: BaseChatModel | None = None
        self.llm_model_action: BaseChatModel | None = None
        self.llm_message_history_talk: list[BaseMessage] = []
        self.llm_message_history_action: list[BaseMessage] = []
        # single-turnモードで各日のdaily_initialize/daily_finishスナップショットを蓄積する.
        self.day_events: list[dict[str, Any]] = []

        # Cost metadata captured when the model is created.
        # Keys: provider_key (config llm.type), model_id (actual model name), pricing_mode.
        self.llm_meta_default: dict[str, str] | None = None
        self.llm_meta_talk: dict[str, str] | None = None
        self.llm_meta_action: dict[str, str] | None = None
        # LLM呼び出しごとに生成される CostRecord を時系列で蓄積する.
        self.cost_records: list[CostRecord] = []
        # Game IDはINITIALIZEパケット受信時にself.infoから取得できる.
        self.game_id_cache: str = game_id

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    def _is_separate_langchain(self) -> bool:
        """Return whether LangChain instances are separated by request type.

        リクエスト種別ごとにLangChainを分離するかどうかを返す.

        Returns:
            bool: True if separated / 分離している場合はTrue
        """
        llm_config = self.config.get("llm", {})
        return bool(llm_config.get("separate_langchain", False))

    def _is_single_turn(self) -> bool:
        """Return whether the agent is running in single-turn mode.

        single-turnモードで動作しているかを返す.

        Returns:
            bool: True if single-turn / single-turnの場合はTrue
        """
        return str(self.config.get("mode", "multi_turn")) == "single_turn"

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)

        # グループチャット方式
        if packet.new_talk:
            self.talk_history.append(packet.new_talk)
            self.on_talk_received(packet.new_talk)
        if packet.new_whisper:
            self.whisper_history.append(packet.new_whisper)
            self.on_whisper_received(packet.new_whisper)

        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
            self.llm_message_history_talk: list[BaseMessage] = []
            self.llm_message_history_action: list[BaseMessage] = []
            self.day_events = []
        if self.request in (Request.DAILY_INITIALIZE, Request.DAILY_FINISH) and packet.info is not None:
            self.day_events.append(
                {
                    "day": packet.info.day,
                    "phase": self.request.name.lower(),
                    "medium_result": packet.info.medium_result,
                    "divine_result": packet.info.divine_result,
                    "executed_agent": packet.info.executed_agent,
                    "attacked_agent": packet.info.attacked_agent,
                    "vote_list": packet.info.vote_list,
                    "attack_vote_list": packet.info.attack_vote_list,
                },
            )
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def on_talk_received(self, talk: Talk) -> None:
        """Called when a new talk is received (freeform mode).

        新しいトークを受信した時に呼ばれる (グループチャット方式用).

        Args:
            talk (Talk): Received talk / 受信したトーク
        """

    def on_whisper_received(self, whisper: Talk) -> None:
        """Called when a new whisper is received (freeform mode).

        新しい囁きを受信した時に呼ばれる (グループチャット方式用).

        Args:
            whisper (Talk): Received whisper / 受信した囁き
        """

    async def handle_talk_phase(self, send: Callable[[str], None]) -> None:
        """Handle talk phase in freeform mode.

        グループチャット方式でのトークフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_talk_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.talk()
            if not self.in_talk_phase:
                break
            send(text)
            await asyncio.sleep(5)

    async def handle_whisper_phase(self, send: Callable[[str], None]) -> None:
        """Handle whisper phase in freeform mode.

        グループチャット方式での囁きフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_whisper_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.whisper()
            if not self.in_whisper_phase:
                break
            send(text)
            await asyncio.sleep(5)

    def _resolve_targets(
        self,
        request: Request,
    ) -> list[tuple[BaseChatModel, list[BaseMessage], str, dict[str, str] | None]]:
        """Return list of (model, history, label, meta) tuples to send the prompt to.

        プロンプトの送信先 (モデル, 履歴, ラベル, 料金メタ情報) の組を返す.

        Args:
            request (Request): Request type / リクエストタイプ

        Returns:
            list[tuple[BaseChatModel, list[BaseMessage], str, dict[str, str] | None]]:
                Send targets / 送信先のリスト
        """
        if not self._is_separate_langchain():
            if self.llm_model is None:
                return []
            return [(self.llm_model, self.llm_message_history, "default", self.llm_meta_default)]

        targets: list[tuple[BaseChatModel, list[BaseMessage], str, dict[str, str] | None]] = []
        if request in _SHARED_REQUESTS:
            if self.llm_model_talk is not None:
                targets.append(
                    (self.llm_model_talk, self.llm_message_history_talk, "talk", self.llm_meta_talk),
                )
            if self.llm_model_action is not None:
                targets.append(
                    (self.llm_model_action, self.llm_message_history_action, "action", self.llm_meta_action),
                )
        elif request in _TALK_REQUESTS:
            if self.llm_model_talk is not None:
                targets.append(
                    (self.llm_model_talk, self.llm_message_history_talk, "talk", self.llm_meta_talk),
                )
        elif request in _ACTION_REQUESTS:
            if self.llm_model_action is not None:
                targets.append(
                    (self.llm_model_action, self.llm_message_history_action, "action", self.llm_meta_action),
                )
        return targets

    def _record_cost(
        self,
        ai: AIMessage,
        meta: dict[str, str] | None,
        request_key: str,
        label: str,
    ) -> CostRecord | None:
        """Extract token usage from an AIMessage and append a CostRecord.

        AIMessage から token usage を抽出し CostRecord を蓄積する.

        Args:
            ai (AIMessage): LLM response / LLM応答
            meta (dict | None): Model meta info / モデルメタ情報
            request_key (str): Request key / リクエストキー
            label (str): Target label (default/talk/action) / ターゲットラベル

        Returns:
            CostRecord | None: Created record / 生成した CostRecord
        """
        if meta is None:
            return None
        usage_md = getattr(ai, "usage_metadata", None)
        resp_md = getattr(ai, "response_metadata", None)
        record = build_record(
            meta["provider_key"],
            meta["model_id"],
            meta["pricing_mode"],
            usage_md,
            resp_md,
            _PRICING_TABLE,
        )
        record.details = {
            "request_key": request_key,
            "label": label,
            "agent": self.agent_name,
            "game_id": self._current_game_id(),
        }
        self.cost_records.append(record)
        self.agent_logger.logger.info(
            [
                "COST",
                label,
                request_key,
                record.provider,
                record.model_id,
                record.pricing_mode,
                f"in={record.input_tokens}",
                f"cached={record.cached_input_tokens}",
                f"out={record.output_tokens}",
                f"think={record.thinking_tokens}",
                f"usd={record.cost_usd:.6f}",
                f"unknown={record.unknown_pricing}",
            ],
        )
        self._write_cost_json(record, request_key)
        return record

    def _write_cost_json(self, record: CostRecord, request_key: str) -> None:
        """Append a record to the shared cost_summary.json with file locking.

        ロック付きで cost_summary.json に1件追記する. 例外は握りつぶす
        (ログ書き込み失敗でゲーム処理を止めない).

        Args:
            record (CostRecord): Cost record / 料金レコード
            request_key (str): Request key / リクエストキー
        """
        if not bool(self.config.get("log", {}).get("file_output", False)):
            return
        game_id = self._current_game_id()
        if not game_id:
            return
        try:
            cost_dir = resolve_game_log_dir(self.config, game_id)
            append_cost_record(
                cost_dir,
                self.agent_name,
                record,
                request_key,
                game_id,
                str(self.config.get("mode", "multi_turn")),
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to update cost_summary.json")

    def _current_game_id(self) -> str:
        """Return the current game_id (prefer info.game_id, fallback to cache).

        現在の game_id を返す (info.game_id を優先し, なければ初期キャッシュを使う).
        """
        if self.info is not None and getattr(self.info, "game_id", None):
            return str(self.info.game_id)
        return self.game_id_cache

    def _resolve_local_profile(
        self, lang: str,
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        """Return (local_profile, profile_encoding) honoring config.profile.source.

        config.profile.source == "local" のとき, info.agent で data/prompts/profiles.<lang>.yml を
        参照し, マッチすればそのエントリと profile_encoding を返す. マッチしない,
        あるいは source != "local" のときは (None, None) を返す. identity.jinja 側は
        local_profile が None のときサーバ由来の info.profile 文字列にフォールバックする.
        """
        profile_source = str((self.config.get("profile") or {}).get("source", "server"))
        if profile_source != "local":
            return None, None
        agent_name = self.info.agent if self.info is not None else None
        local_profile = resolve_profile(lang, agent_name)
        if local_profile is None:
            return None, None
        _, profile_encoding = load_profile_data(lang)
        return local_profile, profile_encoding

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        is_single_turn = self._is_single_turn()
        # single-turn では共通リクエストはLLMに送らず, day_events等としてコンテキスト保持のみ行う.
        if is_single_turn and request in _SHARED_REQUESTS:
            return None
        request_key = request.lower()
        if request_key not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request_key]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        lang = str(self.config.get("lang", "jp"))
        local_profile, profile_encoding = self._resolve_local_profile(lang)
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "day_events": self.day_events,
            "mode": self.config.get("mode", "multi_turn"),
            "request_key": request_key,
            "headings": self.config.get("headings") or {},
            "local_profile": local_profile,
            "profile_encoding": profile_encoding,
        }
        env = _get_jinja_env(lang)
        template = env.from_string(prompt)
        prompt = template.render(**key).strip()
        targets = self._resolve_targets(request)
        if not targets:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        last_response: str | None = None
        for model, history, label, meta in targets:
            try:
                if is_single_turn:
                    ai = model.invoke([HumanMessage(content=prompt)])
                else:
                    history.append(HumanMessage(content=prompt))
                    ai = model.invoke(history)
                    history.append(ai)
                response = ai.content if isinstance(ai.content, str) else str(ai.content)
                self._record_cost(ai, meta, request_key, label)
                self.agent_logger.logger.info(["LLM", label, prompt, response])
                last_response = response
            except Exception:
                self.agent_logger.logger.exception("Failed to send message to LLM (%s)", label)
                continue
        return last_response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def _create_llm_model(
        self,
        model_type: str,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[BaseChatModel, dict[str, str]]:
        """Thin wrapper around utils.llm_builder.build_llm_model.

        config の provider セクションを base に, ロール側の overrides を上書き適用して
        LLM インスタンスと料金メタを生成する.
        """
        provider_section = self.config.get(model_type, {}) or {}
        return build_llm_model(model_type, provider_section, overrides)

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        INITIALIZE プロンプト送信前に, 設定されていればお手本台本を LLM に読ませる
        (multi-turn モードのみ). separate_langchain=true の場合は talk/action の
        両系統の履歴に同じ台本を積む.
        """
        if self.info is None:
            return

        llm_cfg = self.config["llm"]
        default_type = str(llm_cfg.get("type", ""))

        if self._is_separate_langchain():
            talk_cfg = llm_cfg.get("talk") or {}
            action_cfg = llm_cfg.get("action") or {}
            # type は省略時 llm.type をデフォルトとして使う.
            talk_type = str(talk_cfg.get("type") or default_type)
            action_type = str(action_cfg.get("type") or default_type)
            talk_overrides = extract_llm_overrides(talk_cfg, role_name="talk")
            action_overrides = extract_llm_overrides(action_cfg, role_name="action")
            self.llm_model_talk, self.llm_meta_talk = self._create_llm_model(
                talk_type, talk_overrides,
            )
            self.llm_model_action, self.llm_meta_action = self._create_llm_model(
                action_type, action_overrides,
            )
        else:
            default_overrides = extract_llm_overrides(llm_cfg, role_name="")
            self.llm_model, self.llm_meta_default = self._create_llm_model(
                default_type, default_overrides,
            )

        self._feed_sample_games()
        self._send_message_to_llm(self.request)

    def _feed_sample_games(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Feed reference game scripts to the LLM as pre-initialize context.

        初期化プロンプト送信の前に, お手本台本を LLM に読ませる.
        multi-turn のときのみ llm_message_history に積む. single-turn 時はスキップ.
        separate_langchain=true の場合は talk/action 両系統に同じものを積む.

        scenario.ack_mode:
          - 'llm_summary' (default): 事前 prewarm したキャッシュから LLM 応答を読み込み
            AIMessage として履歴に残す. キャッシュミス時の挙動は on_cache_miss で制御.
          - 'static': 固定の承諾文を AIMessage として積む (API コール無し)

        scenario.on_cache_miss (ack_mode=llm_summary のときのみ有効):
          - 'static' (既定, 推奨): キャッシュ無い場合は static_ack にフォールバック. タイムアウト安全.
          - 'live': 実行時に LLM を呼ぶ (タイムアウトリスクあり. 旧挙動).
          - 'error': 例外を投げて INITIALIZE を失敗させる.
        """
        if self._is_single_turn():
            return
        scenario_cfg = self.config.get("scenario") or {}
        if not bool(scenario_cfg.get("enabled", False)):
            return
        project_root = Path(__file__).resolve().parent.parent.parent
        agent_cfg = self.config.get("agent") or {}
        sample_dir = resolve_sample_dir(scenario_cfg, agent_cfg, project_root)
        # glob は単一文字列でも, リスト (複数パターンをマージ) でも可. 既定はフォルダ内全 .md.
        glob_cfg = scenario_cfg.get("glob", "*.md")
        glob: str | list[str] = (
            list(glob_cfg) if isinstance(glob_cfg, (list, tuple)) else str(glob_cfg)
        )
        bodies = load_scenario_bodies(sample_dir, glob)
        if not bodies:
            self.agent_logger.logger.warning(
                "scenario enabled but no scripts found at %s (glob=%s)",
                sample_dir,
                glob,
            )
            return

        ack_mode = str(scenario_cfg.get("ack_mode", "llm_summary"))
        use_cache = bool(scenario_cfg.get("use_cache", True))
        on_cache_miss = str(scenario_cfg.get("on_cache_miss", "static"))
        cache_dir = resolve_cache_dir(scenario_cfg, agent_cfg, project_root)

        lang = str(self.config.get("lang", "jp"))
        env = _get_jinja_env(lang)
        template = env.get_template("scenario.jinja")
        system_template = env.get_template("scenario_system.jinja")

        static_ack = str(scenario_cfg.get(
            "ack_static_text",
            "承知しました。台本を参考に、議論展開・発話のテンポ・キャラクターの口調を踏まえて演じます。",
        ))

        # separate_langchain の場合は talk と action の両方に積む. そうでない場合は default のみ.
        # target_role は scenario.jinja で要約観点 (発話運び vs アクション判断) を分岐するためのキー.
        if self._is_separate_langchain():
            targets: list[tuple[BaseChatModel | None, list[BaseMessage], str, dict[str, str] | None]] = [
                (self.llm_model_talk, self.llm_message_history_talk, "talk", self.llm_meta_talk),
                (self.llm_model_action, self.llm_message_history_action, "action", self.llm_meta_action),
            ]
        else:
            targets = [(self.llm_model, self.llm_message_history, "default", self.llm_meta_default)]

        for model, history, label, meta in targets:
            # 1. SystemMessage: 本番出力時の運用ルール (target_role 別).
            #    形式の禁止事項などはこちら側に集約し, scenario.jinja は台本提示と要約指示に集中させる.
            system_text = system_template.render(target_role=label).strip()
            history.append(SystemMessage(content=system_text))
            self.agent_logger.logger.info(
                [
                    "SCENARIO",
                    "system",
                    label,
                    f"chars={len(system_text)}",
                    system_text,
                ],
            )
            # 2. HumanMessage: 台本 + 要約指示.
            #    agent_num は scenario.jinja の action 観点を 5p / 9p で切り替えるために渡す.
            prompt = template.render(
                scenario_bodies=bodies,
                scenario_count=len(bodies),
                ack_mode=ack_mode,
                target_role=label,
                agent_num=agent_cfg.get("num"),
                headings=self.config.get("headings") or {},
            ).strip()
            history.append(HumanMessage(content=prompt))
            # 送信した HumanMessage (台本本文込み) を記録. 診断時に台本が実際に渡ったか確認できる.
            self.agent_logger.logger.info(
                [
                    "SCENARIO",
                    "prompt",
                    label,
                    f"count={len(bodies)}",
                    f"ack_mode={ack_mode}",
                    f"chars={len(prompt)}",
                    prompt,
                ],
            )
            if ack_mode == "static":
                history.append(AIMessage(content=static_ack))
                self.agent_logger.logger.info(
                    ["SCENARIO", "ack", label, "static", static_ack],
                )
                continue

            # cache key の (provider, model). scenario.prewarm.<label> があればそちらを優先.
            # (なければ runtime の meta から).
            prewarm_identity = resolve_prewarm_identity(label, scenario_cfg)
            if prewarm_identity is not None:
                cache_provider, cache_model_id = prewarm_identity
            elif meta is not None:
                cache_provider, cache_model_id = meta["provider_key"], meta["model_id"]
            else:
                cache_provider, cache_model_id = "", ""

            # ack_mode = llm_summary: キャッシュヒットを優先. prewarm 済みなら LLM 呼び出しはゼロで完了する.
            cached_response: str | None = None
            if use_cache and (prewarm_identity is not None or meta is not None):
                cached_response = load_cached_response(
                    cache_dir,
                    cache_provider,
                    cache_model_id,
                    lang,
                    label,
                    prompt,
                    system_text=system_text,
                )
            if cached_response is not None:
                history.append(AIMessage(content=cached_response))
                self.agent_logger.logger.info(
                    ["SCENARIO", "ack", label, "cache_hit", cached_response],
                )
                continue

            # キャッシュミス.
            if on_cache_miss == "error":
                msg = (
                    f"Scenario cache miss for target_role={label}, "
                    f"provider={cache_provider or '?'}, "
                    f"model={cache_model_id or '?'}. "
                    "Run `uv run scripts/prewarm_scenario.py` or set scenario.on_cache_miss to "
                    "'static' / 'live'."
                )
                self.agent_logger.logger.error(msg)
                raise RuntimeError(msg)
            if on_cache_miss != "live" or model is None:
                # 既定 (static): LLM を呼ばず静的 ack にフォールバック. タイムアウト安全.
                history.append(AIMessage(content=static_ack))
                self.agent_logger.logger.warning(
                    [
                        "SCENARIO",
                        "ack",
                        label,
                        "cache_miss_static_fallback",
                        "run prewarm_scenario.py to populate cache",
                        static_ack,
                    ],
                )
                continue
            # on_cache_miss == 'live': 実行時に LLM を呼び, 得られた結果をキャッシュにも保存.
            try:
                ai = model.invoke(history)
                history.append(ai)
                self._record_cost(ai, meta, "scenario", label)
                response = ai.content if isinstance(ai.content, str) else str(ai.content)
                if use_cache and (prewarm_identity is not None or meta is not None):
                    try:
                        saved_path = save_cache_entry(
                            cache_dir,
                            cache_provider,
                            cache_model_id,
                            lang,
                            label,
                            prompt,
                            response,
                            system_text=system_text,
                        )
                        self.agent_logger.logger.info(
                            ["SCENARIO", "cache_saved", label, str(saved_path)],
                        )
                    except OSError:
                        self.agent_logger.logger.exception(
                            "Failed to save scenario cache entry",
                        )
                self.agent_logger.logger.info(
                    ["SCENARIO", "ack", label, "live_llm_summary", response],
                )
            except Exception:
                self.agent_logger.logger.exception("Failed to feed scenario to LLM (%s)", label)
                history.append(AIMessage(content=static_ack))
                self.agent_logger.logger.warning(
                    ["SCENARIO", "ack", label, "static_fallback_on_error", static_ack],
                )

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        response = self._send_message_to_llm(Request.TALK)
        self.sent_talk_count = len(self.talk_history)
        return response or ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """
        if not bool(self.config.get("log", {}).get("file_output", False)):
            return
        game_id = self._current_game_id()
        if not game_id:
            return
        try:
            cost_dir = resolve_game_log_dir(self.config, game_id)
            render_markdown(cost_dir)
        except Exception:
            self.agent_logger.logger.exception("Failed to render cost_summary.md")

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
