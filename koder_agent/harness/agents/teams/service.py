"""Filesystem-backed team lifecycle and mailbox service."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from koder_agent.harness.agents.hooks import dispatch_project_hook_event
from koder_agent.harness.agents.messages import AgentMessage
from koder_agent.utils.atomic_file import write_text_atomic

from .memory_sync import TeamMemoryStatus, TeamMemorySyncResult
from .models import TeamHistoryEntry, TeamMailboxMessage, TeamMemberRecord, TeamRecord
from .permission_bridge import PermissionBridge
from .runtime import TEAM_LEAD_NAME, default_tasks_root, default_teams_root
from .task_service import TeamTaskService
from .tmux_backend import TmuxBackend


def _sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).lower()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _locked_team(method):
    """Serialize local lifecycle mutations, including nested mailbox writes."""

    @wraps(method)
    def locked(self, team_id, *args, **kwargs):
        with self._team_lock(team_id):
            return method(self, team_id, *args, **kwargs)

    return locked


class TeamService:
    """Runtime-managed teams persisted under `~/.koder/teams/<team>/`."""

    def __init__(
        self,
        *,
        teams_root: Path | None = None,
        tasks_root: Path | None = None,
        cwd: str | Path | None = None,
        backend: TmuxBackend | None = None,
        permission_bridge: PermissionBridge | None = None,
    ):
        self.teams_root = (teams_root or default_teams_root()).expanduser()
        self.tasks_root = (tasks_root or default_tasks_root()).expanduser()
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.backend = backend
        self.permission_bridge = permission_bridge
        self._team_scope: tuple[str, str] | None = None
        self._member_scope: tuple[str, str, str, str] | None = None
        self.teams_root.mkdir(parents=True, exist_ok=True)
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_test(
        cls,
        *,
        root: Path | None = None,
        cwd: str | Path | None = None,
    ) -> "TeamService":
        if root is None:
            root = Path.cwd() / ".tmp-team-tests"
        return cls(
            teams_root=root / "teams",
            tasks_root=root / "tasks",
            cwd=cwd or root,
        )

    def _team_id(self, name: str) -> str:
        return _sanitize(name)

    def _team_dir(self, team_id: str) -> Path:
        if not team_id or team_id != _sanitize(team_id):
            raise ValueError("Invalid team identifier")
        return self.teams_root / team_id

    def _config_path(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "config.json"

    def _mailbox_dir(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "inboxes"

    def _mailbox_path(self, team_id: str, recipient: str) -> Path:
        return self._mailbox_dir(team_id) / f"{_sanitize(recipient)}.json"

    def _history_path(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "history.jsonl"

    def _runtime_memory_dir(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "memory"

    def _project_memory_dir(self, team_id: str) -> Path:
        return self.cwd / ".koder" / "team-memory" / team_id

    def _team_lock_path(self, team_id: str) -> Path:
        self._team_dir(team_id)  # Validate without resurrecting a deleted directory.
        return self.teams_root / f".{team_id}.lock"

    def _team_lock(self, team_id: str) -> FileLock:
        return FileLock(str(self._team_lock_path(team_id).resolve()), timeout=5, is_singleton=True)

    @contextmanager
    def lifecycle(self, team_id: str, generation: str | None = None):
        """A short synchronous transaction, never held across an await."""
        with self._team_lock(team_id):
            payload = self._read_config(team_id)
            if generation is not None and payload["generation"] != generation:
                raise KeyError(team_id)
            yield

    @_locked_team
    def bind_team(self, team_id: str) -> "TeamService":
        """Fence leader tools to this team lifetime, including delete/recreate."""
        payload = self._read_config(team_id)
        scoped = TeamService(
            teams_root=self.teams_root,
            tasks_root=self.tasks_root,
            cwd=self.cwd,
            backend=self.backend,
            permission_bridge=self.permission_bridge,
        )
        scoped._team_scope = (team_id, payload["generation"])
        scoped._member_scope = self._member_scope
        return scoped

    @_locked_team
    def bind_member(self, team_id: str, agent_id: str) -> "TeamService":
        """Bind run callbacks/tools to this exact team and membership lifetime."""
        payload = self._read_config(team_id)
        member = next((m for m in payload.get("members", []) if m["agent_id"] == agent_id), None)
        if member is None or not member["is_active"]:
            raise KeyError(agent_id)
        scoped = TeamService(
            teams_root=self.teams_root,
            tasks_root=self.tasks_root,
            cwd=self.cwd,
            backend=self.backend,
            permission_bridge=self.permission_bridge,
        )
        scoped._member_scope = (team_id, payload["generation"], agent_id, member["generation"])
        return scoped

    def _read_config(self, team_id: str) -> dict:
        path = self._config_path(team_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise KeyError(team_id) from None
        # Old snapshots remain readable; every newly created lifetime has a UUID.
        payload.setdefault("generation", payload["created_at"])
        for member in payload.get("members", []):
            member.setdefault("generation", member["joined_at"])
        if self._team_scope is not None and self._team_scope != (
            team_id,
            payload["generation"],
        ):
            raise KeyError(team_id)
        if self._member_scope is not None:
            scoped_team, generation, agent_id, membership = self._member_scope
            if (
                team_id != scoped_team
                or payload["generation"] != generation
                or not any(
                    m["agent_id"] == agent_id and m["generation"] == membership and m["is_active"]
                    for m in payload.get("members", [])
                )
            ):
                raise KeyError(agent_id)
        return payload

    def _write_config(self, team_id: str, payload: dict) -> None:
        path = self._config_path(team_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _record_from_payload(self, payload: dict) -> TeamRecord:
        return TeamRecord(
            id=payload["id"],
            name=payload["name"],
            description=payload.get("description"),
            lead_agent_id=payload["lead_agent_id"],
            lead_session_id=payload.get("lead_session_id"),
            config_path=str(self._config_path(payload["id"])),
            created_at=payload["created_at"],
            generation=payload["generation"],
        )

    def get(self, team_id: str) -> TeamRecord:
        payload = self._read_config(team_id)
        return self._record_from_payload(payload)

    def create_team(
        self,
        name: str,
        *,
        description: str | None = None,
        lead_agent_id: str = TEAM_LEAD_NAME,
        lead_session_id: str | None = None,
    ) -> str:
        team_id = self._team_id(name)
        with self._team_lock(team_id):
            if self._config_path(team_id).exists():
                existing = self._read_config(team_id)
                if existing["name"] != name:
                    raise ValueError(f"Team name aliases an existing team: {name}")
                # /peers also uses create_team to ensure its sample team exists.
                # Repeated creation must not reset active members or approvals.
                return team_id
            payload = TeamRecord.create(
                team_id=team_id,
                name=name,
                description=description,
                lead_agent_id=lead_agent_id,
                lead_session_id=lead_session_id,
                config_path=str(self._config_path(team_id)),
            ).__dict__.copy()
            payload["members"] = []
            payload["hidden_pane_ids"] = []
            payload["plan_approvals"] = {}
            payload["shutdown_requests"] = {}
            self._write_config(team_id, payload)
            self._mailbox_dir(team_id).mkdir(parents=True, exist_ok=True)
            TeamTaskService(team_id, root=self.tasks_root, cwd=self.cwd)
        return team_id

    @_locked_team
    def delete_team(self, team_id: str) -> None:
        payload = self._read_config(team_id)
        active_teammates = [
            member for member in payload.get("members", []) if member.get("is_active", True)
        ]
        if active_teammates:
            raise RuntimeError(f"Cannot clean up active team: {team_id}")
        self.task_service(team_id).cleanup()
        team_dir = self._team_dir(team_id)
        if team_dir.exists():
            shutil.rmtree(team_dir)

    def validate_member_name(self, team_id: str, name: str, *, agent_id: str | None = None) -> None:
        """Reject names or IDs that share another member's mailbox address."""
        payload = self._read_config(team_id)
        aliases = {_sanitize(name)}
        if agent_id is not None:
            aliases.add(_sanitize(agent_id))
        reserved = {"", _sanitize(TEAM_LEAD_NAME), _sanitize(payload["lead_agent_id"])}
        if aliases & reserved:
            raise ValueError(f"Reserved team recipient: {name}")
        for member in payload.get("members", []):
            if member["agent_id"] == agent_id:
                continue
            addresses = {_sanitize(member["name"]), _sanitize(member["agent_id"])}
            if aliases & addresses:
                raise ValueError(f"Team recipient already exists: {name}")

    @_locked_team
    def add_member(
        self,
        team_id: str,
        agent_id: str,
        *,
        name: str | None = None,
        agent_type: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        color: str | None = None,
        plan_mode_required: bool = False,
        cwd: str | Path | None = None,
        worktree_path: str | None = None,
        session_id: str | None = None,
        mode: str | None = None,
        is_active: bool = True,
    ) -> TeamMemberRecord:
        self.validate_member_name(team_id, name or agent_id, agent_id=agent_id)
        payload = self._read_config(team_id)
        members = list(payload.get("members", []))
        member_name = name or agent_id
        existing = next((item for item in members if item["agent_id"] == agent_id), None)
        record = TeamMemberRecord.create(
            agent_id=agent_id,
            name=member_name,
            agent_type=agent_type,
            model=model,
            prompt=prompt,
            color=color,
            plan_mode_required=plan_mode_required,
            cwd=str(Path(cwd or self.cwd)),
            worktree_path=worktree_path,
            session_id=session_id,
            mode=mode,
            is_active=is_active,
        )
        if existing is None:
            members.append(record.__dict__.copy())
        else:
            index = members.index(existing)
            members[index] = record.__dict__.copy()
        payload["members"] = members
        if existing is not None:
            # Re-registering even the same ID starts a new membership lifetime.
            self.task_service(team_id).cancel_owner_tasks(agent_id)
        self._write_config(team_id, payload)
        return record

    def set_member_active(self, team_id: str, agent_id: str, is_active: bool) -> None:
        with self.lifecycle(team_id):
            payload = self._read_config(team_id)
            generation = payload["generation"]
            original = next(
                (m for m in payload.get("members", []) if m["agent_id"] == agent_id), None
            )
            if original is None:
                raise KeyError(agent_id)
        if original["is_active"] and not is_active:
            self._dispatch_teammate_idle_hook(team_id, original)
        with self.lifecycle(team_id, generation):
            payload = self._read_config(team_id)
            member = next(
                (m for m in payload.get("members", []) if m["agent_id"] == agent_id), None
            )
            if member is None or member["generation"] != original["generation"]:
                raise KeyError(agent_id)
            if not is_active:
                self.task_service(team_id).cancel_owner_tasks(agent_id)
            if is_active and not member["is_active"]:
                member["joined_at"] = _utc_now_iso()
                member["generation"] = uuid4().hex
            member["is_active"] = is_active
            self._write_config(team_id, payload)

    @_locked_team
    def member_records(self, team_id: str) -> list[TeamMemberRecord]:
        payload = self._read_config(team_id)
        return [TeamMemberRecord(**member) for member in payload.get("members", [])]

    @_locked_team
    def set_member_mode(self, team_id: str, agent_id: str, mode: str) -> TeamMemberRecord:
        payload = self._read_config(team_id)
        for member in payload.get("members", []):
            if member["agent_id"] == agent_id:
                member["mode"] = mode
                self._write_config(team_id, payload)
                return TeamMemberRecord(**member)
        raise KeyError(agent_id)

    @_locked_team
    def set_all_member_modes(self, team_id: str, mode: str) -> list[TeamMemberRecord]:
        payload = self._read_config(team_id)
        updated: list[TeamMemberRecord] = []
        for member in payload.get("members", []):
            member["mode"] = mode
            updated.append(TeamMemberRecord(**member))
        self._write_config(team_id, payload)
        return updated

    def members(self, team_id: str) -> list[str]:
        return [member.agent_id for member in self.member_records(team_id)]

    @_locked_team
    def _append_history_event(self, team_id: str, event: dict) -> None:
        self._read_config(team_id)
        history_path = self._history_path(team_id)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(history_path) + ".lock", timeout=5)
        with lock:
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def history_entries(self, team_id: str) -> list[TeamHistoryEntry]:
        self._read_config(team_id)
        history_path = self._history_path(team_id)
        entries: list[TeamHistoryEntry] = []
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append(
                    TeamHistoryEntry(
                        event=payload.get("event", "message_sent"),
                        created_at=payload.get("created_at", ""),
                        sender=payload.get("sender"),
                        recipient=payload.get("recipient"),
                        content=payload.get("content"),
                        read=payload.get("read"),
                        agent_id=payload.get("agent_id"),
                        member_name=payload.get("member_name"),
                        state=payload.get("state"),
                        source=payload.get("source"),
                    )
                )
            return sorted(entries, key=lambda item: item.created_at)

        for mailbox_path in sorted(self._mailbox_dir(team_id).glob("*.json")):
            try:
                payload = json.loads(mailbox_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in payload:
                entries.append(
                    TeamHistoryEntry(
                        event="message_sent",
                        created_at=item.get("created_at", ""),
                        sender=item.get("sender", TEAM_LEAD_NAME),
                        recipient=item.get("recipient", mailbox_path.stem),
                        content=item.get("content"),
                        read=bool(item.get("read", False)),
                        agent_id=item.get("agent_id"),
                    )
                )
        return sorted(entries, key=lambda item: item.created_at)

    @_locked_team
    def record_run(
        self,
        team_id: str,
        *,
        agent_id: str,
        member_name: str,
        prompt: str,
        output: str,
        state: str,
        source: str | None = None,
    ) -> None:
        if not any(
            member.agent_id == agent_id and member.is_active
            for member in self.member_records(team_id)
        ):
            raise KeyError(agent_id)
        self._append_history_event(
            team_id,
            {
                "event": "run_completed",
                "created_at": _utc_now_iso(),
                "agent_id": agent_id,
                "member_name": member_name,
                "recipient": member_name,
                "content": output,
                "prompt": prompt,
                "state": state,
                "source": source,
            },
        )

    def _dispatch_teammate_idle_hook(self, team_id: str, member: dict) -> None:
        hook_result = dispatch_project_hook_event(
            cwd=self.cwd,
            event_name="TeammateIdle",
            match_value=member.get("name") or member["agent_id"],
            payload={
                "event": "TeammateIdle",
                "team_name": team_id,
                "agent_id": member["agent_id"],
                "agent_name": member.get("name") or member["agent_id"],
                "agent_type": member.get("agent_type"),
            },
        )
        if getattr(hook_result, "blocked", False):
            raise RuntimeError(
                hook_result.block_reason or "Teammate idle transition blocked by hook"
            )

    def notify_member_idle(self, team_id: str, agent_id: str) -> None:
        """Dispatch the teammate-idle hook without removing the teammate from the team."""
        payload = self._read_config(team_id)
        member = next(
            (item for item in payload.get("members", []) if item["agent_id"] == agent_id), None
        )
        if member is None or not member["is_active"]:
            raise KeyError(agent_id)
        self._dispatch_teammate_idle_hook(team_id, member)

    def mailbox_entries(
        self,
        team_id: str,
        *,
        recipient: str = TEAM_LEAD_NAME,
    ) -> list[TeamMailboxMessage]:
        mailbox_path = self._mailbox_path(team_id, recipient)
        if not mailbox_path.exists():
            return []
        payload = json.loads(mailbox_path.read_text(encoding="utf-8"))
        return [
            TeamMailboxMessage(
                agent_id=item["agent_id"],
                content=item["content"],
                created_at=item["created_at"],
                sender=item.get("sender", TEAM_LEAD_NAME),
                recipient=item.get("recipient", recipient),
                read=bool(item.get("read", False)),
            )
            for item in payload
        ]

    @_locked_team
    def consume_next_mailbox_entry(
        self,
        team_id: str,
        *,
        recipient: str,
        preferred_senders: tuple[str, ...] = (TEAM_LEAD_NAME,),
    ) -> TeamMailboxMessage | None:
        """Consume the next unread mailbox entry, prioritizing preferred senders."""
        self._read_config(team_id)
        mailbox_path = self._mailbox_path(team_id, recipient)
        if not mailbox_path.exists():
            return None
        lock = FileLock(str(mailbox_path) + ".lock", timeout=5)
        with lock:
            payload = json.loads(mailbox_path.read_text(encoding="utf-8"))
            selected_index = -1
            for sender in preferred_senders:
                for idx, item in enumerate(payload):
                    if not item.get("read", False) and item.get("sender", TEAM_LEAD_NAME) == sender:
                        selected_index = idx
                        break
                if selected_index != -1:
                    break
            if selected_index == -1:
                for idx, item in enumerate(payload):
                    if not item.get("read", False):
                        selected_index = idx
                        break
            if selected_index == -1:
                return None
            item = payload[selected_index]
            payload[selected_index] = {**item, "read": True}
            write_text_atomic(mailbox_path, json.dumps(payload, indent=2, ensure_ascii=False))
        self._append_history_event(
            team_id,
            {
                "event": "message_read",
                "created_at": _utc_now_iso(),
                "sender": item.get("sender", TEAM_LEAD_NAME),
                "recipient": item.get("recipient", recipient),
                "agent_id": item["agent_id"],
                "content": item["content"],
                "read": True,
            },
        )
        return TeamMailboxMessage(
            agent_id=item["agent_id"],
            content=item["content"],
            created_at=item["created_at"],
            sender=item.get("sender", TEAM_LEAD_NAME),
            recipient=item.get("recipient", recipient),
            read=True,
        )

    @_locked_team
    def route(
        self,
        team_id: str,
        content: str,
        *,
        recipient: str = TEAM_LEAD_NAME,
        sender: str = TEAM_LEAD_NAME,
    ) -> AgentMessage:
        payload = self._read_config(team_id)
        if any(
            _sanitize(sender) in {_sanitize(m["name"]), _sanitize(m["agent_id"])}
            and not m["is_active"]
            for m in payload.get("members", [])
        ):
            raise KeyError(sender)
        mailbox_path = self._mailbox_path(team_id, recipient)
        mailbox_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(mailbox_path) + ".lock", timeout=5)
        with lock:
            if mailbox_path.exists():
                current = json.loads(mailbox_path.read_text(encoding="utf-8"))
            else:
                current = []
            message = AgentMessage.create(agent_id=recipient, content=content)
            envelope = {
                "agent_id": message.agent_id,
                "content": message.content,
                "created_at": message.created_at,
                "sender": sender,
                "recipient": recipient,
                "read": False,
            }
            current.append(envelope)
            write_text_atomic(mailbox_path, json.dumps(current, indent=2, ensure_ascii=False))
        self._append_history_event(
            team_id,
            {
                "event": "message_sent",
                "created_at": message.created_at,
                "sender": sender,
                "recipient": recipient,
                "agent_id": message.agent_id,
                "content": message.content,
                "read": False,
            },
        )
        return message

    def read_mailbox(self, team_id: str, *, recipient: str = TEAM_LEAD_NAME) -> list[AgentMessage]:
        return [
            AgentMessage(
                agent_id=item.agent_id,
                content=item.content,
                created_at=item.created_at,
            )
            for item in self.mailbox_entries(team_id, recipient=recipient)
        ]

    @_locked_team
    def task_service(self, team_id: str) -> TeamTaskService:
        generation = self._read_config(team_id)["generation"]
        return TeamTaskService(
            team_id,
            root=self.tasks_root,
            cwd=self.cwd,
            lifecycle_guard=lambda: self.lifecycle(team_id, generation),
            claimant_is_active=lambda owner: any(
                m.agent_id == owner and m.is_active for m in self.member_records(team_id)
            ),
        )

    def team_memory_status(self, team_id: str) -> TeamMemoryStatus:
        self._read_config(team_id)
        from .memory_sync import team_memory_status

        return team_memory_status(
            team_id=team_id,
            project_dir=self._project_memory_dir(team_id),
            runtime_dir=self._runtime_memory_dir(team_id),
        )

    def sync_team_memory(self, team_id: str) -> TeamMemorySyncResult:
        self._read_config(team_id)
        from .memory_sync import sync_team_memory_dirs

        return sync_team_memory_dirs(
            team_id=team_id,
            project_dir=self._project_memory_dir(team_id),
            runtime_dir=self._runtime_memory_dir(team_id),
        )

    def list_plan_approvals(self, team_id: str) -> dict[str, dict]:
        payload = self._read_config(team_id)
        return dict(payload.get("plan_approvals", {}))

    @_locked_team
    def request_plan_approval(
        self,
        team_id: str,
        *,
        agent_id: str,
        plan: str,
        requested_permission_mode: str = "default",
    ) -> None:
        payload = self._read_config(team_id)
        approvals = dict(payload.get("plan_approvals", {}))
        approvals[agent_id] = {
            "agent_id": agent_id,
            "plan": plan,
            "requested_permission_mode": requested_permission_mode,
        }
        payload["plan_approvals"] = approvals
        self._write_config(team_id, payload)
        self.route(
            team_id,
            f"plan approval requested: {agent_id}",
            recipient=TEAM_LEAD_NAME,
            sender=agent_id,
        )

    @_locked_team
    def respond_plan_approval(
        self,
        team_id: str,
        *,
        agent_id: str,
        approved: bool,
        permission_mode: str | None = None,
        feedback: str | None = None,
    ) -> None:
        payload = self._read_config(team_id)
        approvals = dict(payload.get("plan_approvals", {}))
        approval = approvals.pop(agent_id, None)
        if approval is None:
            raise KeyError(agent_id)
        payload["plan_approvals"] = approvals
        self._write_config(team_id, payload)
        if approved:
            self.set_member_mode(
                team_id,
                agent_id,
                permission_mode or approval.get("requested_permission_mode") or "default",
            )
        message = "plan approved" if approved else "plan rejected"
        if feedback:
            message += f": {feedback}"
        self.route(team_id, message, recipient=agent_id, sender=TEAM_LEAD_NAME)

    def list_shutdown_requests(self, team_id: str) -> dict[str, dict]:
        payload = self._read_config(team_id)
        return dict(payload.get("shutdown_requests", {}))

    @_locked_team
    def request_shutdown(
        self,
        team_id: str,
        *,
        agent_id: str,
        reason: str | None = None,
    ) -> None:
        payload = self._read_config(team_id)
        requests = dict(payload.get("shutdown_requests", {}))
        requests[agent_id] = {
            "agent_id": agent_id,
            "reason": reason or "shutdown requested",
        }
        payload["shutdown_requests"] = requests
        self._write_config(team_id, payload)
        self.route(
            team_id,
            f"shutdown requested: {reason or 'shutdown requested'}",
            recipient=agent_id,
            sender=TEAM_LEAD_NAME,
        )

    async def respond_shutdown(
        self,
        team_id: str,
        *,
        agent_id: str,
        approved: bool,
        feedback: str | None = None,
        agent_service=None,
    ) -> None:
        with self.lifecycle(team_id):
            payload = self._read_config(team_id)
            generation = payload["generation"]
            requests = dict(payload.get("shutdown_requests", {}))
            request = requests.pop(agent_id, None)
            if request is None:
                raise KeyError(agent_id)
            member_generation = next(
                (m["generation"] for m in payload.get("members", []) if m["agent_id"] == agent_id),
                None,
            )
            if approved:
                self.set_member_active(team_id, agent_id, False)
            # Re-read after the membership update; never restore its stale snapshot.
            payload = self._read_config(team_id)
            payload["shutdown_requests"] = requests
            self._write_config(team_id, payload)
        if approved:
            if agent_service is not None:
                try:
                    await agent_service.cancel_background(agent_id)
                except KeyError:
                    pass
        message = "shutdown approved" if approved else "shutdown rejected"
        if feedback:
            message += f": {feedback}"
        with self.lifecycle(team_id, generation):
            if not any(
                m["agent_id"] == agent_id and m["generation"] == member_generation
                for m in self._read_config(team_id).get("members", [])
            ):
                raise KeyError(agent_id)
            self.route(team_id, message, recipient=agent_id, sender=TEAM_LEAD_NAME)
