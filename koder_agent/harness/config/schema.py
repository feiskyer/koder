"""Runtime config schema layered on top of the existing Koder config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from koder_agent.config.models import KoderConfig
from koder_agent.harness.reasoning_display import (
    ReasoningDisplayMode,
    normalize_reasoning_display_mode,
)

from .task_delegate_limits import (
    DEFAULT_TASK_DELEGATE_BATCH_SIZE,
    DEFAULT_TASK_DELEGATE_MAX_CONCURRENCY,
    HARD_MAX_TASK_DELEGATE_BATCH_SIZE,
    parse_task_delegate_limit,
)


class HarnessCompanionConfig(BaseModel):
    """Persisted companion soul for the local /buddy runtime."""

    name: str
    personality: str
    hatched_at: int


class HarnessRuntimeConfig(BaseModel):
    """Runtime-owned harness settings."""

    permission_mode: str = Field(default="default")
    teammate_mode: Literal["auto", "tmux", "in-process"] = Field(default="auto")
    last_release_notes_seen: str | None = Field(default=None)
    advisor_model: str | None = Field(default=None)
    brief_mode_enabled: bool = Field(default=False)
    companion: HarnessCompanionConfig | None = Field(default=None)
    companion_muted: bool = Field(default=False)
    reasoning_display: ReasoningDisplayMode = Field(default="off")
    auto_dream_write_mode: Literal["off", "review", "automatic"] = Field(default="review")
    task_delegate_max_batch_size: int = Field(
        default=DEFAULT_TASK_DELEGATE_BATCH_SIZE,
        ge=1,
        le=HARD_MAX_TASK_DELEGATE_BATCH_SIZE,
        description=(
            "Maximum number of tasks accepted by one task_delegate call. "
            "Env: KODER_TASK_DELEGATE_MAX_BATCH_SIZE"
        ),
    )
    task_delegate_max_concurrency: int = Field(
        default=DEFAULT_TASK_DELEGATE_MAX_CONCURRENCY,
        ge=1,
        le=HARD_MAX_TASK_DELEGATE_BATCH_SIZE,
        description=(
            "Maximum number of delegated tasks that may run concurrently. "
            "Env: KODER_TASK_DELEGATE_MAX_CONCURRENCY"
        ),
    )

    @staticmethod
    def _legacy_auto_dream_mode(value: object) -> object:
        if value is True:
            return "review"
        if value is False:
            return "off"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "on", "enabled", "review"}:
                return "review"
            if normalized in {"false", "no", "off", "disabled"}:
                return "off"
        return value

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_auto_dream_config(cls, value: object) -> object:
        if not isinstance(value, dict) or "auto_dream_write_mode" in value:
            return value
        migrated = dict(value)
        for legacy_key in ("auto_dream_enabled", "auto_dream"):
            if legacy_key in migrated:
                migrated["auto_dream_write_mode"] = cls._legacy_auto_dream_mode(
                    migrated[legacy_key]
                )
                break
        return migrated

    @field_validator("reasoning_display", mode="before")
    @classmethod
    def _coerce_reasoning_display(cls, value: object) -> ReasoningDisplayMode:
        return normalize_reasoning_display_mode(value)

    @field_validator(
        "task_delegate_max_batch_size",
        "task_delegate_max_concurrency",
        mode="before",
    )
    @classmethod
    def _parse_task_delegate_limit(cls, value: object, info: ValidationInfo) -> int:
        return parse_task_delegate_limit(
            value,
            source=f"harness.{info.field_name}",
        )

    @model_validator(mode="after")
    def _validate_task_delegate_limits(
        self,
        info: ValidationInfo,
    ) -> "HarnessRuntimeConfig":
        if info.context and info.context.get("defer_task_delegate_limit_relation"):
            return self
        if self.task_delegate_max_concurrency > self.task_delegate_max_batch_size:
            raise ValueError(
                "task_delegate_max_concurrency must be less than or equal to "
                "task_delegate_max_batch_size"
            )
        return self


class RuntimeConfig(KoderConfig):
    """Extended runtime config stored at the existing config path."""

    harness: HarnessRuntimeConfig = Field(default_factory=HarnessRuntimeConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_root_auto_dream_config(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        harness = migrated.get("harness")
        if isinstance(harness, HarnessRuntimeConfig):
            harness_data = harness.model_dump()
        elif isinstance(harness, dict):
            harness_data = dict(harness)
        elif "harness" not in migrated:
            harness_data = {}
        else:
            # Let the declared field validate malformed input; only an absent
            # section receives defaults.
            return migrated
        if "auto_dream_write_mode" not in harness_data:
            for key in ("auto_dream_write_mode", "auto_dream_enabled", "auto_dream"):
                if key in migrated:
                    mode = migrated[key]
                    if key != "auto_dream_write_mode":
                        mode = HarnessRuntimeConfig._legacy_auto_dream_mode(mode)
                    harness_data["auto_dream_write_mode"] = mode
                    break
        migrated["harness"] = harness_data
        return migrated


def parse_runtime_config_source(data: object) -> RuntimeConfig:
    """Parse persisted config before effective environment precedence is known."""
    return RuntimeConfig.model_validate(
        data,
        context={"defer_task_delegate_limit_relation": True},
    )
