from enum import StrEnum
from datetime import datetime
from pydantic import BaseModel, Field


class SandboxStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "killed"
    PAUSED = "paused"
    ERROR = "error"


class SandboxCreateConfig(BaseModel):
    template: str | None = None
    additional_configs: dict[str, str] = Field(default_factory=dict)


class SandboxUpdateConfig(BaseModel):
    keepalive_longer_by_seconds: int


class SandboxRuntimeInfo(BaseModel):
    sandbox_id: str
    sandbox_status: SandboxStatus
    sandbox_created_at: datetime
    sandbox_expires_at: datetime


class SandboxCommandOutput(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
