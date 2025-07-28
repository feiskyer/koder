"""Configuration models for MCP servers."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MCPServerType(str, Enum):
    """Types of MCP servers supported."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str = Field(..., description="Unique name for the server")
    transport_type: MCPServerType = Field(..., description="Transport type (stdio, sse, http)")

    # For stdio servers
    command: Optional[str] = Field(None, description="Command to run for stdio servers")
    args: Optional[List[str]] = Field(
        default_factory=list, description="Arguments for stdio command"
    )
    env_vars: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Environment variables"
    )

    # For SSE and HTTP servers
    url: Optional[str] = Field(None, description="URL for SSE/HTTP servers")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="HTTP headers")

    # Optional configurations
    cache_tools_list: bool = Field(default=False, description="Whether to cache the tools list")
    allowed_tools: Optional[List[str]] = Field(
        None, description="List of allowed tools (allowlist)"
    )
    blocked_tools: Optional[List[str]] = Field(
        None, description="List of blocked tools (blocklist)"
    )

    @field_validator("command")
    @classmethod
    def validate_stdio_command(cls, v: Optional[str], info) -> Optional[str]:
        """Validate that stdio servers have a command."""
        if (
            hasattr(info, "data")
            and info.data.get("transport_type") == MCPServerType.STDIO
            and not v
        ):
            raise ValueError("stdio servers must have a command")
        return v

    @field_validator("url")
    @classmethod
    def validate_remote_url(cls, v: Optional[str], info) -> Optional[str]:
        """Validate that SSE/HTTP servers have a URL."""
        if hasattr(info, "data"):
            transport_type = info.data.get("transport_type")
            if transport_type in [MCPServerType.SSE, MCPServerType.HTTP] and not v:
                raise ValueError(f"{transport_type} servers must have a URL")
        return v

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "name": self.name,
            "transport_type": self.transport_type,
            "command": self.command,
            "args": json.dumps(self.args) if self.args else None,
            "env_vars": json.dumps(self.env_vars) if self.env_vars else None,
            "url": self.url,
            "headers": json.dumps(self.headers) if self.headers else None,
            "cache_tools_list": int(self.cache_tools_list),
            "allowed_tools": json.dumps(self.allowed_tools) if self.allowed_tools else None,
            "blocked_tools": json.dumps(self.blocked_tools) if self.blocked_tools else None,
        }

    @classmethod
    def from_db_dict(cls, data: Dict) -> "MCPServerConfig":
        """Create from database dictionary."""
        # Parse JSON fields
        args = json.loads(data["args"]) if data["args"] else []
        env_vars = json.loads(data["env_vars"]) if data["env_vars"] else {}
        headers = json.loads(data["headers"]) if data["headers"] else {}
        allowed_tools = json.loads(data["allowed_tools"]) if data["allowed_tools"] else None
        blocked_tools = json.loads(data["blocked_tools"]) if data["blocked_tools"] else None

        return cls(
            name=data["name"],
            transport_type=MCPServerType(data["transport_type"]),
            command=data["command"],
            args=args,
            env_vars=env_vars,
            url=data["url"],
            headers=headers,
            cache_tools_list=bool(data["cache_tools_list"]),
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
        )


class MCPAddRequest(BaseModel):
    """Request model for adding MCP servers via CLI."""

    name: str
    transport_type: MCPServerType = MCPServerType.STDIO
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    cache_tools_list: bool = False
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None

    def to_server_config(self) -> MCPServerConfig:
        """Convert to MCPServerConfig."""
        return MCPServerConfig(
            name=self.name,
            transport_type=self.transport_type,
            command=self.command,
            args=self.args,
            env_vars=self.env_vars,
            url=self.url,
            headers=self.headers,
            cache_tools_list=self.cache_tools_list,
            allowed_tools=self.allowed_tools,
            blocked_tools=self.blocked_tools,
        )
