"""
工具和技能数据模型
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MCPTool:
    """MCP 工具"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    endpoint: str = ""
    server_name: str = ""

    def to_agent_tool(self) -> "AgentTool":
        """转换为 AgentTool"""
        from app.agent.react import AgentTool
        return AgentTool(
            name=self.name,
            description=self.description,
            func=None  # MCP 工具需要通过 provider 调用
        )


@dataclass
class MCPManifest:
    """MCP 服务器清单"""
    server_name: str
    tools: List[MCPTool] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    endpoint: str = ""


