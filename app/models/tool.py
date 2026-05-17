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


@dataclass
class Skill:
    """Agent 技能"""
    name: str
    description: str
    tools: List[str] = field(default_factory=list)  # 关联的工具名列表
    prompt_template: str = ""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt_template:
            self.prompt_template = f"你是一个AI助手，擅长{self.description}。"


@dataclass
class SkillRegistry:
    """技能注册中心（运行时）"""
    skills: Dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill):
        """注册技能"""
        self.skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self.skills.get(name)

    def list_all(self) -> List[Skill]:
        """列出所有技能"""
        return list(self.skills.values())

    def list_enabled(self) -> List[Skill]:
        """列出启用的技能"""
        return [s for s in self.skills.values() if s.enabled]

    def get_tool_names(self, skill_name: str) -> List[str]:
        """获取技能关联的工具名列表"""
        skill = self.get(skill_name)
        return skill.tools if skill else []