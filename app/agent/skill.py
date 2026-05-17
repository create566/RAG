"""
Skills 能力扩展
声明式定义 Agent 技能，支持动态加载/切换
"""
from typing import List, Dict, Any, Optional, Callable
from app.models.tool import Skill


class SkillManager:
    """技能管理器"""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._tool_registry: Dict[str, Callable] = {}

    def register_skill(self, skill: Skill):
        """注册技能"""
        self._skills[skill.name] = skill
        print(f"[SKILL] Registered skill: {skill.name}")

    def register_tool(self, name: str, func: Callable):
        """注册工具函数"""
        self._tool_registry[name] = func
        print(f"[SKILL] Registered tool: {name}")

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        """列出所有技能"""
        return list(self._skills.values())

    def list_enabled_skills(self) -> List[Skill]:
        """列出启用的技能"""
        return [s for s in self._skills.values() if s.enabled]

    def get_tools_for_skill(self, skill_name: str) -> List[str]:
        """获取技能关联的工具名列表"""
        skill = self.get(skill_name)
        return skill.tools if skill else []

    def get_tool_function(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        return self._tool_registry.get(name)

    def load_skills_from_config(self, skills_config: List[Dict[str, Any]]):
        """从配置加载技能"""
        for cfg in skills_config:
            skill = Skill(
                name=cfg.get("name", ""),
                description=cfg.get("description", ""),
                tools=cfg.get("tools", []),
                prompt_template=cfg.get("prompt_template", ""),
                enabled=cfg.get("enabled", True),
                config=cfg.get("config", {})
            )
            self.register_skill(skill)


# 全局技能管理器实例
skill_manager = SkillManager()


def get_skill_manager() -> SkillManager:
    """获取技能管理器单例"""
    return skill_manager