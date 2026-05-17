"""
共享环境变量和配置加载工具
"""
import os
from pathlib import Path
from typing import Dict, Any


def resolve_env(value: str) -> str:
    """解析 ${VAR} 或 ${VAR:-default} 格式的环境变量引用"""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and ":-" in value and value.endswith("}"):
        inner = value[2:-1]
        var_name, default = inner.split(":-", 1)
        return os.environ.get(var_name, default)
    elif value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def load_yaml_config(config_path: str = None) -> Dict[str, Any]:
    """加载 YAML 配置文件，自动解析环境变量引用"""
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if not Path(config_path).exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    return config


def load_dotenv():
    """加载 .env 文件到环境变量"""
    from dotenv import load_dotenv as _load
    env_path = Path(__file__).parent.parent.parent / ".env"
    _load(env_path)
