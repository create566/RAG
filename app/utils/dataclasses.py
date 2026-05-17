"""
UUID <-> int ID 转换工具
"""
import re


def uuid_or_int_to_uuid(value) -> str:
    """将 value 转换为 UUID 字符串（如果已经是，直接返回）"""
    s = str(value)
    if len(s) == 36 and "-" in s:
        return s
    return ""


def uuid_to_int_prefix(uuid_str: str) -> int:
    """UUID 前 8 位 hex -> int"""
    try:
        return int(uuid_str[:8], 16)
    except (ValueError, TypeError):
        return 0


def int_to_uuid_prefix(int_id: int) -> str:
    """int -> 8 位 hex 前缀"""
    return format(int_id & 0xFFFFFFFF, '08x')
