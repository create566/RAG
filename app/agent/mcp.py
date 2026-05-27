"""
MCP (Model Context Protocol) 工具协议
支持从外部 MCP 服务器动态发现和调用工具
"""
import httpx
from typing import List, Dict, Any, Optional
import json
from app.models.tool import MCPTool, MCPManifest
from app.core.logging import get_logger

logger = get_logger(__name__)


class MCPToolProvider:
    """MCP 工具提供者"""

    def __init__(self, servers: List[str] = None):
        self.servers = servers or []
        self._tools: Dict[str, MCPTool] = {}
        self._manifests: Dict[str, MCPManifest] = {}
        self._tool_to_server: Dict[str, str] = {}

    async def discover_tools(self, server_endpoint: str) -> List[MCPTool]:
        """从 MCP 服务器发现工具 (JSON-RPC 2.0)"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # MCP 协议: 发送 initialize 请求
                init_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "clientInfo": {"name": "super-agent", "version": "1.0.0"}
                    }
                }
                resp = await client.post(server_endpoint, json=init_request, headers={"Accept": "text/event-stream"})
                resp = await client.post(server_endpoint, json=init_request)

                # 发送 tools/list 请求
                list_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
                response = await client.post(server_endpoint, json=list_request)

                if response.status_code == 200:
                    data = response.json()
                    tools = data.get("result", {}).get("tools", [])

                    discovered = []
                    for tool in tools:
                        mcp_tool = MCPTool(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            input_schema=tool.get("inputSchema", {}),
                            endpoint=server_endpoint,
                            server_name=self._manifests.get(server_endpoint, MCPManifest(server_name=server_endpoint)).server_name
                        )
                        self._tools[mcp_tool.name] = mcp_tool
                        self._tool_to_server[mcp_tool.name] = server_endpoint
                        discovered.append(mcp_tool)

                    logger.info(f"[MCP] Discovered {len(discovered)} tools from {server_endpoint}")
                    return discovered

        except Exception as e:
            logger.info(f"[MCP] Failed to discover tools from {server_endpoint}: {e}")

        return []

    async def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """调用 MCP 工具 (JSON-RPC 2.0)"""
        server_endpoint = self._tool_to_server.get(tool_name)
        if not server_endpoint:
            return {"error": f"Tool {tool_name} not found"}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                call_request = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": params
                    }
                }
                response = await client.post(server_endpoint, json=call_request)

                if response.status_code == 200:
                    data = response.json()
                    return data.get("result", {})
                else:
                    return {"error": f"HTTP {response.status_code}: {response.text}"}

        except Exception as e:
            return {"error": str(e)}

    def add_server(self, endpoint: str, name: str = ""):
        """添加 MCP 服务器"""
        manifest = MCPManifest(
            server_name=name or endpoint,
            endpoint=endpoint
        )
        self._manifests[endpoint] = manifest
        if endpoint not in self.servers:
            self.servers.append(endpoint)
        logger.info(f"[MCP] Added server: {name or endpoint}")

    async def discover_all_servers(self) -> List[MCPTool]:
        """发现所有服务器上的工具"""
        all_tools = []
        for server in self.servers:
            tools = await self.discover_tools(server)
            all_tools.extend(tools)
        return all_tools

    def list_tools(self) -> List[MCPTool]:
        """列出所有已发现的工具"""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """获取指定工具"""
        return self._tools.get(name)


# 全局 MCP 提供者实例
mcp_provider = MCPToolProvider()


def get_mcp_provider() -> MCPToolProvider:
    """获取 MCP 提供者单例"""
    return mcp_provider