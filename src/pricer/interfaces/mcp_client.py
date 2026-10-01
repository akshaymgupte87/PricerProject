"""Small stdio client demonstrating local MCP discovery and tool calls."""

from __future__ import annotations

import asyncio
import json
import sys


async def main() -> None:
    try:
        from mcp.client import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as error:
        raise SystemExit("Install project dependencies to run the MCP client") from error

    server = StdioServerParameters(command=sys.executable, args=["-m", "pricer.interfaces.mcp_server"])
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(json.dumps([tool.name for tool in tools.tools], indent=2))
            result = await session.call_tool("list_opportunities", {"status": ""})
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
