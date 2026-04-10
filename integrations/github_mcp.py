import asyncio, traceback
from core.config import ENABLE_GITHUB_MCP, GITHUB_PAT
try:
    from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
    HAS_MCP = True
    MCP_IMPORT_ERROR = None
except ImportError as e:
    BasicMCPClient = None
    McpToolSpec = None
    HAS_MCP = False
    MCP_IMPORT_ERROR = e

async def load_mcp_github_tools(info_callback):
    if not HAS_MCP:
        info_callback(f"MCP support not available in this interpreter. Import error: {MCP_IMPORT_ERROR}", "yellow")
        return []
    if not ENABLE_GITHUB_MCP:
        info_callback("GitHub MCP disabled by ENABLE_GITHUB_MCP=0.", "yellow")
        return []
    if not GITHUB_PAT:
        info_callback("GitHub MCP disabled: GITHUB_PERSONAL_ACCESS_TOKEN is missing in .env", "yellow")
        return []
    try:
        client = BasicMCPClient("docker", args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"], env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PAT})
        tool_spec = McpToolSpec(client=client)
        try:
            tools = await asyncio.wait_for(tool_spec.to_tool_list_async(), timeout=25)
        except asyncio.TimeoutError:
            info_callback("GitHub MCP timed out while loading tools. Continuing without GitHub tools.", "yellow")
            return []
        info_callback(f"Loaded {len(tools)} GitHub MCP tool(s).", "green")
        return tools
    except Exception as e:
        info_callback("GitHub MCP failed to load. Continuing without GitHub tools.\n\n" + f"{e}\n\n{traceback.format_exc(limit=8)}", "red")
        return []
