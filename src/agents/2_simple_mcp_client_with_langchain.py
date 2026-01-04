from __future__ import annotations

"""
MCP (Model Context Protocol)
----------------------------

MCP is an open standard for how LLM apps talk to external tools, APIs, and data.
It defines:
- How tools are listed and described
- How requests/responses are structured
- How results are returned in a predictable way

So instead of every app inventing its own tool-calling format,
MCP gives everyone the same “language” for LLM ↔ tool communication.


Why we need MCP?
----------------

Before MCP:
- Every integration used custom JSON + custom logic
- Tools weren’t portable between LLM apps
- Lots of duplicated glue code and debugging pain

With MCP:
- Tool authors implement ONE standard interface
- LLM clients support ONE standard protocol
- The same tool can work across many apps without rewrites

MCP = interoperability + portability + less boilerplate.


Analogy: MCP is like a USB port for LLMs
----------------------------------------

Without USB:
- Every device has a different cable + protocol

With USB:
- The OS knows how to discover + talk to any USB device

Now replace devices with tools and the OS with an LLM app:

      +-------------------+
      |     LLM App       |
      +---------+---------+
                |
                |  MCP (standard protocol)
                v
      +-------------------+
      |   MCP Server      |
      +---------+---------+
                |
        +-------+--------+
        |       |        |
     [DB]    [API]   [Files]

The LLM only speaks MCP.
The server handles everything behind it.


Flow in practice (very simple)
------------------------------

1. LLM client connects to an MCP server
2. It asks: “What tools do you expose?”
3. The model decides when to call a tool
4. The server runs it and returns structured results
5. The model keeps reasoning, using those results

So MCP standardizes the *plumbing* —
letting everyone focus on tool logic, not protocol design.
"""



"""
----------------------------------------------------------------------
About this file 2_simple_mcp_client_with_langchain.py
======================================================================
This file demonstrates a simple MCP client that:
1. Connetts to a local MCP server (2_simple_mcp_server.py).
   1.1. Via Stdio.
   1.2. Via Streamable-http.
2. Lists all available tools.
3. Selects a subset of tools and binds them to ChatOllama (gpt-oss:20b).
4. Asks natural-language questions and lets the LLM decide when to call tools.

To run this MCP client, install the MCP SDK:
pip install mcp langchain-ollama langchain-mcp-adapters
"""

import asyncio
from pprint import pprint

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient, MultiServerMCPClient


# ----------------------------------------------------------------------
# 1.1. Configure MultiServerMCPClient to spawn our local MCP server
# ----------------------------------------------------------------------
client = MultiServerMCPClient(
    {
        "local-demo": {
            "command": "python",
            "args": ["2_simple_mcp_server.py"],
            "transport": "stdio",
        }
    }
)

# ----------------------------------------------------------------------
# 1.2. Configure MultiServerMCPClient to connect to our HTTP MCP server
# ----------------------------------------------------------------------
# client = MultiServerMCPClient(
#     {
#         "local-demo": {
#             "transport": "http",  # HTTP (streamable HTTP) transport
#             "url": "http://127.0.0.1:8000/mcp",
#         }
#     }
# )

async def discover_tools():
    """
    Ask the MCP server which tools it has, and print them.
    Returns the list of LangChain tools.
    """
    tools = await client.get_tools()

    print("\n=== Discovered tools from local-demo MCP server ===")
    for t in tools:
        desc = getattr(t, "description", "") or ""
        print(f"- {t.name}: {desc}")

    if not tools:
        raise RuntimeError("No tools discovered from local MCP server.")

    return tools


async def main():
    # ------------------------------------------------------------------
    # 2. Discover tools from the MCP server
    # ------------------------------------------------------------------
    tools = await discover_tools()

    # We'll use a subset with ChatOllama
    desired_names = {"get_weather", "add", "get_time"}
    selected_tools = [t for t in tools if t.name in desired_names]

    print("\n=== Tools selected for the LLM ===")
    for t in selected_tools:
        print(f"- {t.name}")

    if not selected_tools:
        raise RuntimeError(
            "Expected tools (get_weather, add, get_time) not found.\n"
            "Did you modify 2_simple_mcp_server.py?"
        )

    # ------------------------------------------------------------------
    # 3. Initialize ChatOllama with tool-calling
    # ------------------------------------------------------------------
    llm = ChatOllama(
        model="gpt-oss:20b",  # requires `ollama pull gpt-oss:20b`
        temperature=0.2,
        # `format="json"` helps keep tool call arguments structured
        format="json",
    )

    llm_with_tools = llm.bind_tools(selected_tools)

    # ------------------------------------------------------------------
    # 4. Ask questions that should trigger tool calls
    # ------------------------------------------------------------------
    questions = [
        "What is the weather like in New York? Just give a short summary.",
        "Please add 42.5 and 13.7 and explain your reasoning.",
        "What time is it in UTC+2 right now?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print("User:", q)
        print("=" * 80)

        response = await llm_with_tools.ainvoke(q)

        print("\nRaw response object:")
        pprint(response)

        # Show tool calls the model decided to make
        print("\nTool calls from model:")
        for tc in response.tool_calls:
            print(f"- name={tc['name']} args={tc['args']} id={tc['id']}")

        # Actually run the tools that the model requested
        print("\nExecuting tool calls...")
        for tc in response.tool_calls:
            tool = next(t for t in selected_tools if t.name == tc["name"])
            result = await tool.ainvoke(tc["args"])
            print(f"Result from {tc['name']}: {result}")

    # ------------------------------------------------------------------
    # 5. Direct tool call from MCP server (no LLM) -- Optional
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Direct MCP tool call: get_weather(city='London')")
    print("=" * 80)

    weather_tool = next(t for t in tools if t.name == "get_weather")
    direct_result = await weather_tool.ainvoke({"city": "London"})
    pprint(direct_result)



if __name__ == "__main__":
    asyncio.run(main())
