# 2_simple_mcp_server.py
"""
Local MCP server exposing a few simple tools:

- get_weather(city): fake weather snapshot for a city
- add(a, b): add two numbers
- echo(text): echo text back
- get_time(offset_hours): current UTC time plus optional offset in hours

Transport: stdio (for use with MultiServerMCPClient).
Transport: streamable-http (for use with MultiServerMCPClient).

To run this MCP server standalone (outside of MultiServerMCPClient), install the MCP SDK:
pip install mcp langchain-ollama langchain-mcp-adapters pydantic
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP  # Official MCP Python SDK

# Create the MCP server
mcp = FastMCP("local-demo")


@mcp.tool()
def get_weather(city: str) -> Dict[str, Any]:
    """
    Return a simple fake weather snapshot for a given city.

    This is intentionally dumb (no external API) so it works offline.
    Replace this with a real API call if you want live weather.
    """
    city_normalized = city.strip().title()

    # Toy temperature table just so output isn't identical every time.
    base_temps = {
        "New York": 22,
        "London": 18,
        "San Francisco": 19,
        "Tokyo": 24,
    }
    temperature = base_temps.get(city_normalized, 21)

    return {
        "city": city_normalized,
        "temperature_c": temperature,
        "condition": "Sunny with scattered clouds",
        "source": "local-demo (fake)",
    }


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Echo a string back to the caller (useful for debugging)."""
    return text


@mcp.tool()
def get_time(offset_hours: int = 0) -> str:
    """
    Return the current UTC time plus an optional offset (in hours) as ISO8601.

    Example: offset_hours=5 -> UTC+5
    """
    now_utc = datetime.now(timezone.utc)
    shifted = now_utc + timedelta(hours=offset_hours)
    return shifted.isoformat()


if __name__ == "__main__":
    # Run as a stdio MCP server. The client will spawn this file as a subprocess.
    
    # ----------------------------------------------------------------------
    # 1.1. MultiServerMCPClient will handle the protocol over stdin/stdout.
    # ----------------------------------------------------------------------
    mcp.run(transport="stdio")

    # ----------------------------------------------------------------------
    # 1.2. MultiServerMCPClient will handle the protocol over streamable-http.
    # ----------------------------------------------------------------------
    # mcp.run(transport="streamable-http")
