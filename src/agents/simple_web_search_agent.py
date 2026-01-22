#!/usr/bin/env python3
"""
Stronger RAG-style Web QA Agent with LangGraph + Ollama (single file).
For materials goto - https://github.com/AIxorDie/ai-decoded

What this script does
---------------------
This implements a small, composable Web-QA agent that can:
  1) search the web for relevant pages
  2) fetch a few pages
  3) summarize each page *specifically for the user's question*
  4) write a final answer based ONLY on those summaries (RAG-style evidence)

Key idea: "RAG-style" here means we *retrieve* text from the web and feed it back
to the model as context, instead of relying on the model's parametric memory.

How the agent works (high-level)
--------------------------------
We build a LangGraph state machine with two nodes:
  - "agent": the LLM decides what to do next (call a tool, or answer)
  - "tools": executes the requested tool calls and returns tool outputs

The loop continues until the LLM stops asking for tools.

ASCII flow (LangGraph loop)
---------------------------
              +-------------------+
User Question  |   state.messages  |
    +--------->| (System+Human...) |
              +---------+---------+
                        |
                        v
                 +------+------+
                 |   AGENT     |  (LLM w/ tools bound)
                 |  call_model |
                 +------+------+
                        |
          tool_calls?   |    no tool_calls
             +----------+----------+
             |                     |
             v                     v
      +------+------+: tools   +---+---+
      |   TOOLS      |         |  END  |
      | ToolNode     |         +-------+
      +------+------+
             |
             v
     (ToolMessage outputs appended)
             |
             v
          back to AGENT  (loop)

Tool-level flow (retrieval + summarization)
-------------------------------------------
        query
         |
         v
   web_search(query)  ->  [ {title,url}, ... ]
         |
         v
 choose 2-3 URLs
         |
         v
 fetch_and_summarize(url, question)
         |
         +--> fetch_page(url) -> cleaned text
         |
         +--> summarize_for_question(text, question) -> bullet summary
         |
         v
   LLM final answer based on summaries

Notes / guardrails
------------------
- Tools are intentionally small and composable.
- summarize_for_question uses a plain LLM call (no tools) to avoid recursion.
- pretty_run() streams graph steps and prints tool calls + tool outputs.

Dependencies (typical)
----------------------
pip install langgraph langchain-core langchain-ollama langchain-community requests beautifulsoup4
(Also requires Ollama running locally with model pulled, e.g. `ollama pull gpt-oss:20b`.)
"""

from typing import TypedDict, Annotated, Sequence, List, Dict, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# ============================================
# Tools
# ============================================

@tool
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web and return a short list of result metadata.

    Returns:
        List of {"title": str, "url": str}

    Why this exists:
        Keep search separate from fetching/summarizing so the agent can:
          - search broadly
          - then choose a few best sources to fetch
    """
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    except ModuleNotFoundError:
        return [
            {
                "title": "ERROR: missing dependency",
                "url": "",
                "note": "Install `langchain-community` to use web_search.",
            }
        ]

    try:
        wrapper = DuckDuckGoSearchAPIWrapper()
        raw_results = wrapper.results(query, max_results=max_results)
        results: List[Dict[str, Any]] = []
        for r in raw_results:
            # DuckDuckGo wrapper fields can vary; we defensively pick common keys.
            title = r.get("title") or r.get("body") or "No title"
            url = r.get("link") or r.get("href") or ""
            if url:
                results.append({"title": title, "url": url})
        return results
    except Exception as e:
        return [
            {
                "title": "ERROR during search",
                "url": "",
                "note": str(e),
            }
        ]


@tool
def fetch_page(url: str, max_chars: int = 6000) -> str:
    """
    Fetch a single web page and return cleaned plain text.

    max_chars:
        Hard cap on returned text length to keep context manageable.

    Why max_chars matters:
        Web pages can be huge; truncation keeps:
          - tool output small
          - summarization cost bounded
          - model context under control
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as e:
        return (
            "ERROR: missing dependency. Install:\n"
            "  pip install requests beautifulsoup4\n"
            f"Details: {e}"
        )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL {url!r}: {e}"

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")

    # Drop obvious noise that harms summarization quality.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())

    if len(text) > max_chars:
        text = text[:max_chars] + " ... [truncated]"
    return text


@tool
def summarize_for_question(
    page_text: str,
    question: str,
    source: str = "",
) -> str:
    """
    Summarize a page *with respect to* the user question.

    This is where we compress long pages into short, question-focused notes.

    Args:
        page_text: Cleaned text from the page (already truncated).
        question: Original user question.
        source: Optional human-readable source label (URL, title, or both).

    Returns:
        A short, focused summary that the main agent can use as RAG context.

    Important:
        We deliberately use a plain LLM call here (no tools) to avoid
        an infinite loop where the summarizer tries to call tools again.
    """
    if not page_text.strip():
        return "Empty page_text; nothing to summarize."

    # Use a plain LLM instance here (no tools) to avoid recursion.
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="gpt-oss:20b", temperature=0)

    prompt = (
        "You are a helper that summarizes a single web page for a QA system.\n\n"
        f"User question:\n{question}\n\n"
        f"Source (for reference): {source}\n\n"
        "Page text:\n"
        f"{page_text}\n\n"
        "Task:\n"
        "- Write a concise bullet-point summary (5–10 bullets).\n"
        "- Focus ONLY on information that helps answer the question.\n"
        "- If the page is mostly irrelevant, say so.\n"
        "- Do NOT include speculation or outside knowledge.\n"
    )

    resp = llm.invoke(prompt)
    return resp.content


@tool
def fetch_and_summarize(
    url: str,
    question: str,
    max_chars: int = 6000,
    source: str = "",
) -> str:
    """
    Convenience tool: fetch a page and immediately summarize it for a question.

    This is just a thin wrapper around the existing tools:
      1) fetch_page(url, max_chars)
      2) summarize_for_question(page_text, question, source)

    Why this tool exists:
        It reduces the number of tool calls the agent needs to make.
        Instead of:
          fetch_page -> summarize_for_question (two tool calls)
        the agent can do:
          fetch_and_summarize (one tool call)

    Args:
        url: Page URL to fetch.
        question: The user's question to focus the summary.
        max_chars: Passed through to fetch_page to cap page text size.
        source: Optional label for the summarizer (defaults to url if empty).

    Returns:
        Question-focused bullet summary for that page (or an error/irrelevance note).
    """
    # Note: `.invoke({...})` is how LangChain tools are called from Python.
    # When the agent calls tools, LangGraph/ToolNode does this internally.
    page_text = fetch_page.invoke({"url": url, "max_chars": max_chars})

    # If fetch_page returned an error string, just pass it through.
    if isinstance(page_text, str) and page_text.strip().lower().startswith("error"):
        return page_text

    if not source:
        source = url

    return summarize_for_question.invoke(
        {"page_text": page_text, "question": question, "source": source}
    )


# ============================================
# Agent state
# ============================================

class AgentState(TypedDict):
    """
    Conversation state carried around the graph.

    messages:
        Full history of user, AI, and tool messages.

    Why we store messages in state:
        - LLM decisions depend on the conversation context
        - tool outputs become ToolMessages appended to the same history
        - the next LLM call can "see" the tool results
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================
# Core agent logic
# ============================================

def should_continue(state: AgentState) -> str:
    """
    Decide whether to call tools again or stop.

    If last AI message contains tool_calls → 'continue'.
    Else → 'end'.

    In other words:
      - The model "asks" for tools by emitting tool_calls.
      - If it doesn't ask, we assume it's ready to answer.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    return "end"


def call_model(state: AgentState, llm) -> dict:
    """
    Call the main LLM with the conversation so far.

    Returns a dict shaped like the state update LangGraph expects:
      {"messages": [new_message]}
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================
# Agent factory
# ============================================

def create_web_rag_agent(llm):
    """
    Build and compile the LangGraph agent with the tools.

    We bind tools to the LLM so it can:
      - emit tool_calls with structured args
      - and let ToolNode execute them

    Note:
      We keep the tool list small on purpose:
        - web_search: broad retrieval (URLs)
        - fetch_and_summarize: targeted evidence building (page -> summary)
    """
    tools = [web_search, fetch_and_summarize]  # fetch_page, summarize_for_question
    llm_with_tools = llm.bind_tools(tools)

    workflow = StateGraph(AgentState)

    # "agent" node: LLM decides (tool call or final answer)
    workflow.add_node("agent", lambda state: call_model(state, llm_with_tools))

    # "tools" node: executes whatever tool_calls the LLM requested
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")

    # If the agent requested tools, go run them; otherwise stop.
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # After tools run, return control to the agent with ToolMessages appended.
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================
# Pretty printing helpers
# ============================================

def _truncate(text: str, max_len: int = 400) -> str:
    """Shorten long strings for nicer terminal output."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " ... [truncated]"


def pretty_run(agent, query: str):
    """
    Run the agent on one query and show each reasoning/tool step.

    What you will see:
      - When the model chooses tools: tool name + args
      - Tool outputs (truncated)
      - Final answer
    """
    print("\n" + "=" * 80)
    print(f"🧩 Query: {query}")
    print("=" * 80)

    # System message sets the "policy" of how the agent should behave.
    system_msg = SystemMessage(
        content=(
            "You are a web RAG QA agent.\n"
            "- First, use `web_search` to get a small list of relevant URLs.\n"
            "- Then, choose at most 2–3 promising URLs and call `fetch_and_summarize` on them.\n"
            "- Use the summaries as your main evidence.\n"
            "- After you have enough evidence, stop calling tools and answer.\n"
            "- You should also stop if you reach 15 tool calls.\n"
            "- Base your answer ONLY on fetched/summarized content.\n"
            "- When helpful, mention which sources (titles/URLs) you used.\n"
        )
    )

    # Initialize the graph state with the system + user message.
    state = {"messages": [system_msg, HumanMessage(content=query)]}
    prev_len = len(state["messages"])

    print("\n➡️  Step 0: User asks:")
    print(f'   "{query}"')

    last_state = state

    # stream_mode="values" yields successive full states.
    for step_idx, current_state in enumerate(
        agent.stream(state, stream_mode="values"), start=1
    ):
        last_state = current_state
        messages = current_state["messages"]
        new_messages = messages[prev_len:]
        prev_len = len(messages)

        for msg in new_messages:
            # AI step (either tool_calls or final answer)
            if isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    print(f"\n🤖 Step {step_idx}: Agent decides to use tools")
                    for tc in msg.tool_calls:
                        print(f"   • Tool: {tc['name']}")
                        print(f"     Args: {tc['args']}")
                else:
                    print(f"\n✅ Step {step_idx}: Agent produces final answer")
                    print("-" * 80)
                    print(msg.content)
                    print("-" * 80)

            # Tool output step
            elif isinstance(msg, ToolMessage):
                print(f"\n🛠️  Tool '{msg.name}' returned:")
                print("-" * 80)
                print(_truncate(str(msg.content)))
                print("-" * 80)

    # ----------------------------
    # Pretty print of the final answer (this is what we wanted)
    # ----------------------------
    final_message = last_state["messages"][-1]
    tool_calls_so_far = sum(isinstance(m, ToolMessage) for m in last_state["messages"])

    print("🏁"*40)
    print("-" * 80)
    print("AGENT CONCLUDED")
    print("-" * 80)
    print(f"🔧 Tool calls used: {tool_calls_so_far}/15")
    print("-" * 80)

    if isinstance(final_message, AIMessage):
        final_text = final_message.content.strip() or "[Empty final answer]"
        print(final_text)
    else:
        print("[Warning] Final message is not an AIMessage; something is off.")

    print("🏁"*40)
    print("═" * 80 + "\n\n\n\n")


# ============================================
# Main / CLI
# ============================================

def main():
    """
    Run a couple of example queries for quick testing.

    Tip:
      This is mainly a smoke test. For real use, run:
        ./script.py "your question here"
    """
    print("🚀 Initializing Stronger Web RAG Agent with Ollama (gpt-oss:20b)...")

    example_queries = [
        "Does Etihad Airways provides a free hotel for 8 hours layover?",
        "What is LangGraph and what are its main use cases?",
        "What were some of the most notable AI breakthroughs in 2024?",
    ]

    for q in example_queries:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="gpt-oss:20b",
            temperature=0,
        )
        agent = create_web_rag_agent(llm)
        pretty_run(agent, q)


if __name__ == "__main__":
    import sys
    from langchain_ollama import ChatOllama

    # If user provides a CLI query, run just that. Otherwise run examples.
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        agent = create_web_rag_agent(llm)
        pretty_run(agent, user_query)
    else:
        main()
