import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich import box
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.tools.duckduckgo.base import DuckDuckGoSearchToolSpec
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from core.config import (
    MAIN_MODEL, CODE_MODEL, REFACTOR_MODEL, MEMORY_PATH, HF_CACHE_DIR,
    RICH_REFRESH_PER_SECOND, ROUTER_ESCALATION_THRESHOLD,
    FAST_PATH_SIMPLE_CODE, MAX_FINAL_ANSWER_CHARS,
)
from core.memory import init_chat_store, build_memory
from integrations.github_mcp import load_mcp_github_tools
from tools.slurp_tools import slurp_url, slurp_to_obsidian
from tools.summary_tools import summarize_medical_text, summarize_meeting_text
from tools.local_commands import handle_local_command
from ui.dashboard import RichDashboard
from ui.events import extract_thinking_text, extract_tool_output, extract_response_text
from utils.formatting import format_elapsed
from routing.router import TaskRouter
from reasoning.web_reasoner import WebReasoner
from reasoning.code_reasoner import CodeReasoner
from orchestration.handoff import build_handoff_packet
from orchestration.policies import should_escalate
from telemetry.jsonl_logger import JsonlTelemetryLogger
from observability.otel import setup_otel
from metrics.runtime_metrics import RuntimeMetrics

console = Console()
telemetry = JsonlTelemetryLogger()
runtime_metrics = RuntimeMetrics()
tracer, meter = setup_otel("elfagentplus-v4")

@dataclass
class AgentEventRecord:
    source: str
    kind: str
    message: str
    model: Optional[str] = None


def info_panel(message, color="yellow"):
    console.print(Panel(message, border_style=color, box=box.ROUNDED))


def print_rich_banner(tool_count):
    console.print(Panel(
        f"🤖 ElfAgentPlusV4 Initialized\nLoaded tools: {tool_count}\nHugging Face cache: {HF_CACHE_DIR}\nChat/debug model: {MAIN_MODEL}\nCode model: {CODE_MODEL}\nRefactor model: {REFACTOR_MODEL}",
        border_style="bright_blue",
        box=box.ROUNDED,
    ))


def emit_event(ui, source, kind, message, model=None, telemetry_kind="agent_event"):
    record = AgentEventRecord(source=source, kind=kind, message=message, model=model)
    if source == "router":
        style = "cyan"
    elif source in {"web_reasoner", "code_reasoner"}:
        style = "magenta"
    elif source == "main_model":
        style = "green"
    elif source.startswith("tool"):
        style = "yellow"
    else:
        style = "white"
    ui.log(f"[{source}] {message}", style)
    telemetry.write(telemetry_kind, {"source": record.source, "kind": record.kind, "message": record.message, "model": record.model})


def is_simple_code_request(user_msg: str) -> bool:
    q = user_msg.lower()
    simple_markers = ["write a ", "script", "example", "hello world", "program that", "in ruby", "in python", " in c ", "in nim", "zipcode", "zip code", "weather"]
    hard_markers = ["debug", "analyze this codebase", "refactor", "optimize", "architecture", "benchmark", "multi-file", "agent", "workflow"]
    return any(m in q for m in simple_markers) and not any(m in q for m in hard_markers)


def is_complex_refactor_request(user_msg: str) -> bool:
    q = user_msg.lower()
    markers = ["refactor", "rewrite this module", "restructure", "large codebase", "complex migration", "deep cleanup", "multi-file refactor"]
    return any(m in q for m in markers)


def choose_main_model(user_msg: str, route: dict) -> str:
    if is_complex_refactor_request(user_msg):
        return REFACTOR_MODEL
    if route.get("route") == "code_reasoning" or is_simple_code_request(user_msg):
        return CODE_MODEL
    return MAIN_MODEL


async def build_agent(llm_model: str | None = None, preloaded_tools: list | None = None):
    selected_model = llm_model or MAIN_MODEL
    llm = Ollama(model=selected_model, request_timeout=180.0, context_window=16384)
    ddg_spec = DuckDuckGoSearchToolSpec()
    if preloaded_tools is not None:
        tools = preloaded_tools
    else:
        tools = []
        tools.extend(ddg_spec.to_tool_list())
        tools.extend(CodeInterpreterToolSpec().to_tool_list())
        tools.extend([
            FunctionTool.from_defaults(fn=slurp_url, name="slurp_url", description="Fetch a webpage, extract readable main content, convert it to Markdown, and save it locally."),
            FunctionTool.from_defaults(fn=slurp_to_obsidian, name="slurp_to_obsidian", description="Send a webpage URL to the Obsidian Slurp plugin using obsidian://slurp."),
            FunctionTool.from_defaults(fn=summarize_medical_text, name="summarize_medical_text", description="Summarize medical or healthcare text using Falconsai/medical_summarization."),
            FunctionTool.from_defaults(fn=summarize_meeting_text, name="summarize_meeting_text", description="Summarize meeting transcripts or dialogue; automatically chunks long transcripts before summarization."),
        ])
        tools.extend(await load_mcp_github_tools(info_panel))
    chat_store = init_chat_store(MEMORY_PATH)
    memory = build_memory(chat_store)
    SYSTEM_PROMPT = (
        "You are a helpful coding and research assistant. "
        "The code_interpreter tool executes PYTHON ONLY. "
        "For assembly (ARM, RISC-V, x86, PowerPC, MIPS, etc.), C, C++, Nim, Rust, "
        "Zig, Ruby, or any non-Python language, respond DIRECTLY with the code — "
        "do NOT attempt to run it with code_interpreter. "
        "Only use code_interpreter when you need to compute, verify, or run actual Python."
    )
    agent = ReActAgent(tools=tools, llm=llm, memory=memory, verbose=False,
                       system_prompt=SYSTEM_PROMPT)
    return agent, chat_store, tools, ddg_spec


async def process_agent_events(handler, ui: RichDashboard):
    async for event in handler.stream_events():
        event_name = type(event).__name__
        if event_name in {"AgentOutput", "AgentThought", "ReasoningEvent", "ThinkingEvent"}:
            thought_text = extract_thinking_text(event)
            if thought_text:
                source = getattr(event, "source", None) or getattr(event, "model_name", None) or getattr(event, "agent_name", None) or "main_model"
                ui.set_phase(f"Reasoning: {source}")
                ui.add_reasoning(f"[{source}] {thought_text[:1200]}")
                telemetry.write("reasoning_update", {"source": source, "event_name": event_name, "chars": len(thought_text)})
                runtime_metrics.emit("reasoning_chars", len(thought_text), {"source": source})
        elif event_name == "ToolCall":
            ui.archive_reasoning()
            tool_name = getattr(event, "tool_name", "unknown")
            tool_kwargs = getattr(event, "tool_kwargs", {})
            ui.set_phase(f"Using tool: {tool_name}")
            ui.start_tool(tool_name, tool_kwargs)
            telemetry.write("tool_call", {"tool_name": tool_name, "tool_kwargs": str(tool_kwargs)[:1000]})
        elif event_name == "ToolCallResult":
            tool_name = getattr(event, "tool_name", "unknown")
            output_text = extract_tool_output(event)
            ui.set_phase(f"Processed tool: {tool_name}")
            ui.finish_tool(tool_name, output_text)
            telemetry.write("tool_result", {"tool_name": tool_name, "output_preview": output_text[:1000]})
            ui.add_reasoning(f"[tool:{tool_name}] {output_text[:800]}")
        elif event_name in {"AgentStream", "StopEvent"}:
            continue
        else:
            ui.log(f"Event: {event_name}", "white")
            telemetry.write("event", {"event_name": event_name})


async def run_turn_rich(agent, user_msg, ddg_spec, tools):
    started = time.perf_counter()
    span_cm = tracer.start_as_current_span("agent_turn") if tracer else None
    if span_cm:
        span_cm.__enter__()
    try:
        ui = RichDashboard()
        ui.next_turn()
        ui.set_phase("Routing")
        emit_event(ui, "router", "query_start", "New query received")
        telemetry.write("query_start", {"query": user_msg})

        router = TaskRouter()
        route = router.route(user_msg)
        selected_model = choose_main_model(user_msg, route)
        emit_event(ui, "router", "route_selected", f"Selected {route['route']} ({route['confidence']:.2f})", model=router.model_name)
        emit_event(ui, "router", "model_selected", f"Using model {selected_model}", model=selected_model)
        telemetry.write("route_selected", route)
        runtime_metrics.emit("route_confidence", route.get("confidence", 0.0), {"route": route.get("route", "unknown")})

        if FAST_PATH_SIMPLE_CODE and is_simple_code_request(user_msg):
            route = {"route": "fast_code", "confidence": 1.0, "reason": "simple_code_fast_path"}
            emit_event(ui, "router", "fast_path", "Simple code request detected; bypassing helper handoff", model=router.model_name)
            agent_input = user_msg + "\n\nKeep the answer concise. Provide only the code first, then 2-4 short bullets max."
        elif should_escalate(route, ROUTER_ESCALATION_THRESHOLD):
            emit_event(ui, "router", "escalation", "Route confidence below threshold; escalating to main model", model=router.model_name)
            telemetry.write("route_escalation", {"threshold": ROUTER_ESCALATION_THRESHOLD, "route": route})
            agent_input = user_msg
        elif route["route"] == "web_research":
            ui.set_phase("Ultra-light helpers")
            payload = await asyncio.to_thread(WebReasoner(ddg_spec).run, user_msg)
            emit_event(ui, "web_reasoner", "evidence", f"Collected {len(payload.get('evidence', []))} evidence items", model=payload.get("model"))
            handoff = build_handoff_packet(user_msg, route, payload)
            telemetry.write("web_handoff", {"evidence_count": len(payload.get('evidence', [])), "raw_result_count": payload.get('raw_result_count', 0)})
            runtime_metrics.emit("web_evidence_count", len(payload.get("evidence", [])))
            agent_input = f"Use this structured handoff packet as context for your final response. Be concise and avoid repetition.\n\n{handoff}"
        elif route["route"] == "code_reasoning":
            ui.set_phase("Medium reasoning")
            payload = await asyncio.to_thread(CodeReasoner().run, user_msg)
            emit_event(ui, "code_reasoner", "analysis", f"Prepared code analysis with next step {payload.get('next_step')}", model=payload.get("model"))
            handoff = build_handoff_packet(user_msg, route, payload)
            telemetry.write("code_handoff", {"next_step": payload.get('next_step')})
            agent_input = f"Use this structured handoff packet as context for your final response. Keep it short, practical, and non-repetitive.\n\n{handoff}"
        else:
            agent_input = user_msg

        agent_for_turn, _, _, _ = await build_agent(selected_model, preloaded_tools=tools)
        handler = agent_for_turn.run(
            user_msg=agent_input,
            max_iterations=50,
            early_stopping_method="generate",
        )
        response_holder = {"response": None}

        async def run_response():
            try:
                response_holder["response"] = await handler
            except Exception as exc:
                telemetry.write("agent_error", {"error": str(exc)})
                class _Fallback:
                    response = f"(Agent stopped early: {exc})"
                response_holder["response"] = _Fallback()

        event_task = asyncio.create_task(process_agent_events(handler, ui))
        response_task = asyncio.create_task(run_response())

        with Live(ui.render(), console=console, refresh_per_second=RICH_REFRESH_PER_SECOND, screen=True) as live:
            while not response_task.done():
                live.update(ui.render())
                await asyncio.sleep(0.05)
            await event_task
            live.update(ui.render())

        final_text = extract_response_text(response_holder["response"]).strip() or "(No final response text returned.)"
        if len(final_text) > MAX_FINAL_ANSWER_CHARS:
            final_text = final_text[:MAX_FINAL_ANSWER_CHARS].rstrip() + "\n\n[truncated]"

        ui.set_phase("Final answer ready")
        emit_event(ui, "main_model", "final", "Final answer captured", model=selected_model)
        total_elapsed = time.perf_counter() - started
        console.print(Panel(Markdown(f"**Total query time:** {format_elapsed(total_elapsed)}\n\n{final_text}"), title="🤖 Final Answer", border_style="bright_green", box=box.ROUNDED, padding=(0, 1)))
        telemetry.write("final_answer", {"elapsed_s": total_elapsed, "chars": len(final_text), "route": route.get("route"), "model": selected_model})
    finally:
        if span_cm:
            span_cm.__exit__(None, None, None)


async def main():
    agent, _, tools, ddg_spec = await build_agent(MAIN_MODEL)
    print_rich_banner(len(tools))
    while True:
        try:
            user_msg = console.input("[bold cyan]elf_g > [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break
        if not user_msg:
            continue

        cmd_lower = user_msg.strip().lower()
        if cmd_lower in ("/exit", "/quit"):
            console.print("[yellow]Saving memory and exiting...[/yellow]")
            try:
                from core.config import MEMORY_PATH
                agent.memory.chat_store.persist(persist_path=MEMORY_PATH)
            except Exception as e:
                console.print(f"[red]Failed to save memory: {e}[/red]")
            break

        if cmd_lower == "/compact":
            console.print("[yellow]Forcing memory compaction...[/yellow]")
            if hasattr(agent.memory, "force_compact"):
                success, msg = agent.memory.force_compact()
                if success:
                    from core.config import MEMORY_PATH
                    agent.memory.chat_store.persist(persist_path=MEMORY_PATH)
                    console.print(f"[green]Success: {msg}[/green]")
                else:
                    console.print(f"[yellow]{msg}[/yellow]")
            else:
                console.print("[red]Current memory buffer does not support force compaction.[/red]")
            continue

        local = await handle_local_command(user_msg, tools)
        if local == "quit":
            console.print("[yellow]Saving memory and exiting...[/yellow]")
            try:
                from core.config import MEMORY_PATH
                agent.memory.chat_store.persist(persist_path=MEMORY_PATH)
            except Exception:
                pass
            break
        if local == "handled":
            continue
        await run_turn_rich(agent, user_msg, ddg_spec, tools)
