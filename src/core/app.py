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
    FAST_PATH_SIMPLE_CODE, MAX_FINAL_ANSWER_CHARS, PLAN_CHECKPOINT_PATH,
)
from core.engram import EngramStore
from core.memory import init_chat_store, build_memory
from integrations.github_mcp import load_mcp_github_tools
from tools.slurp_tools import slurp_url, slurp_to_obsidian
from tools.summary_tools import summarize_medical_text, summarize_meeting_text
from tools.local_commands import handle_local_command
from tools.search_tools import web_search, wikipedia_search, brave_search
from tools.file_tools import read_file, write_file, list_directory, grep_files
from tools.shell_tools import run_shell_command, git_status, git_log
from tools.datetime_tools import get_current_datetime, get_timezone_time
from tools.doc_tools import parse_pdf, extract_pdf_tables
from tools.http_tools import http_request
from tools.patch_tools import apply_patch, create_diff
from tools.rag_tools import index_directory, semantic_search
from tools.structured_tools import parse_structured_file
from tools.system_tools import system_info, list_serial_ports
from tools.vision_tools import describe_image, ocr_image
from tools.ast_tools import python_ast_query
from tools.symbol_tools import list_c_symbols, list_asm_symbols, find_symbol_references, read_binary_symbols
from ui.dashboard import RichDashboard
from ui.events import extract_thinking_text, extract_tool_output, extract_response_text
from utils.formatting import format_elapsed
from routing.router import TaskRouter
from routing.heuristics import is_complex_refactor_request, is_simple_code_request
from reasoning.critic_reasoner import CriticReasoner
from reasoning.task_planner import (
    Plan,
    PlanExecutor,
    Step,
    StepResult,
    build_default_plan,
    render_plan_status,
    should_use_task_planner,
)
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
engram_store = EngramStore()
critic_reasoner = CriticReasoner()
SYSTEM_PROMPT = (
    "You are a helpful coding and research assistant. "
    "The code_interpreter tool executes PYTHON ONLY. "
    "For assembly (ARM, RISC-V, x86, PowerPC, MIPS, etc.), C, C++, Nim, Rust, "
    "Zig, Ruby, or any non-Python language, respond DIRECTLY with the code — "
    "do NOT attempt to run it with code_interpreter. "
    "Only use code_interpreter when you need to compute, verify, or run actual Python.\n"
    "Use web_search for general web queries; it automatically falls back across "
    "multiple search engines if one is rate-limited.  "
    "Use wikipedia_search for factual / encyclopaedic lookups.  "
    "Use read_file / list_directory / grep_files to inspect project files.  "
    "Use run_shell_command for builds, tests, and git operations (allowlisted).  "
    "Use get_current_datetime when you need today's date or current time."
)


@dataclass
class AgentEventRecord:
    source: str
    kind: str
    message: str
    model: Optional[str] = None


def info_panel(message, color="yellow"):
    console.print(Panel(message, border_style=color, box=box.ROUNDED))


def build_runtime_profile(startup_model: str, tool_count: int, ddg_spec) -> dict:
    return {
        "tool_count": tool_count,
        "hf_cache_dir": HF_CACHE_DIR,
        "startup_model": startup_model,
        "router_model": TaskRouter.model_name,
        "web_model": WebReasoner.model_name if hasattr(WebReasoner, "model_name") else "(web reasoner)",
        "code_model": CodeReasoner.model_name if hasattr(CodeReasoner, "model_name") else CODE_MODEL,
        "refactor_model": REFACTOR_MODEL,
    }


def print_rich_banner(runtime_profile: dict):
    console.print(Panel(
        "🤖 ElfAgentPlusV4 Initialized\n"
        f"Loaded tools: {runtime_profile['tool_count']}\n"
        f"Hugging Face cache: {runtime_profile['hf_cache_dir']}\n"
        f"Startup model: {runtime_profile['startup_model']}\n"
        f"Router model: {runtime_profile['router_model']}\n"
        f"Web helper model: {runtime_profile['web_model']}\n"
        f"Code helper model: {runtime_profile['code_model']}\n"
        f"Refactor model: {runtime_profile['refactor_model']}",
        border_style="bright_blue",
        box=box.ROUNDED,
    ))


def emit_event(ui, source, kind, message, model=None, telemetry_kind="agent_event"):
    record = AgentEventRecord(source=source, kind=kind, message=message, model=model)
    style = {
        "router": "cyan",
        "web_reasoner": "magenta",
        "code_reasoner": "magenta",
        "main_model": "green",
    }.get(source, "yellow" if source.startswith("tool") else "white")
    ui.log(f"[{source}] {message}", style)
    telemetry.write(telemetry_kind, {"source": record.source, "kind": record.kind, "message": record.message, "model": record.model})


def choose_main_model(user_msg: str, route: dict) -> str:
    if is_complex_refactor_request(user_msg):
        return REFACTOR_MODEL
    if route.get("route") == "code_reasoning" or is_simple_code_request(user_msg):
        return CODE_MODEL
    return MAIN_MODEL


def apply_engram_context(prompt: str, engram_matches: list[dict]) -> str:
    memory_block = engram_store.format_for_prompt(engram_matches)
    if not memory_block:
        return prompt
    return f"{memory_block}\n\nCurrent request:\n{prompt}"


async def build_agent(
    llm_model: str | None = None,
    preloaded_tools: list | None = None,
    chat_store=None,
    memory=None,
):
    selected_model = llm_model or MAIN_MODEL
    llm = Ollama(model=selected_model, request_timeout=180.0, context_window=16384)
    ddg_spec = DuckDuckGoSearchToolSpec()
    if preloaded_tools is not None:
        tools = preloaded_tools
    else:
        tools = []
        # --- Legacy DDG tools (kept for compatibility; web_search supersedes them) ---
        tools.extend(ddg_spec.to_tool_list())
        # --- Code interpreter ---
        tools.extend(CodeInterpreterToolSpec().to_tool_list())
        # --- Multi-engine search ---
        tools.append(FunctionTool.from_defaults(
            fn=web_search, name="web_search",
            description="Search the web using multiple engines (Brave, SearXNG, DuckDuckGo, Wikipedia) with automatic rate-limit failover."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=wikipedia_search, name="wikipedia_search",
            description="Search Wikipedia and return article summaries for factual and encyclopaedic queries."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=brave_search, name="brave_search",
            description="Search using the Brave Search API (requires BRAVE_API_KEY env var)."
        ))
        # --- Web fetch / slurp ---
        tools.append(FunctionTool.from_defaults(
            fn=slurp_url, name="slurp_url",
            description="Fetch a webpage, extract readable main content, convert it to Markdown, and save it locally."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=slurp_to_obsidian, name="slurp_to_obsidian",
            description="Send a webpage URL to the Obsidian Slurp plugin using obsidian://slurp."
        ))
        # --- Summarisation ---
        tools.append(FunctionTool.from_defaults(
            fn=summarize_medical_text, name="summarize_medical_text",
            description="Summarize medical or healthcare text using Falconsai/medical_summarization."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=summarize_meeting_text, name="summarize_meeting_text",
            description="Summarize meeting transcripts or dialogue; automatically chunks long transcripts."
        ))
        # --- File tools ---
        tools.append(FunctionTool.from_defaults(
            fn=read_file, name="read_file",
            description="Read a text file from disk and return its contents."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=write_file, name="write_file",
            description="Write content to a file on disk, creating parent directories as needed."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=list_directory, name="list_directory",
            description="List files and directories at a given path with optional glob filtering."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=grep_files, name="grep_files",
            description="Search for a regex pattern across files in a directory (like grep -rn)."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=parse_structured_file, name="parse_structured_file",
            description="Parse JSON, TOML, YAML, or CBOR files into normalized structured output."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=python_ast_query, name="python_ast_query",
            description="Inspect a Python file's AST to list symbols, imports, or call sites."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=list_c_symbols, name="list_c_symbols",
            description="List C/C++ functions, declarations, structs, enums, typedefs, and macros from source files."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=list_asm_symbols, name="list_asm_symbols",
            description="List assembly labels, globals, and typed symbols from .s/.S/.asm files."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=find_symbol_references, name="find_symbol_references",
            description="Find likely references to a C or ASM symbol across source files."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=read_binary_symbols, name="read_binary_symbols",
            description="Read a binary, ELF, or object-file symbol table via nm."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=parse_pdf, name="parse_pdf",
            description="Extract structured text from a local PDF with page-number annotations."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=extract_pdf_tables, name="extract_pdf_tables",
            description="Extract tables from a local PDF using PyMuPDF's table detector."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=describe_image, name="describe_image",
            description="Describe the contents of a local image using a local vision-capable Ollama model."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=ocr_image, name="ocr_image",
            description="Extract visible text from a local image using a local vision-capable Ollama model."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=index_directory, name="index_directory",
            description="Build a local semantic vector index for source code and documentation in a directory."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=semantic_search, name="semantic_search",
            description="Search the local semantic index for code or document evidence before using the web."
        ))
        # --- Shell tools ---
        tools.append(FunctionTool.from_defaults(
            fn=run_shell_command, name="run_shell_command",
            description="Run an allowlisted shell command (git, make, gcc, pytest, etc.) and return output."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=git_status, name="git_status",
            description="Run git status --short in the given directory."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=git_log, name="git_log",
            description="Return the last N git log entries in oneline format."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=create_diff, name="create_diff",
            description="Create a unified diff from two text blobs or two file paths."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=apply_patch, name="apply_patch",
            description="Apply a unified diff in the current working directory using the patch tool."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=http_request, name="http_request",
            description="Make HTTP GET, POST, PUT, PATCH, or DELETE requests to local or remote services."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=system_info, name="system_info",
            description="Inspect the current machine's CPU, memory, disk, and top processes."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=list_serial_ports, name="list_serial_ports",
            description="List likely serial devices such as /dev/ttyUSB* and /dev/ttyACM*."
        ))
        # --- Date/time ---
        tools.append(FunctionTool.from_defaults(
            fn=get_current_datetime, name="get_current_datetime",
            description="Return the current UTC date and time. Use this whenever you need to know today's date."
        ))
        tools.append(FunctionTool.from_defaults(
            fn=get_timezone_time, name="get_timezone_time",
            description="Return the current time in a given IANA timezone (e.g. 'America/New_York')."
        ))
        # --- GitHub MCP ---
        tools.extend(await load_mcp_github_tools(info_panel))
    chat_store = chat_store or init_chat_store(MEMORY_PATH)
    memory = memory or build_memory(chat_store)
    agent = ReActAgent(tools=tools, llm=llm, memory=memory, verbose=False,
                       system_prompt=SYSTEM_PROMPT)
    return agent, chat_store, memory, tools, ddg_spec


async def process_agent_events(handler, ui: RichDashboard):
    async for event in handler.stream_events():
        event_name = type(event).__name__
        if event_name in {"AgentOutput", "AgentThought", "ReasoningEvent", "ThinkingEvent"}:
            thought_text = extract_thinking_text(event)
            if thought_text:
                source = (
                    getattr(event, "source", None)
                    or getattr(event, "model_name", None)
                    or getattr(event, "agent_name", None)
                    or "main_model"
                )
                ui.set_phase(f"Reasoning: {source}")
                ui.add_reasoning(f"[{source}] {thought_text[:1200]}")
                telemetry.write("reasoning_update", {"source": source, "event_name": event_name, "chars": len(thought_text)})
                runtime_metrics.emit("reasoning_chars", len(thought_text), {"source": source})
        elif event_name == "AgentStream":
            delta = (
                getattr(event, "delta", None)
                or getattr(event, "text", None)
                or getattr(event, "response", None)
                or ""
            )
            if delta and isinstance(delta, str):
                ui.update_answer_stream(ui.answer_stream + delta)
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
        elif event_name == "StopEvent":
            continue
        else:
            ui.log(f"Event: {event_name}", "white")
            telemetry.write("event", {"event_name": event_name})


def persist_chat_store(chat_store):
    chat_store.persist(persist_path=MEMORY_PATH)


async def execute_agent_prompt(agent_for_turn, agent_input: str, ui: RichDashboard) -> str:
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

    return extract_response_text(response_holder["response"]).strip() or "(No final response text returned.)"


def build_plan_step_prompt(
    user_msg: str,
    route: dict,
    payload: dict,
    plan: Plan,
    step: Step,
) -> str:
    completed_steps = "\n".join(
        f"- {item.title}: {item.result[:800]}"
        for item in plan.steps
        if item.status == "done" and item.result
    ) or "- No completed steps yet."
    return (
        "You are executing one tracked plan step for the user's request.\n\n"
        f"User request:\n{user_msg}\n\n"
        f"Route:\n{route}\n\n"
        f"Normalized request:\n{payload.get('normalized')}\n\n"
        f"Current plan status:\n{render_plan_status(plan)}\n\n"
        f"Current step:\n- Title: {step.title}\n- Description: {step.description}\n\n"
        f"Completed step results:\n{completed_steps}\n\n"
        "Instructions:\n"
        "1. Focus only on completing the current step.\n"
        "2. Use tools as needed.\n"
        "3. If the step order is wrong or you are blocked on missing context, "
        "start the final answer with 'REVISE PLAN:' and explain what new step is needed.\n"
        "4. Otherwise start the final answer with 'STEP DONE:' and summarize the work, "
        "including any files changed or commands/tests run.\n"
        "5. Be concrete and concise.\n"
    )


def build_plan_final_prompt(user_msg: str, route: dict, payload: dict, plan: Plan) -> str:
    step_results = "\n".join(
        f"- {step.title} [{step.status}]: {(step.result or step.error)[:1200]}"
        for step in plan.steps
    )
    return (
        "You have completed a tracked plan for the user's request.\n\n"
        f"User request:\n{user_msg}\n\n"
        f"Route:\n{route}\n\n"
        f"Normalized request:\n{payload.get('normalized')}\n\n"
        f"Final plan status:\n{render_plan_status(plan)}\n\n"
        f"Step outcomes:\n{step_results}\n\n"
        "Respond to the user with the actual result of the work. "
        "Lead with what changed, then what was verified, then any remaining risk or blocker. "
        "Do not narrate the internal plan machinery."
    )


async def execute_planned_turn(
    user_msg: str,
    route: dict,
    payload: dict,
    selected_model: str,
    tools,
    chat_store,
    memory,
    ui: RichDashboard,
) -> str:
    try:
        plan = Plan.from_dict(payload.get("plan") or {})
        if not plan.steps:
            raise ValueError("empty plan")
    except Exception:
        plan = build_default_plan(user_msg)

    async def run_step(step: Step, active_plan: Plan) -> StepResult:
        step_index = active_plan.steps.index(step) + 1
        emit_event(
            ui,
            "code_reasoner",
            "plan_step",
            f"Running plan step {step_index}/{len(active_plan.steps)}: {step.title}",
            model=selected_model,
        )
        ui.set_phase(f"Plan step {step_index}/{len(active_plan.steps)}")
        agent_for_step, _, _, _, _ = await build_agent(
            selected_model,
            preloaded_tools=tools,
            chat_store=chat_store,
            memory=memory,
        )
        step_prompt = build_plan_step_prompt(user_msg, route, payload, active_plan, step)
        step_text = await execute_agent_prompt(agent_for_step, step_prompt, ui)
        if step_text.lstrip().upper().startswith("REVISE PLAN:"):
            reason = step_text.split(":", 1)[1].strip() or "The current plan needs another step."
            return StepResult(status="revise", summary=reason)
        return StepResult(status="done", summary=step_text)

    def replan(active_plan: Plan, failed_step: Step, exc: Exception) -> Plan | None:
        if active_plan.revision_count >= 2:
            return None
        reason = str(exc).strip() or f"Blocked while executing {failed_step.title}."
        active_plan.notes.append(f"Revision after '{failed_step.title}': {reason}")
        idx = active_plan.steps.index(failed_step)
        failed_step.status = "pending"
        failed_step.error = ""
        recovery_title = f"Resolve blocker for {failed_step.title}"
        already_present = any(
            step.title == recovery_title and step.status != "done"
            for step in active_plan.steps
        )
        if not already_present:
            active_plan.steps.insert(
                idx,
                Step(
                    title=recovery_title,
                    description=reason[:240],
                    status="pending",
                ),
            )
        telemetry.write("plan_revision", {"step": failed_step.title, "reason": reason})
        return active_plan

    executor = PlanExecutor(
        step_runner=run_step,
        checkpoint_path=PLAN_CHECKPOINT_PATH,
        replanner=replan,
    )
    completed_plan = await executor.run_async(plan)
    emit_event(
        ui,
        "code_reasoner",
        "plan_complete",
        f"Completed plan with {len(completed_plan.steps)} step(s) and {completed_plan.revision_count} revision(s)",
        model=selected_model,
    )
    ui.set_phase("Synthesizing planned result")
    agent_for_turn, _, _, _, _ = await build_agent(
        selected_model,
        preloaded_tools=tools,
        chat_store=chat_store,
        memory=memory,
    )
    final_prompt = build_plan_final_prompt(user_msg, route, payload, completed_plan)
    return await execute_agent_prompt(agent_for_turn, final_prompt, ui)


async def run_turn_rich(user_msg, ddg_spec, tools, chat_store, memory):
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
        try:
            engram_matches = engram_store.retrieve(user_msg)
        except Exception as exc:
            engram_matches = []
            telemetry.write("engram_error", {"phase": "retrieve", "error": str(exc)})
        if engram_matches:
            emit_event(ui, "router", "engram_recall", f"Loaded {len(engram_matches)} engrams for this turn")
            telemetry.write("engram_recall", {"count": len(engram_matches), "query": user_msg[:200]})
            runtime_metrics.emit("engram_hits", len(engram_matches))

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
            agent_input = apply_engram_context(
                user_msg + "\n\nKeep the answer concise. Provide only the code first, then 2-4 short bullets max.",
                engram_matches,
            )
        elif should_escalate(route, ROUTER_ESCALATION_THRESHOLD):
            emit_event(ui, "router", "escalation", "Route confidence below threshold; escalating to main model", model=router.model_name)
            telemetry.write("route_escalation", {"threshold": ROUTER_ESCALATION_THRESHOLD, "route": route})
            agent_input = apply_engram_context(user_msg, engram_matches)
        elif route["route"] == "web_research":
            ui.set_phase("Ultra-light helpers")
            payload = await asyncio.to_thread(WebReasoner(ddg_spec).run, user_msg)
            if engram_matches:
                payload["engram_matches"] = engram_matches
            emit_event(
                ui, "web_reasoner", "evidence",
                f"AutoResearch collected {len(payload.get('evidence', []))} evidence items "
                f"across {len(payload.get('rounds', []))} rounds",
                model=payload.get("model"),
            )
            handoff = build_handoff_packet(user_msg, route, payload)
            compression = handoff.get("payload", {}).get("turboquant", {})
            telemetry.write("web_handoff", {
                "evidence_count": len(payload.get('evidence', [])),
                "raw_result_count": payload.get('raw_result_count', 0),
                "rounds": len(payload.get("rounds", [])),
                "turboquant_saved_chars": compression.get("saved_chars", 0),
            })
            runtime_metrics.emit("web_evidence_count", len(payload.get("evidence", [])))
            runtime_metrics.emit("autoresearch_rounds", len(payload.get("rounds", [])))
            runtime_metrics.emit("turboquant_saved_chars", compression.get("saved_chars", 0))
            agent_input = f"Use this structured handoff packet as context for your final response. Be concise and avoid repetition.\n\n{handoff}"
        elif route["route"] == "code_reasoning":
            ui.set_phase("Medium reasoning")
            payload = await asyncio.to_thread(CodeReasoner().run, user_msg)
            if engram_matches:
                payload["engram_matches"] = engram_matches
            emit_event(ui, "code_reasoner", "analysis", f"Prepared code analysis with next step {payload.get('next_step')}", model=payload.get("model"))
            if should_use_task_planner(user_msg, route, payload.get("normalized")):
                telemetry.write("plan_execution", {"route": route.get("route"), "objective": user_msg[:200]})
                final_text = await execute_planned_turn(
                    user_msg=user_msg,
                    route=route,
                    payload=payload,
                    selected_model=selected_model,
                    tools=tools,
                    chat_store=chat_store,
                    memory=memory,
                    ui=ui,
                )
                critic_review = critic_reasoner.review(final_text)
                if not critic_review.get("ok", True):
                    warnings = "\n".join(f"- {item}" for item in critic_review.get("findings", []))
                    final_text = f"{final_text}\n\nCritic warnings:\n{warnings}"
                    telemetry.write("critic_warning", critic_review)
                if len(final_text) > MAX_FINAL_ANSWER_CHARS:
                    final_text = final_text[:MAX_FINAL_ANSWER_CHARS].rstrip() + "\n\n[truncated]"
                try:
                    remembered = engram_store.remember(
                        user_msg, final_text,
                        route=route.get("route"),
                        metadata={"model": selected_model, "planned": True},
                    )
                except Exception as exc:
                    remembered = None
                    telemetry.write("engram_error", {"phase": "remember", "error": str(exc)})
                if remembered:
                    telemetry.write("engram_write", {"route": remembered.get("route"), "terms": remembered.get("terms", [])})
                ui.set_phase("Final answer ready")
                emit_event(ui, "main_model", "final", "Final answer captured", model=selected_model)
                total_elapsed = time.perf_counter() - started
                console.print(Panel(
                    Markdown(f"**Total query time:** {format_elapsed(total_elapsed)}\n\n{final_text}"),
                    title="🤖 Final Answer", border_style="bright_green", box=box.ROUNDED, padding=(0, 1),
                ))
                telemetry.write("final_answer", {"elapsed_s": total_elapsed, "chars": len(final_text), "route": route.get("route"), "model": selected_model, "planned": True})
                return
            handoff = build_handoff_packet(user_msg, route, payload)
            compression = handoff.get("payload", {}).get("turboquant", {})
            telemetry.write("code_handoff", {"next_step": payload.get('next_step')})
            runtime_metrics.emit("turboquant_saved_chars", compression.get("saved_chars", 0))
            agent_input = f"Use this structured handoff packet as context for your final response. Keep it short, practical, and non-repetitive.\n\n{handoff}"
        else:
            agent_input = apply_engram_context(user_msg, engram_matches)

        agent_for_turn, _, _, _, _ = await build_agent(
            selected_model,
            preloaded_tools=tools,
            chat_store=chat_store,
            memory=memory,
        )
        final_text = await execute_agent_prompt(agent_for_turn, agent_input, ui)
        critic_review = critic_reasoner.review(final_text)
        if not critic_review.get("ok", True):
            warnings = "\n".join(f"- {item}" for item in critic_review.get("findings", []))
            final_text = f"{final_text}\n\nCritic warnings:\n{warnings}"
            telemetry.write("critic_warning", critic_review)
        if len(final_text) > MAX_FINAL_ANSWER_CHARS:
            final_text = final_text[:MAX_FINAL_ANSWER_CHARS].rstrip() + "\n\n[truncated]"
        try:
            remembered = engram_store.remember(
                user_msg, final_text,
                route=route.get("route"),
                metadata={"model": selected_model},
            )
        except Exception as exc:
            remembered = None
            telemetry.write("engram_error", {"phase": "remember", "error": str(exc)})
        if remembered:
            telemetry.write("engram_write", {"route": remembered.get("route"), "terms": remembered.get("terms", [])})

        ui.set_phase("Final answer ready")
        emit_event(ui, "main_model", "final", "Final answer captured", model=selected_model)
        total_elapsed = time.perf_counter() - started
        console.print(Panel(
            Markdown(f"**Total query time:** {format_elapsed(total_elapsed)}\n\n{final_text}"),
            title="🤖 Final Answer", border_style="bright_green", box=box.ROUNDED, padding=(0, 1),
        ))
        telemetry.write("final_answer", {"elapsed_s": total_elapsed, "chars": len(final_text), "route": route.get("route"), "model": selected_model})
    finally:
        if span_cm:
            span_cm.__exit__(None, None, None)


async def main():
    startup_model = MAIN_MODEL
    _, chat_store, memory, tools, ddg_spec = await build_agent(startup_model)
    print_rich_banner(build_runtime_profile(startup_model, len(tools), ddg_spec))
    while True:
        try:
            user_msg = console.input("[bold cyan]elf_g > [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break
        if not user_msg:
            continue

        if user_msg.strip().lower() == "/compact":
            console.print("[yellow]Forcing memory compaction...[/yellow]")
            if hasattr(memory, "force_compact"):
                success, msg = memory.force_compact()
                if success:
                    persist_chat_store(chat_store)
                    console.print(f"[green]Success: {msg}[/green]")
                else:
                    console.print(f"[yellow]{msg}[/yellow]")
            else:
                console.print("[red]Memory buffer does not support force compaction.[/red]")
            continue

        result = await handle_local_command(user_msg, tools)
        if result is None:
            await run_turn_rich(user_msg, ddg_spec, tools, chat_store, memory)
        elif result[0] == "print":
            console.print(result[1])
        elif result[0] == "clear":
            console.clear()
        elif result[0] == "quit":
            console.print("[yellow]Saving memory and exiting...[/yellow]")
            try:
                persist_chat_store(chat_store)
            except Exception as e:
                console.print(f"[red]Failed to save memory: {e}[/red]")
            break
