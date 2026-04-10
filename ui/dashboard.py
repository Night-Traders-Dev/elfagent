import threading, time
from datetime import datetime
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.pretty import Pretty
from rich.layout import Layout
from rich.table import Table
from rich.columns import Columns
from rich import box
from utils.formatting import truncate, format_elapsed

class RichDashboard:
    def __init__(self):
        self.turn = 0
        self.phase = "Idle"
        self.source_counts = {}
        self.source_logs = {"router": [], "web_reasoner": [], "code_reasoner": [], "main_model": [], "tools": []}
        self.timeline = []
        self.turn_start = None
        self.current_tool_name = None
        self.current_tool_start = None
        self.reasoning_current = []
        self.reasoning_history = []
        self.answer_stream = ""
        self.tool_status = {}
        self.activity_log = []
        self.last_tool_kwargs = {}
        self.lock = threading.Lock()

    def next_turn(self):
        with self.lock:
            self.turn += 1
            self.phase = "Waiting"
            self.turn_start = time.perf_counter()
            self.current_tool_name = None
            self.current_tool_start = None
            self.reasoning_current = []
            self.reasoning_history = []
            self.answer_stream = ""
            self.tool_status = {}
            self.activity_log = []
            self.last_tool_kwargs = {}
            self.source_counts = {}
            self.timeline = []
            self.source_logs = {"router": [], "web_reasoner": [], "code_reasoner": [], "main_model": [], "tools": []}

    def log(self, message, style="white"):
        with self.lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.activity_log.append((ts, message, style))
            self.activity_log = self.activity_log[-20:]
            self.timeline.append((ts, message))
            self.timeline = self.timeline[-40:]
            if message.startswith("[") and "]" in message:
                src = message[1:message.index("]")]
                key = "tools" if src.startswith("tool:") else src
                self.source_counts[src] = self.source_counts.get(src, 0) + 1
                if key in self.source_logs:
                    self.source_logs[key].append((ts, message))
                    self.source_logs[key] = self.source_logs[key][-12:]

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def add_reasoning(self, text):
        text = text.strip()
        if text:
            with self.lock:
                if not self.reasoning_current or self.reasoning_current[-1] != text:
                    self.reasoning_current.append(text)

    def archive_reasoning(self):
        with self.lock:
            if self.reasoning_current:
                combined = "\n\n".join(self.reasoning_current).strip()
                if combined:
                    self.reasoning_history.append(combined)
                    self.reasoning_history = self.reasoning_history[-8:]
            self.reasoning_current = []

    def update_answer_stream(self, text):
        if text:
            with self.lock:
                self.answer_stream = text

    def start_tool(self, tool_name, tool_kwargs):
        with self.lock:
            self.current_tool_name = tool_name
            self.current_tool_start = time.perf_counter()
            self.tool_status[tool_name] = {"status": "RUNNING", "elapsed": None, "last_update": datetime.now().strftime("%H:%M:%S")}
            self.last_tool_kwargs[tool_name] = tool_kwargs
        self.log(f"[tool:{tool_name}] started", "yellow")

    def finish_tool(self, tool_name, output_text):
        with self.lock:
            elapsed = time.perf_counter() - self.current_tool_start if self.current_tool_name == tool_name and self.current_tool_start is not None else None
            self.current_tool_name = None
            self.current_tool_start = None
            failed = any(x in output_text.lower() for x in ["traceback", "error", "exception"])
            self.tool_status[tool_name] = {"status": "FAIL" if failed else "SUCCESS", "elapsed": elapsed, "last_update": datetime.now().strftime("%H:%M:%S")}
        status_word = "failed" if failed else "ok"
        self.log(f"[tool:{tool_name}] {status_word}", "red" if failed else "green")

    def render_status_bar(self):
        total_elapsed = time.perf_counter() - self.turn_start if self.turn_start is not None else 0.0
        current_tool_line = f"{self.current_tool_name} · {format_elapsed(time.perf_counter() - self.current_tool_start)}" if self.current_tool_name and self.current_tool_start is not None else "No active tool"
        stats = Columns([Spinner("dots", text=f"[bold cyan]{self.phase}"), Text(f"Query: {format_elapsed(total_elapsed)}", style="bold green"), Text(f"Tool: {current_tool_line}", style="yellow"), Text(f"Turn: {self.turn}", style="white"), Text(f"Tools used: {len(self.tool_status)}", style="cyan")], expand=True)
        return Panel(stats, title="⚡ Live Status", border_style="bright_blue", box=box.ROUNDED)

    def render_tool_table(self):
        table = Table(expand=True, box=box.SIMPLE_HEAVY)
        table.add_column("Badge", width=5)
        table.add_column("Tool", style="bold")
        table.add_column("Status")
        table.add_column("Elapsed", justify="right")
        table.add_column("Updated", justify="right")
        if not self.tool_status:
            table.add_row("-", "No tools yet", "-", "-", "-")
        else:
            for name, item in list(self.tool_status.items())[-8:]:
                elapsed = item["elapsed"]
                if item["status"] == "RUNNING" and self.current_tool_name == name and self.current_tool_start is not None:
                    elapsed = time.perf_counter() - self.current_tool_start
                table.add_row({"RUNNING": "🟡", "SUCCESS": "✅", "FAIL": "❌"}.get(item["status"], "•"), name, item["status"], format_elapsed(elapsed) if elapsed is not None else "-", item["last_update"] )
        return Panel(table, title="🧰 Tool Execution", border_style="cyan", box=box.ROUNDED)

    def render_activity_log(self):
        table = Table(expand=True, box=box.SIMPLE)
        table.add_column("Time", width=9, style="dim")
        table.add_column("Event")
        if not self.activity_log:
            table.add_row("--:--:--", "Waiting for activity")
        else:
            for ts, msg, style in self.activity_log[-12:]:
                table.add_row(ts, Text(msg, style=style))
        return Panel(table, title="📋 Activity Log", border_style="white", box=box.ROUNDED)

    def render_timeline(self):
        table = Table(expand=True, box=box.SIMPLE_HEAVY)
        table.add_column("Time", width=9, style="dim")
        table.add_column("Event")
        if not self.timeline:
            table.add_row("--:--:--", "No timeline yet")
        else:
            for ts, msg in self.timeline[-12:]:
                table.add_row(ts, msg)
        return Panel(table, title="⏱ Execution Timeline", border_style="bright_cyan", box=box.ROUNDED)

    def render_current_reasoning(self):
        body = "\n\n".join(self.reasoning_current).strip() or "No current reasoning text yet."
        return Panel(Text(truncate(body, 2500), style="white"), title=f"🧠 Current Reasoning ({len(self.reasoning_current)})", border_style="magenta", box=box.ROUNDED)

    def render_reasoning_history(self):
        body = "\n\n---\n\n".join(self.reasoning_history[-4:]).strip() or "No reasoning history yet."
        return Panel(Text(truncate(body, 3000), style="dim"), title=f"📜 Reasoning History ({len(self.reasoning_history)})", border_style="bright_magenta", box=box.ROUNDED)

    def render_source_metrics(self):
        table = Table(expand=True, box=box.SIMPLE_HEAVY)
        table.add_column("Source", style="bold")
        table.add_column("Events", justify="right")
        if not self.source_counts:
            table.add_row("No sources yet", "0")
        else:
            agg = {}
            for src, count in self.source_counts.items():
                key = "tools" if src.startswith("tool:") else src
                agg[key] = agg.get(key, 0) + count
            for src, count in sorted(agg.items()):
                table.add_row(src, str(count))
        return Panel(table, title="📊 Source Metrics", border_style="blue", box=box.ROUNDED)

    def render_source_panel(self, source_key, title, border_style):
        lines = self.source_logs.get(source_key, [])
        body = "No events yet." if not lines else "\n".join(f"{ts} {msg}" for ts, msg in lines[-10:])
        return Panel(Text(truncate(body, 1800), style="white"), title=title, border_style=border_style, box=box.ROUNDED)

    def render_last_tool_input(self):
        if not self.last_tool_kwargs:
            body = Text("No tool input captured yet.", style="dim")
        else:
            kwargs = self.last_tool_kwargs[list(self.last_tool_kwargs.keys())[-1]]
            body = Syntax(kwargs["code"], "python", theme="monokai", line_numbers=True, word_wrap=True) if isinstance(kwargs, dict) and "code" in kwargs and isinstance(kwargs["code"], str) else Pretty(kwargs, expand_all=True)
        return Panel(body, title="🛠 Last Tool Input", border_style="yellow", box=box.ROUNDED)

    def render_answer_stream(self):
        return Panel(Markdown(truncate(self.answer_stream.strip() or "Waiting for final answer stream...", 5000)), title="💬 Streaming Answer", border_style="bright_green", box=box.ROUNDED)

    def render_footer(self):
        return Panel(Columns([Text("/help", style="bold cyan"), Text("/tools", style="bold cyan"), Text("/clear", style="bold cyan"), Text("/pwd", style="bold cyan"), Text("/summarize-medical", style="bold green"), Text("/summarize-meeting", style="bold green"), Text("quit", style="bold red")], expand=True), border_style="bright_black", box=box.ROUNDED)

    def render(self):
        layout = Layout()
        layout.split_column(Layout(name="top", size=3), Layout(name="main"), Layout(name="footer", size=3))
        layout["main"].split_row(Layout(name="left", ratio=3), Layout(name="center", ratio=4), Layout(name="right", ratio=4))
        layout["left"].split_column(Layout(name="tool_table", ratio=2), Layout(name="activity_log", ratio=2), Layout(name="timeline", ratio=2))
        layout["center"].split_column(Layout(name="reasoning_current", ratio=2), Layout(name="reasoning_history", ratio=2), Layout(name="source_metrics", ratio=1))
        layout["right"].split_column(Layout(name="last_tool_input", ratio=2), Layout(name="answer_stream", ratio=2), Layout(name="source_panels", ratio=3))
        layout["top"].update(self.render_status_bar())
        layout["tool_table"].update(self.render_tool_table())
        layout["activity_log"].update(self.render_activity_log())
        layout["timeline"].update(self.render_timeline())
        layout["reasoning_current"].update(self.render_current_reasoning())
        layout["reasoning_history"].update(self.render_reasoning_history())
        layout["source_metrics"].update(self.render_source_metrics())
        layout["last_tool_input"].update(self.render_last_tool_input())
        layout["answer_stream"].update(self.render_answer_stream())
        layout["source_panels"].update(Columns([self.render_source_panel("router", "🧭 Router", "cyan"), self.render_source_panel("web_reasoner", "🌐 Web", "magenta"), self.render_source_panel("code_reasoner", "🧪 Code", "magenta"), self.render_source_panel("main_model", "🧠 Main", "green"), self.render_source_panel("tools", "🛠 Tools", "yellow")], expand=True))
        layout["footer"].update(self.render_footer())
        return layout
