from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Step:
    title: str
    description: str
    status: str = "pending"
    result: str = ""
    error: str = ""


@dataclass
class Plan:
    objective: str
    steps: list[Step]
    notes: list[str] = field(default_factory=list)
    revision_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Plan":
        return cls(
            objective=payload["objective"],
            steps=[Step(**step) for step in payload.get("steps", [])],
            notes=list(payload.get("notes", [])),
            revision_count=int(payload.get("revision_count", 0)),
        )


@dataclass
class StepResult:
    status: str
    summary: str
    error: str = ""


class PlanRevisionNeeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class PlanExecutor:
    """Execute plan steps one by one with checkpointed state."""

    def __init__(
        self,
        step_runner: Callable[[Step, Plan], str],
        checkpoint_path: str | None = None,
        replanner: Callable[[Plan, Step, Exception], Plan | None] | None = None,
    ):
        self.step_runner = step_runner
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        self.replanner = replanner

    def save(self, plan: Plan):
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    def load(self) -> Plan | None:
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return None
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return Plan.from_dict(payload)

    def _normalize_step_result(self, outcome) -> StepResult:
        if isinstance(outcome, StepResult):
            return outcome
        if isinstance(outcome, str):
            return StepResult(status="done", summary=outcome)
        return StepResult(status="done", summary=str(outcome))

    def _maybe_call(self, fn, *args):
        value = fn(*args)
        if inspect.isawaitable(value):
            return value
        return value

    def run(self, plan: Plan) -> Plan:
        self.save(plan)
        index = 0
        while index < len(plan.steps):
            step = plan.steps[index]
            if step.status == "done":
                index += 1
                continue

            step.status = "running"
            self.save(plan)
            try:
                step.result = self.step_runner(step, plan)
                step.status = "done"
                step.error = ""
            except Exception as exc:  # noqa: BLE001
                step.status = "failed"
                step.error = str(exc)
                self.save(plan)
                if not self.replanner:
                    raise
                revised = self.replanner(plan, step, exc)
                if revised is None:
                    raise
                plan = revised
                index = 0
                self.save(plan)
                continue

            self.save(plan)
            index += 1
        return plan

    async def run_async(self, plan: Plan) -> Plan:
        self.save(plan)
        index = 0
        while index < len(plan.steps):
            step = plan.steps[index]
            if step.status == "done":
                index += 1
                continue

            step.status = "running"
            self.save(plan)
            try:
                raw_outcome = self._maybe_call(self.step_runner, step, plan)
                if inspect.isawaitable(raw_outcome):
                    raw_outcome = await raw_outcome
                outcome = self._normalize_step_result(raw_outcome)
                if outcome.status == "revise":
                    raise PlanRevisionNeeded(outcome.summary)
                if outcome.status == "failed":
                    raise RuntimeError(outcome.error or outcome.summary)
                step.result = outcome.summary
                step.status = "done"
                step.error = ""
            except Exception as exc:  # noqa: BLE001
                step.status = "failed"
                step.error = str(exc)
                self.save(plan)
                if not self.replanner:
                    raise
                revised = self._maybe_call(self.replanner, plan, step, exc)
                if inspect.isawaitable(revised):
                    revised = await revised
                if revised is None:
                    raise
                plan = revised
                plan.revision_count += 1
                index = 0
                self.save(plan)
                continue

            self.save(plan)
            index += 1
        return plan


def build_default_plan(objective: str) -> Plan:
    return Plan(
        objective=objective,
        steps=[
            Step(
                title="Inspect current state",
                description="Read the relevant files, tools, and tests before changing anything.",
            ),
            Step(
                title="Implement change",
                description="Modify the code or configuration needed to satisfy the request.",
            ),
            Step(
                title="Verify result",
                description="Run focused validation such as tests, lint, or a smoke check.",
            ),
        ],
        notes=["This plan can be revised mid-execution if a step fails."],
    )


def should_use_task_planner(user_msg: str, route: dict, normalized: dict | None = None) -> bool:
    route_name = route.get("route")
    text = (user_msg or "").lower()
    markers = [
        "refactor",
        "rewrite",
        "restructure",
        "port",
        "migrate",
        "set up",
        "setup",
        "install",
        "configure",
        "multi-file",
        "workflow",
        "plan",
        "verify",
        "fix",
        "patch",
    ]
    if route_name == "code_reasoning":
        if normalized and normalized.get("needs_patch"):
            return True
        return any(marker in text for marker in markers)
    return any(marker in text for marker in ["setup", "install", "configure", "multi-step"])


def render_plan_status(plan: Plan) -> str:
    lines = [f"Objective: {plan.objective}", f"Revisions: {plan.revision_count}"]
    for idx, step in enumerate(plan.steps, start=1):
        lines.append(f"{idx}. [{step.status}] {step.title}: {step.description}")
    if plan.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in plan.notes[-5:])
    return "\n".join(lines)
