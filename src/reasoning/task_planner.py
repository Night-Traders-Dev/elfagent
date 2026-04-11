from __future__ import annotations

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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Plan":
        return cls(
            objective=payload["objective"],
            steps=[Step(**step) for step in payload.get("steps", [])],
            notes=list(payload.get("notes", [])),
        )


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
