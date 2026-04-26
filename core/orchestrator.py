from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import MutationAgent, MutationResult
from .context_builder import build_context_packet
from .executor import Executor, LocalExecutor, RunSpec
from .ids import next_run_id
from .queue import MutationQueue, QueueItem
from .review import Reviewer
from .rp import ResearchProblem
from .run_store import RunArtifacts
from .selection import select_parent
from .store import EvolutionStore
from .workspace import create_workspace, snapshot_hashes, validate_only_allowed_changed


@dataclass
class EvolutionConfig:
    arena_root: Path
    rp: ResearchProblem
    steps: int = 1
    smoke: bool = False
    timeout_s: int | None = None
    extra_args: list[str] = field(default_factory=list)
    validate_single_file: bool = True
    use_queue: bool = True


@dataclass
class EvolutionStepResult:
    run: dict[str, Any]
    review: dict[str, Any]
    queued: list[dict[str, Any]]
    context_path: str
    workspace_path: str

def _num_or_none(value: Any) -> float | None:
    try:
       return None if value in (None, "", "None") else float(value)
    except (TypeError, ValueError):
       return None

def _patch_json_object(path: Path, updates: dict[str, Any]) -> None:
    data: dict[str, Any] = {}

    if path.exists():
        try:
           loaded = json.loads(path.read_text(encoding="utf-8"))
           if isinstance(loaded, dict):
               data = loaded
        except Exception:
            data = {}
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


class EvolutionOrchestrator:
    """Small autonomous loop: select → mutate → execute → review → queue."""

    def __init__(
        self,
        *,
        config: EvolutionConfig,
        agent: MutationAgent,
        reviewer: Reviewer,
        store: EvolutionStore | None = None,
        executor: Executor | None = None,
    ):
        self.config = config
        self.agent = agent
        self.reviewer = reviewer
        self.store = store or EvolutionStore(config.arena_root)
        self.queue = MutationQueue(config.arena_root)
        self.executor = executor or LocalExecutor(RunArtifacts(config.arena_root), timeout_s=config.timeout_s)

    async def evolve_many(self) -> list[EvolutionStepResult]:
        results: list[EvolutionStepResult] = []
        for _ in range(self.config.steps):
            results.append(await self.evolve_one())
        return results

    async def evolve_one(self) -> EvolutionStepResult:
        evolution_started = time.monotonic()
        run_id = next_run_id(self.config.arena_root)
        assignment = self._next_assignment()
        queue_item_id = str(assignment.get("queue_item_id") or "")
        try:
            parent = assignment.get("parent")
            parent_id = parent.get("run_id") if parent else None
            generation = int(parent.get("generation") or 0) + 1 if parent else 0
            parent_model = parent.get("model_path") if parent else None

            workspace_rp = create_workspace(
                arena_root=self.config.arena_root,
                source_rp=self.config.rp,
                run_id=run_id,
                parent_model_path=parent_model,
            )
            before = snapshot_hashes(workspace_rp.path)

            context = self._context(workspace_rp, parent_id, assignment)
            context_path = Path(self.config.arena_root) / "runs" / run_id / "context.md"
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text(context, encoding="utf-8")

            mutation = await self.agent.mutate(workspace_rp, context)
            changed = []
            if self.config.validate_single_file:
                changed = validate_only_allowed_changed(before, workspace_rp.path, {workspace_rp.mutable_file})

            mutation = self._merge_mutation_defaults(mutation, assignment, changed)

            record = await self.executor.execute(
                RunSpec(
                    rp=workspace_rp,
                    run_id=run_id,
                    parent_id=parent_id,
                    generation=generation,
                    smoke=self.config.smoke,
                    extra_args=self.config.extra_args,
                    mutation_type=mutation.mutation_type,
                    mutation_summary=mutation.mutation_summary,
                    hypothesis=mutation.hypothesis,
                )
            )

            child_dict = record.to_dict()
            review = await self.reviewer.review(parent, child_dict, context)
            evolution_time_seconds = round(time.monotonic() - evolution_started, 3)
            record.observation = str(review.get("observation") or record.observation or "")
            record.next_belief = str(review.get("next_belief") or record.next_belief or "")

            run_dir = Path(record.artifact_path)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "context.md").write_text(context, encoding="utf-8")
            (run_dir / "mutation.json").write_text(json.dumps(mutation.__dict__, indent=2, sort_keys=True), encoding="utf-8")
            (run_dir / "ds_review.json").write_text(json.dumps(review, indent=2, sort_keys=True, default=str), encoding="utf-8")

            rnd_metrics = {
                "evolution_time_seconds": evolution_time_seconds,
            }
            _patch_json_object(run_dir / "metrics.json", rnd_metrics)

            registered = self.store.register(record)
            queued = self._enqueue_next(record.run_id, review)

            if queue_item_id:
                self.queue.complete(queue_item_id)

            return EvolutionStepResult(
                run=registered,
                review=review,
                queued=queued,
                context_path=str(context_path),
                workspace_path=str(workspace_rp.path),
            )
        except BaseException:
            if queue_item_id:
                self.queue.fail(queue_item_id)
            raise

    def _next_assignment(self) -> dict[str, Any]:
        if self.config.use_queue:
            item = self.queue.pop_next()
            if item:
                parent = self.store.get(str(item.get("source_run_id"))) if item.get("source_run_id") else None
                return {
                    "parent": parent,
                    "queue_item_id": item.get("item_id"),
                    "mutation_type": item.get("mutation_type") or "safe_refinement",
                    "proposed_mutation": item.get("proposed_mutation") or "Make one bounded mutation.",
                    "priority": item.get("priority", 0.0),
                }

        parent = select_parent(self.store)
        if parent:
            return {
                "parent": parent,
                "mutation_type": "safe_refinement",
                "proposed_mutation": "Make one bounded mutation that improves score or Pareto position.",
                "priority": 0.5,
            }
        return {
            "parent": None,
            "mutation_type": "baseline",
            "proposed_mutation": "Run the current model.py unchanged to establish the first baseline.",
            "priority": 1.0,
        }

    def _context(self, workspace_rp: ResearchProblem, parent_id: str | None, assignment: dict[str, Any]) -> str:
        base = build_context_packet(self.store, workspace_rp, parent_id)
        assigned = f"""

## Assigned Mutation
- Type: {assignment.get('mutation_type')}
- Objective: {assignment.get('proposed_mutation')}
- Priority: {assignment.get('priority')}

Return structured mutation metadata after editing `{workspace_rp.mutable_file}`.
""".rstrip()
        return base + assigned + "\n"

    def _merge_mutation_defaults(self, mutation: MutationResult, assignment: dict[str, Any], changed: list[str]) -> MutationResult:
        if mutation.mutation_type in {"", "safe_refinement"} and assignment.get("mutation_type"):
            mutation.mutation_type = str(assignment["mutation_type"])
        if not mutation.mutation_summary:
            mutation.mutation_summary = str(assignment.get("proposed_mutation") or mutation.mutation_type)
        if not mutation.hypothesis:
            mutation.hypothesis = "Test whether the assigned bounded mutation improves the RP score."
        mutation.changed_files = changed or mutation.changed_files
        return mutation

    def _enqueue_next(self, source_run_id: str, review: dict[str, Any]) -> list[dict[str, Any]]:
        queued: list[dict[str, Any]] = []
        if not self.config.use_queue:
            return queued
        for rec in review.get("recommended_next_mutations") or []:
            item = QueueItem(
                item_id=f"q_{uuid.uuid4().hex[:12]}",
                source_run_id=source_run_id,
                mutation_type=str(rec.get("mutation_type") or "safe_refinement"),
                proposed_mutation=str(rec.get("description") or rec.get("proposed_mutation") or "Make one bounded mutation."),
                priority=float(rec.get("priority") or 0.5),
            )
            if self.queue.enqueue(item):
                queued.append(item.__dict__)
        return queued
