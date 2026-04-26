from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from core.agent import make_agent
from core.context_builder import build_context_packet
from core.env import default_dotenv_paths, load_dotenv_files
from core.executor import CloudRunExecutor, Executor, LocalExecutor, RunSpec
from core.ids import next_run_id
from core.orchestrator import EvolutionConfig, EvolutionOrchestrator
from core.queue import MutationQueue
from core.review import make_reviewer
from core.rp import ResearchProblem
from core.run_store import RunArtifacts
from core.store import EvolutionStore


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal stable self-evolution runner")
    p.add_argument("--arena", default=".arena", help="arena state folder")
    p.add_argument("--env-file", action="append", help="load an additional .env file before running subcommands")
    p.add_argument("--no-auto-env", action="store_true", help="do not auto-load .env from repo/cwd/RP folders")
    p.add_argument("--override-env", action="store_true", help="let .env values override existing process environment variables")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="validate RP and initialize arena")
    init.add_argument("--rp", required=True)

    run = sub.add_parser("run", help="execute RP and register one run")
    run.add_argument("--rp", required=True)
    run.add_argument("--run-id")
    run.add_argument("--parent-id")
    run.add_argument("--generation", type=int, default=0)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--timeout-s", type=int)
    run.add_argument("--mutation-type", default="baseline")
    run.add_argument("--mutation-summary", default="")
    run.add_argument("--hypothesis", default="")
    _add_executor_args(run)
    run.add_argument("extra_args", nargs="*")

    evolve = sub.add_parser("evolve", help="run autonomous select→mutate→execute→review→queue loop")
    evolve.add_argument("--rp", required=True)
    evolve.add_argument("--steps", type=int, default=1)
    evolve.add_argument("--agent", choices=["noop", "claude-code", "external-command"], default="noop")
    evolve.add_argument("--agent-command", help="command for --agent external-command")
    evolve.add_argument("--reviewer", choices=["heuristic", "claude-code", "external-command"], default="heuristic")
    evolve.add_argument("--reviewer-command", help="command for --reviewer external-command")
    evolve.add_argument("--smoke", action="store_true", help="use RP smoke/dry-run command")
    evolve.add_argument("--timeout-s", type=int)
    evolve.add_argument("--no-queue", action="store_true", help="select parents directly instead of consuming/enqueuing queue items")
    evolve.add_argument("--no-validate-single-file", action="store_true", help="allow edits beyond the RP mutable file")
    _add_executor_args(evolve)
    evolve.add_argument("extra_args", nargs="*")

    q = sub.add_parser("queue", help="list queued mutation ideas")
    q.add_argument("--status")

    leader = sub.add_parser("leaderboard")
    leader.add_argument("--limit", type=int, default=10)

    pareto = sub.add_parser("pareto")
    pareto.add_argument("--limit", type=int, default=20)

    context = sub.add_parser("context", help="print compact Claude Code context packet")
    context.add_argument("--rp", required=True)
    context.add_argument("--parent-id")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    demo_ui = sub.add_parser("demo-ui", help="serve read-only NiceGUI dashboard directly from .arena files")
    demo_ui.add_argument("--host", default="127.0.0.1")
    demo_ui.add_argument("--port", type=int, default=8080)
    demo_ui.add_argument("--poll-s", type=float, default=1.0, help="filesystem polling interval")
    demo_ui.add_argument("--title", default="AI Evolver Live")

    args = p.parse_args()
    arena = Path(args.arena).expanduser().resolve()
    _load_cli_env(args)

    if args.cmd == "init":
        rp = ResearchProblem.load(args.rp)
        store = EvolutionStore(arena)
        print(json.dumps({"arena": str(arena), "rp_id": rp.rp_id, "runs": len(store.all_runs())}, indent=2))
    elif args.cmd == "run":
        asyncio.run(_run(args, arena))
    elif args.cmd == "evolve":
        asyncio.run(_evolve(args, arena))
    elif args.cmd == "queue":
        _print_rows(MutationQueue(arena).list(args.status))
    elif args.cmd == "leaderboard":
        _print_rows(EvolutionStore(arena).leaderboard(args.limit))
    elif args.cmd == "pareto":
        _print_rows(EvolutionStore(arena).pareto_front()[: args.limit])
    elif args.cmd == "context":
        print(build_context_packet(EvolutionStore(arena), ResearchProblem.load(args.rp), args.parent_id))
    elif args.cmd == "search":
        _print_rows(EvolutionStore(arena).search(args.query, args.limit))
    elif args.cmd == "serve":
        import uvicorn
        from core.api import create_app

        uvicorn.run(create_app(arena), host=args.host, port=args.port)
    elif args.cmd == "demo-ui":
       from core.ui_demo import run_dashboard
       run_dashboard(arena, host=args.host, port=args.port, poll_s=args.poll_s, title=args.title)


def _load_cli_env(args: argparse.Namespace) -> None:
    rp_path = getattr(args, "rp", None)
    candidates: list[Path] = []
    if not getattr(args, "no_auto_env", False):
        candidates.extend(default_dotenv_paths(cli_file=__file__, cwd=Path.cwd(), rp_path=rp_path))
    for env_file in getattr(args, "env_file", None) or []:
        candidates.append(Path(env_file))
    load_dotenv_files(candidates, override=bool(getattr(args, "override_env", False)))



def _add_executor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--executor", choices=["local", "cloud-run"], default="local", help="where to execute the RP command")
    parser.add_argument("--cloud-runner-script", default="gcp_cloud_runner/application_cloud_runner.py", help="path to application_cloud_runner.py")
    parser.add_argument("--cloud-spec", help="path to cloud_runner.yaml; defaults to <rp>/cloud_runner.yaml")
    parser.add_argument("--cloud-dataset", help="optional gs:// dataset override for Cloud Run")
    parser.add_argument("--cloud-env", action="append", default=[], help="Cloud Run execution env override; repeat KEY=VALUE")
    parser.add_argument("--cloud-poll-interval-s", type=int, default=15)
    parser.add_argument("--cloud-log-format", choices=["text", "json"], default="text")


async def _run(args: argparse.Namespace, arena: Path) -> None:
    rp = ResearchProblem.load(args.rp)
    run_id = args.run_id or next_run_id(arena)
    executor = _make_executor(args, arena)
    record = await executor.execute(
        RunSpec(
            rp=rp,
            run_id=run_id,
            parent_id=args.parent_id,
            generation=args.generation,
            smoke=args.smoke,
            extra_args=args.extra_args,
            mutation_type=args.mutation_type,
            mutation_summary=args.mutation_summary,
            hypothesis=args.hypothesis,
        )
    )
    EvolutionStore(arena).register(record)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))


async def _evolve(args: argparse.Namespace, arena: Path) -> None:
    rp = ResearchProblem.load(args.rp)
    agent = make_agent(args.agent, command=args.agent_command, timeout_s=args.timeout_s)
    reviewer = make_reviewer(args.reviewer, command=args.reviewer_command, timeout_s=args.timeout_s)
    orchestrator = EvolutionOrchestrator(
        config=EvolutionConfig(
            arena_root=arena,
            rp=rp,
            steps=args.steps,
            smoke=args.smoke,
            timeout_s=args.timeout_s,
            extra_args=args.extra_args,
            validate_single_file=not args.no_validate_single_file,
            use_queue=not args.no_queue,
        ),
        agent=agent,
        reviewer=reviewer,
        executor=_make_executor(args, arena),
    )
    results = await orchestrator.evolve_many()
    print(json.dumps([r.__dict__ for r in results], indent=2, sort_keys=True, default=str))


def _make_executor(args: argparse.Namespace, arena: Path) -> Executor:
    artifacts = RunArtifacts(arena)
    if args.executor == "local":
        return LocalExecutor(artifacts, timeout_s=args.timeout_s)
    return CloudRunExecutor(
        artifacts,
        cloud_runner_script=_resolve_repo_path(args.cloud_runner_script),
        cloud_spec=Path(args.cloud_spec).expanduser().resolve() if args.cloud_spec else None,
        timeout_s=args.timeout_s,
        dataset=args.cloud_dataset,
        env=_parse_key_value(args.cloud_env),
        poll_interval_s=args.cloud_poll_interval_s,
        log_format=args.cloud_log_format,
    )


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _parse_key_value(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid empty environment variable name in: {item!r}")
        env[key] = value
    return env


def _print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[]")
        return
    keep = [
        "run_id", "item_id", "source_run_id", "parent_id", "status", "score", "val_accuracy",
        "parameter_count", "model_bytes", "latency_ms", "mutation_type", "mutation_summary",
        "proposed_mutation", "priority",
    ]
    print(json.dumps([{k: r.get(k) for k in keep if k in r} for r in rows], indent=2, default=str))


if __name__ == "__main__":
    main()
