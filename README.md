
## Installation

**Clone the repository**

```bash
git clone https://github.com/akaliutau/evolyth
cd evolyth
```

**Create and activate a Conda environment**

```bash
conda create -n evolyth python=3.12 -y
conda activate evolyth
```

**Install dependencies**

```bash
pip install -r requirements.txt
```


# Stable Evolver Minimal

A compact implementation of the stable self-evolving software loop for model architecture research problems.

The only user-facing entry point is a Research Problem (RP) folder. The RP must contain:

```text
goal_prompt.md
train_eval.py
model.py
```

Optional `rp_contract.json` can override commands and output paths, but the defaults match `rp_tiny_cifar`:

```bash
python train_eval.py --dry-run --dataset synthetic --run-id <run_id>
python train_eval.py --dataset synthetic --run-id <run_id>
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Setup and configure the Caude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

set the `ANTHROPIC_API_KEY` var in `.env` or 
claude config set apiKey "sk-ant-..."

## One-shot run against an RP

```bash
python cli.py --arena .arena init --rp examples/tiny_dummy_rp

python cli.py --arena .arena run \
  --rp examples/tiny_dummy_rp \
  --smoke \
  --mutation-type baseline \
  --mutation-summary "initial smoke baseline"

python cli.py --arena .arena leaderboard
python cli.py --arena .arena pareto
python cli.py --arena .arena context --rp examples/tiny_dummy_rp
```

Extra arguments after the CLI flags are passed to the RP train script:

```bash
python cli.py --arena .arena run --rp examples/tiny_dummy_rp -- --epochs 2 --max-steps 200
```

## Autonomous evolution loop

The `evolve` command performs:

```text
select parent or queue item
→ create isolated RP workspace
→ build compact context packet
→ call mutation agent
→ validate only model.py changed
→ execute RP
→ extract metrics.json
→ reviewer produces DS review
→ register run in JSONL + LanceDB + NetworkX
→ enqueue next mutations
```

Smoke-test the loop without Claude Code:

```bash
python cli.py --arena .arena evolve \
  --rp examples/tiny_dummy_rp \
  --steps 2 \
  --agent noop \
  --reviewer heuristic \
  --smoke

python cli.py --arena .arena leaderboard
python cli.py --arena .arena queue
```

Use a simple external command mutation agent:

```bash
python cli.py --arena .arena evolve \
  --rp examples/tiny_dummy_rp \
  --steps 1 \
  --agent external-command \
  --agent-command "python examples/agents/simple_mutator.py" \
  --reviewer heuristic \
  --smoke
```

Use Claude Code as the mutation worker:

```bash
python cli.py --arena .arena evolve \
  --rp examples/tiny_dummy_rp \
  --steps 5 \
  --agent claude-code \
  --reviewer heuristic \
  --smoke
```

Use Claude Code for both mutation and Data Scientist review:

```bash
python cli.py --arena .arena evolve \
  --rp /path/to/rp_tiny_cifar \
  --steps 5 \
  --agent claude-code \
  --reviewer claude-code \
  --smoke
```

For full runs, remove `--smoke` and pass RP-specific args after `--`:

```bash
python cli.py --arena .arena evolve \
  --rp /path/to/rp_tiny_cifar \
  --steps 10 \
  --agent claude-code \
  --reviewer heuristic \
  -- --epochs 2 --max-steps 200
```

## External LLM integration contract

For mutation, implement any command that reads JSON on stdin:

```json
{
  "rp_path": ".../.arena/workspaces/run_000001",
  "mutable_file": "model.py",
  "context": "compact evolution context",
  "current_model": "full model.py contents"
}
```

Return JSON. Either edit the workspace directly or return `model_py`:

```json
{
  "mutation_type": "safe_refinement",
  "mutation_summary": "one sentence",
  "hypothesis": "why this should help",
  "changed_files": ["model.py"],
  "model_py": "full replacement contents, optional"
}
```

For review, implement any command that reads:

```json
{
  "parent": {},
  "child": {},
  "context": "..."
}
```

and returns:

```json
{
  "valid": true,
  "is_improvement": true,
  "branch_recommendation": "continue",
  "observation": "what happened",
  "next_belief": "what this suggests",
  "recommended_next_mutations": [
    {
      "mutation_type": "safe_refinement",
      "description": "bounded next idea",
      "expected_benefit": "why",
      "priority": 0.7
    }
  ]
}
```

## Serve a tiny API

```bash
python cli.py --arena .arena serve --port 8000
```

Endpoints:

```text
GET /leaderboard
GET /pareto
GET /queue
GET /runs/{run_id}
GET /runs/{run_id}/lineage
GET /search?q=depthwise
POST /runs/register
```

## Design

```text
RP folder pointer
  -> isolated workspace per run
  -> MutationAgent interface
  -> Claude Code / external-command / noop adapters
  -> single-file edit validation
  -> async Executor interface
  -> RP writes metrics/events
  -> extractor maps metrics to RunRecord
  -> Reviewer interface
  -> RunArtifacts snapshots files/review/context
  -> EvolutionStore writes JSONL + LanceDB and updates NetworkX
  -> queue / leaderboard / Pareto / lineage / search / context packet
```

The code is intentionally small:

```text
core/rp.py              RP contract loader
core/workspace.py       isolated workspaces + single-file validation
core/agent.py           MutationAgent, ClaudeCodeAgent, ExternalCommandAgent
core/orchestrator.py    select→mutate→execute→review→queue loop
core/executor.py        Executor interface, LocalExecutor, ModalExecutor stub
core/store.py           LanceDB + NetworkX write-through store
core/run_store.py       filesystem artifacts and manifests
core/extractor.py       metrics.json -> RunRecord
core/pareto.py          leaderboard and Pareto front
core/context_builder.py compact Claude Code context packet
core/selection.py       simple parent priority
core/queue.py           tiny durable mutation queue
core/review.py          Reviewer interface, heuristic + Claude/external adapters
core/print_util.py      print + JSONL event helper
core/api.py             FastAPI wrapper
cli.py                  minimal CLI
```

## Modal connector

`ModalExecutor` is intentionally only an interface stub. Implement it by:

1. uploading/copying the isolated RP workspace,
2. running the same command generated by `ResearchProblem.command()`,
3. downloading the RP run folder,
4. returning a `RunRecord`.

No other core code should need to change.
