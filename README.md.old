
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

The only user-facing entry point is a Research Problem (RP) folder. The RP must contain (all the rest files if any must be hidden from model):

```text
goal_prompt.md
train_eval.py
model.py
```


Install extra requirements 

Setup and configure the Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

set the `ANTHROPIC_API_KEY` var in `.env` or use command
`claude config set apiKey "sk-ant-..."`

## One-shot run against an RP

```bash
python cli.py --arena .arena init --rp examples/tiny_rp

python cli.py --arena .arena run \
  --rp examples/tiny_rp \
  --smoke \
  --mutation-type baseline \
  --mutation-summary "initial smoke baseline"

python cli.py --arena .arena leaderboard
python cli.py --arena .arena pareto
python cli.py --arena .arena context --rp examples/tiny_rp
```

Extra arguments after the CLI flags are passed to the RP train script:

```bash
python cli.py --arena .arena run --rp examples/tiny_rp -- --epochs 2 --max-steps 200
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
  --rp examples/tiny_rp \
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
  --rp examples/tiny_rp \
  --steps 1 \
  --agent external-command \
  --agent-command "python examples/agents/simple_mutator.py" \
  --reviewer heuristic \
  --smoke
```

Use Claude Code as the mutation worker:

```bash
python cli.py --arena .arena evolve \
  --rp examples/tiny_rp \
  --steps 5 \
  --agent claude-code \
  --reviewer heuristic \
  --smoke
```

Use Claude Code for both mutation and Data Scientist review:

```bash
python cli.py --arena .arena evolve \
  --rp examples/tiny_rp \
  --steps 5 \
  --agent claude-code \
  --reviewer claude-code \
  --smoke
```

For full runs, remove `--smoke` and pass RP-specific args after `--`:

```bash
python cli.py --arena .arena evolve \
  --rp examples/tiny_rp \
  --steps 10 \
  --agent claude-code \
  --reviewer heuristic \
  -- --epochs 2 --max-steps 200
```

## Cloud Run execution

Use the same `run` and `evolve` commands with `--executor cloud-run`. The executor calls
`gcp_cloud_runner/application_cloud_runner.py`, passes the evolver run id through to
Cloud Run, appends RP args after `--` to the YAML `runtime.command`, and syncs outputs
back into the same canonical artifact directory used locally:

```text
.arena/runs/<run_id>/
  model.py
  goal_prompt.md
  metrics.json
  events.jsonl
  run_summary.md
  stdout.txt
  stderr.txt
  manifest.json
  _acr/...              # Cloud Runner metadata/logs when using --executor cloud-run
```

The RP should write outputs under `$ACR_ARTIFACT_DIR`. `train_eval.py` already does this,
and also reads `DATASET_DIR` when the Cloud Runner downloads a configured dataset.

```bash
python cli.py --arena .arena evolve \
  --rp /path/to/tiny-cifar \
  --steps 5 \
  --agent claude-code \
  --reviewer heuristic \
  --executor cloud-run \
  --cloud-spec /path/to/tiny-cifar/cloud_runner.yaml \
  -- --dataset cifar10 --epochs 2 --max-steps 200
```

If `--cloud-spec` is omitted, the executor uses `<rp>/cloud_runner.yaml`.

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
