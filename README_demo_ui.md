# AI Evolver demo UI

A read-only NiceGUI dashboard for the Stable Evolver arena.

## Install

```bash
pip install nicegui
```

## Add to the project

```bash
mkdir -p core
cp ui_demo.py core/ui_demo.py
patch -p0 < cli_demo_ui.patch
```

## Run

```bash
python cli.py --arena .arena demo-ui --port 8080
```

Or run the UI file directly:

```bash
python core/ui_demo.py --arena .arena --port 8080
```

## What it reads

The UI does not use FastAPI and does not write to the arena. It polls these files:

```text
.arena/evolution.jsonl
.arena/queue.json
.arena/runs/<run_id>/manifest.json
.arena/runs/<run_id>/metrics.json
.arena/runs/<run_id>/events.jsonl
.arena/runs/<run_id>/ds_review.json
.arena/runs/<run_id>/mutation.json
```

## What it shows

- live branch constellation with parent-child edges, latest run pulse, best run highlight, and Pareto orbit
- best score, run counts, queue/running/failed status
- score skyline over time using inline SVG, not a chart dependency
- selected branch score, accuracy, loss, params, bytes, latency, train time, and dry-run status
- reviewer decision cockpit with recommendation, confidence, observation, next belief, and suggested next mutations
- leaderboard, queue, and agent/executor/reviewer event stream

