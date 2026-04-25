import json
import os
import pathlib
import time

artifact_dir = pathlib.Path(os.environ.get("ACR_ARTIFACT_DIR", "./artifacts"))
artifact_dir.mkdir(parents=True, exist_ok=True)

# Example: consume env vars passed by --env from the parent pipeline.
epochs = int(os.environ.get("EPOCHS", "2"))
lr = float(os.environ.get("LR", "0.001"))
print("EPOCHS:", epochs)
print("LR:", lr)

metrics = []
for epoch in range(epochs):
    # Replace this with real torch training/eval.
    loss = 1.0 / (epoch + 1)
    metrics.append({"epoch": epoch, "loss": loss, "lr": lr})
    print(json.dumps({"event": "train_epoch", "epoch": epoch, "loss": loss}), flush=True)
    time.sleep(1)

(artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(artifact_dir / "model.txt").write_text("pretend model checkpoint\n", encoding="utf-8")
print(json.dumps({"event": "train_done", "artifact_dir": str(artifact_dir)}), flush=True)
