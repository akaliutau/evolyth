FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /rp

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /rp/requirements.txt
RUN pip install --no-cache-dir -r /rp/requirements.txt

COPY . /rp

CMD ["python", "train_eval.py", "--dry-run", "--dataset", "synthetic", "--run-id", "docker_dry_run"]
