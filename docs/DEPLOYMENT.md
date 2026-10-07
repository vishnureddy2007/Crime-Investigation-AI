# Deployment Guide

Three free, open-source ways to run the **AI-Based Crime Investigation
Assistant**. Pick the one that matches your environment.

> **Audience:** reviewers, evaluators, and anyone who wants to run the
> app without writing installation commands. Already running it locally?
> See [`INSTALLATION.md`](INSTALLATION.md) for the manual venv flow.

---

## 1. Three ways to run the app

### a. One-command local launcher (recommended)

macOS / Linux:

```bash
./scripts/run.sh
```

Windows (cmd.exe or PowerShell):

```bat
scripts\run.bat
```

What it does:

- Creates a `.venv` on first run.
- Installs `requirements.txt` (skips up-to-date packages on subsequent runs).
- Launches Streamlit on `http://localhost:8501` by default.

```
+------------------+      +-----------------+      +--------------------+
| ./scripts/run.sh | ---> | creates .venv   | ---> | streamlit run app.py|
+------------------+      +-----------------+      +--------------------+
```

### b. Docker

Two flavours are supported.

**Plain Docker:**

```bash
docker build -t crime-ai .
docker run -p 8501:8501 crime-ai
# Then open http://localhost:8501
```

**docker-compose (recommended for any persistent run):**

```bash
docker compose up --build
# or, with a custom .env file:
cp .env.example .env
docker compose --env-file .env up -d
```

The compose stack:

- persists `outputs/`, `logs/`, and the SQLite database as named
  volumes so generated artefacts survive container restarts;
- exposes a healthcheck (`/_stcore/health`);
- reads `APP_PORT`, `APP_THEME`, `LOG_LEVEL` from the environment
  (or `.env` file).

The image is a single-stage `python:3.11-slim` build with `ffmpeg` +
the OpenCV system deps. The whole thing weighs in around **1.2 GB**;
the FLAN-T5 weights (~990 MB) are pulled on the first request to the
**AI Summary** page and cached in the image's `/app/.cache` layer.

```
+-----------+     +---------------------+     +-----------------------------+
| docker    | --> | builds python:3.11  | --> | docker run -p 8501:8501 ... |
| build -t  |     | -slim + ffmpeg      |     | listens on 0.0.0.0:8501    |
+-----------+     +---------------------+     +-----------------------------+
                                                          |
                                                          v
                                              http://localhost:8501
```

The image includes a built-in `HEALTHCHECK` that polls
`http://localhost:8501/_stcore/health` every 30 s.

### c. Manual venv (the classic way)

See [`INSTALLATION.md`](INSTALLATION.md). Use this if you want full
control over which Python or interpreter to use.

---

## 2. Environment variables

All overrides are optional. The app falls back to the local defaults
listed below, so behaviour with no configuration is unchanged from the
previous milestones.

| Variable      | Default                              | Purpose |
|---------------|---------------------------------------|---------|
| `APP_PORT`    | `8501`                                | Streamlit server port. Honoured by `scripts/run.sh`, `scripts/run.bat`, and the Docker `EXPOSE`. |
| `APP_ADDRESS` | `localhost`                           | Bind address. Set to `0.0.0.0` to accept external connections. |
| `APP_HEADLESS`| `false`                               | Set `true` to run Streamlit without auto-opening a browser. |
| `APP_THEME`   | `light`                               | Initial UI theme (`light` or `dark`). Switchable at runtime from **Settings**. |
| `HF_HOME`     | `~/.cache/huggingface`                | Where HuggingFace stores FLAN-T5 weights after first run. |
| `LOG_LEVEL`   | `INFO`                                | Root logger level (`DEBUG` / `INFO` / `WARNING` / `ERROR`). |
| `LOG_FILE`    | `logs/app.log`                        | Rotating log file (10 MB × 5 backups). |

Example:

```bash
APP_PORT=9000 APP_HEADLESS=true ./scripts/run.sh
```

Or, with Docker:

```bash
docker run -e APP_PORT=9000 -p 9000:9000 crime-ai
```

---

## 3. Smoke check (no Streamlit)

A 10-second import-only sanity check that doesn't start the server.
Useful when a sandbox blocks long-running processes.

```bash
python scripts/smoke_app.py
```

Expected output:

```
OK  config
OK  config.settings
OK  database
...
OK  models.dashboard

All imports OK.
```

Any `FAIL <module>` line means a broken import — fix it before
launching the actual app.

---

## 4. Healthcheck

Streamlit exposes a tiny endpoint for load balancers and container
orchestrators:

```bash
curl -fsS http://localhost:8501/_stcore/health
```

- **200 OK** with body `"ok"` — Streamlit is ready to serve pages.
- **000** or non-200 — the server hasn't bound yet (give it ~10 s after
  the first launch).

This is what the Docker `HEALTHCHECK` and the `docker-build` CI job use.

---

## 5. Production notes

> **This is an academic demo, not a production system.** A few specific
> constraints from the original B.Tech brief:

- **Single-user SQLite.** `database/investigations.db` is meant for one
  workstation. Multiple concurrent writers will lead to `database is
  locked` errors. Switch to PostgreSQL before deploying for any shared
  use.
- **Model weights live on disk, not in the image.** The Docker image
  caches FLAN-T5 only after the first request. Cold starts are slow;
  warm starts are fast.
- **Reports are not encrypted.** Generated PDFs / DOCX land in
  `outputs/reports/`. Treat them like drafts.
- **No authentication.** The Streamlit app exposes every page to
  anyone who can reach the port. Bind to `localhost` (`APP_ADDRESS=localhost`)
  unless you've added auth at a reverse proxy.
- **Streamlit is headless in the container.** The `ENTRYPOINT` in
  `Dockerfile` runs with `--server.headless=true`.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Port 8501 is already in use` | Another Streamlit instance or a different app | `./scripts/run.sh --server.port 9000` |
| `Permission denied` on `./scripts/run.sh` | File not executable | `chmod +x scripts/run.sh` |
| `python3: command not found` on macOS | Xcode CLT not installed | `xcode-select --install` then install Python via `brew install python@3.11` |
| Docker build fails on `libsm6` | apt cache stale | `docker build --no-cache -t crime-ai .` |
| First AI Summary request takes 2+ min | FLAN-T5 downloading | Wait; subsequent runs use `HF_HOME` cache. |
| `database is locked` | Two Streamlit instances writing | Close the duplicate; switch to a server DB before scaling up. |
| `ModuleNotFoundError: ultralytics` after a Docker pull | Stale container without the dependency | `docker pull` then rebuild — `requirements.txt` only includes runtime deps. |
| Windows path with spaces | Quote the Python interpreter | Pass the full quoted path to `py -3.11`. |

---

## See also

- [INSTALLATION.md](INSTALLATION.md) — manual venv flow.
- [DEVELOPER.md](DEVELOPER.md) — architecture and extension guide.
- [API.md](API.md) — every public symbol.
- [PROGRESS.md](PROGRESS.md) — what each milestone shipped.