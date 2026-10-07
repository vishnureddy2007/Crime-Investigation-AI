# Installation Guide

This document walks you through getting the **AI-Based Crime Investigation
Assistant** running on your machine.

> **Audience:** anyone who wants to *use* the app or evaluate the project.
> If you plan to *develop* or extend the code, see
> [`DEVELOPER.md`](DEVELOPER.md) after completing these steps.

---

## 1. Overview

The application is a Python 3.10+ project that ships:

- A **Streamlit** web dashboard (multi-page).
- An **YOLOv8** object detector for crime-relevant objects.
- A **FLAN-T5** language model for AI summaries.
- An **OpenCV + Pillow** pipeline for video frames and storyboards.
- A local **SQLite** database for case history.

Everything is open source. No paid API keys. No cloud accounts.

---

## 2. Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python      | 3.10    | 3.11 or 3.12 |
| Disk space  | 3 GB    | 5 GB        |
| RAM         | 4 GB    | 8 GB        |
| Internet    | Required on first run only (model downloads) | — |
| OS          | Windows 10/11, macOS 12+, Ubuntu 20.04+ | — |

> **Note:** The YOLOv8 nano weights are ~6 MB. The FLAN-T5 base model is
> ~990 MB. Both are cached locally after the first run, in
> `~/.cache/huggingface/` and `~/.cache/torch/`.

---

## 3. Quick start

```bash
# 1. Clone
git clone https://github.com/<your-username>/Crime-Investigation-AI.git
cd Crime-Investigation-AI

# 2. Create a virtual environment
python -m venv venv
# Activate:
#   Windows (Git Bash / PowerShell):
venv\Scripts\activate
#   macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Open <http://localhost:8501> in a browser. The sidebar lists all pages.

---

## 4. Detailed install per OS

### Windows (PowerShell)

```powershell
git clone https://github.com/<your-username>/Crime-Investigation-AI.git
cd Crime-Investigation-AI
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks script activation, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### macOS / Linux

```bash
git clone https://github.com/<your-username>/Crime-Investigation-AI.git
cd Crime-Investigation-AI
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

---

## 5. Verifying the install

After `streamlit run app.py` opens, you should see:

```
🔍 AI Crime Investigation Assistant
A lightweight AI application ...
Version 0.11.0
```

To confirm the test suite passes:

```bash
pip install -r requirements-dev.txt
pytest -q
```

Expected: **360+ test functions** pass with no failures, ~91% line
coverage across `app/`, `core/`, `services/`, `models/`, `utils/`,
`database/`, `config/`.

---

## 6. First-run model download

The first time you reach the **Image Detection** page or the
**AI Summary** page, the app will download:

- `yolov8n.pt` (~6 MB) — Ultralytics-hosted.
- `google/flan-t5-base` (~990 MB) — HuggingFace-hosted.

This happens once. Subsequent runs use the local cache.

---

## 7. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `streamlit: command not found` | venv not activated | Activate the venv (see step 2). |
| `Port 8501 is already in use` | Another Streamlit running | `streamlit run app.py --server.port 8502` |
| `python -m streamlit` works, `streamlit` doesn't | PATH issue on Windows | Use `python -m streamlit run app.py` |
| `ModuleNotFoundError: ultralytics` | Wrong interpreter | Ensure the venv is active (`which python` on macOS/Linux). |
| MP4 download fails on Linux | Missing `libx264` | `sudo apt install ffmpeg libsm6 libxext6` |
| FLAN-T5 download is very slow | Network | Pre-download with `huggingface-cli download google/flan-t5-base` |

---

## 8. Uninstall

```bash
# 1. Deactivate the venv
deactivate

# 2. Remove the project folder (this also removes the SQLite DB)
rm -rf Crime-Investigation-AI

# 3. (Optional) Remove cached model weights
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
```

That is the entire footprint. Nothing else to clean up.

---

## Next steps

- Browse the [API reference](API.md) to see every public symbol.
- Read the [Developer Guide](DEVELOPER.md) if you plan to modify the code.
- Check the [Progress table](PROGRESS.md) for what each milestone shipped.