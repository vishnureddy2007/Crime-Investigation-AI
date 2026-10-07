# Documentation

Markdown documentation for the **AI-Based Crime Investigation Assistant**.
Four core docs, one index.

| Document | Read this if you want to... |
|---|---|
| [INSTALLATION.md](INSTALLATION.md) | Install the app, run it locally, and verify the setup. |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Run the app via one-command launcher or Docker; env-var overrides. |
| [DEVELOPER.md](DEVELOPER.md) | Understand the architecture and extend the codebase. |
| [API.md](API.md) | Look up a function signature, dataclass field, or session-state key. |
| [PROGRESS.md](PROGRESS.md) | See what each milestone delivered and how the test suite grew. |

## How to read this docs set

- **Reviewers / evaluators:** start with [INSTALLATION.md](INSTALLATION.md)
  to see how to run the app, then skim [PROGRESS.md](PROGRESS.md) for
  what each milestone produced.
- **New contributors:** start with [DEVELOPER.md](DEVELOPER.md) for
  architecture + extension cookbook, then keep [API.md](API.md) open as
  you read code.
- **Anyone debugging:** the [API.md](API.md) session-state-key table is
  the fastest way to figure out where a piece of data lives.

All diagrams in these docs are ASCII, rendered inline — no external
images, no PDFs. The docs track the current `APP_VERSION` in
`config/settings.py`.