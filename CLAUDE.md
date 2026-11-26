## Language

All repository content must be written in English: commit messages, branch names, issues, pull requests, review comments, code, comments and documentation. This applies to every collaborator and agent, whatever language is used in conversation.

## Agent skills

### Issue tracker

Issues and specs are tracked as GitHub issues on thomas-hinton/stackoverflow-tag-prediction (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Illustrations

When a change touches what a user sees (API docs, demo UI, README, report, or a PR showing a visual change), dispatch a subagent to refresh and embed illustrations, following `docs/agents/illustrations.md`. Add an illustration only where it shows something prose cannot.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
