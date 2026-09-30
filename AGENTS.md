# AGENTS.md — anduril-track-envelope-compiler

**Company:** Anduril
**Domain:** Command Authority & Mission Assurance

## Quick Rules
- **Test command:** `PYTHONPATH=src pytest tests/ -v`
- **Lint:** `ruff check src/ tests/`
- **No drive-by edits** — load the skill first.

## Architecture
- `src/anduril_track_envelope_compiler/core.py` — Domain logic (Command Authority & Mission Assurance)
- `tests/` — Verified test suite
- `.github/workflows/ci.yml` — Enforced CI pipeline
