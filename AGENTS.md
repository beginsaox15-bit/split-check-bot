# Agent guide for split-check-bot

This file applies to all coding-agent work in this repository (including Codex).
Read `README.md` and existing tests before editing. Follow the current codebase; do not replace working components just to adopt a framework.

## Project and priorities

- Python 3.12, aiogram 3.x, asyncio, Pydantic, Decimal; Telegram bot for splitting restaurant receipts.
- Existing features: /start, photo and manual receipt entry, recognition, review/edit/confirmation.
- Next product feature: participants, assigning receipt items, and calculating who owes whom. Do not expand OCR experimentation unless the task explicitly concerns OCR.
- Keep modifications minimal and easy to revert. Avoid unrelated refactors and dependencies.

## Correctness and money

- Use `Decimal` (never binary float) for prices, quantities, discounts, subtotals, and splits.
- Preserve item duplicates and per-guest ownership. Never fabricate missing items to force a total to balance.
- Keep distinct: sum of item lines, gross total, prepayment, amount due, guest subtotals.
- If recognized values conflict, report the mismatch and offer editing; do not silently overwrite OCR facts.
- For eventual multi-user features, scope every query and callback to the appropriate Telegram user/company/receipt, guard against stale callbacks, and enforce authorization on the server.
- Design future write operations to avoid duplicate charges/assignments on retries (idempotency), validate input, and prevent concurrent update loss.

## OCR and external services

- Primary backend: OpenRouter vision `nvidia/nemotron-3.5-super-vl:free` only; do not remove `:free`, enable paid fallback, or change model providers without explicit permission.
- Preserve the Tesseract fallback. Do not let a failing remote call crash the Telegram conversation.
- Separate transport status, incomplete responses, JSON/schema validation, and receipt validation. Diagnose the specific failure instead of masking everything as generic OCR failure.
- Structured JSON must pass Pydantic validation before becoming `RecognizedReceipt`. Log safe diagnostics (HTTP status, finish reason, timings, usage, validation error paths, counts, sums) without dumping entire responses by default.
- Treat receipt photos, base64, names, phone numbers, API keys, and Telegram tokens as sensitive. Do not print, commit, or attach them to public logs, fixtures, issues or PRs.
- No real network/API calls in unit or integration tests. Any live provider benchmark is opt-in, once per explicitly requested experiment; report latency, correctness, and cost.

## Testing and reviews

- Before changing code, identify the relevant handler/service and existing test patterns.
- Add regression tests for bugs. Cover success, failure/fallback, cancellation, duplicate messages/callbacks, schema issues, and stale FSM state when relevant.
- For changes to receipt parsing or money calculations, run the receipt benchmark with known fixtures and compare matched/missed/false-positive counts and sum differences.
- Run the targeted tests and then the repository's full unit/integration suite when feasible. Report exact commands, passes, skips, and failures; never claim a test passed if it was not run.
- Review changes for authorization, exposed secrets, rounding, event-loop blocking, timeout/retry explosion, and user-visible Telegram flows.
- Do not add autonomous background agents, desktop automation, general-purpose harnesses, or new external model gateways unless a task clearly requires them and their security/cost implications have been reviewed.
- Keep a single Telegram long-polling instance per BOT_TOKEN; diagnose 409 Conflict rather than launching another one.

## Change discipline

- Show the root cause or evidence before altering production behavior.
- Do not rewrite installation scripts, Codex environment settings, or secrets as part of an unrelated fix.
- Never commit `.env`, tokens, receipt originals, logs with user data, or benchmark artifacts.
- Unless explicitly requested, do not automatically commit, push, merge, deploy, or run a paid service.
- Conclude with: what changed, what tests ran, remaining risks, and the next actionable step.

These are intentionally narrow project instructions inspired by the useful testing/security habits in ECC and backendskills; third-party code and broad agent frameworks are not installed.
