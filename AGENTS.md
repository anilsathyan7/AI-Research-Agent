
# Project Guidelines

## Coding Principles
- Prefer simple, readable, maintainable code.
- Make minimal changes; avoid unnecessary abstractions.
- Reuse existing code and dependencies where possible.
- Preserve existing behavior unless changes are requested.

## Comments and Documentation
- Explain WHY, not obvious WHAT.
- Comment only non-obvious logic and assumptions.
- Use concise Google-style docstrings where useful.

## ML / Deep Learning
- Prevent data leakage across train, validation, and test splits.
- Ensure reproducibility through seeds and saved configurations.
- Validate tensor shapes, dtypes, devices, and model I/O.
- Use appropriate metrics and evaluation procedures.
- Avoid unnecessary computation and GPU memory usage.
- Never fabricate training results or performance metrics.

## Agentic AI / LLM
- Prefer single-agent designs unless multiple agents add value.
- Use structured outputs and validate tool inputs/results.
- Keep prompts and context concise; avoid redundant LLM calls.
- Manage state, memory, and tool execution explicitly.
- Handle tool failures, retries, timeouts, and iteration limits.
- Evaluate agent reliability, accuracy, latency, and cost.
- Never execute untrusted generated code without isolation.

## Verification
- Match testing effort to change complexity.
- Small edits: minimal relevant checks.
- Functional changes: targeted tests.
- Major changes: broader verification when warranted.
- Avoid redundant reviews and repeated checks.
- Never claim tests passed unless executed.

## Skills
- Use `.agents/skills/karpathy-guidelines/SKILL.md`
  for all non-trivial coding tasks.
- Skip for typos, formatting, comments, and trivial edits.
- Read once per task unless necessary again.
- When uncertain, prefer using the skill.

## Output
- Summarize changes and verification results concisely.
- Avoid lengthy explanations unless requested.
