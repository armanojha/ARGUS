# ARGUS — OpenCode Operating Contract

## Project Memory

The ARGUS project vault is located at:

E:\ARGUS_VAULT

The vault is the authoritative source of project state, architecture,
phase specifications, and handoffs.

Before modifying code, read:

1. E:\ARGUS_VAULT\00_CLAUDE_BOOT.md
2. E:\ARGUS_VAULT\01_MASTER_PHASE_INDEX.md
3. E:\ARGUS_VAULT\02_PROJECT_STATE.md
4. E:\ARGUS_VAULT\03_ARCHITECTURE_DECISIONS.md
5. E:\ARGUS_VAULT\handoffs\CURRENT_HANDOFF.md
6. The active phase file specified by the master phase index.

## Rules

- Inspect the existing repository before writing code.
- Follow the active phase specification.
- Respect phase dependencies.
- Do not silently change architectural decisions.
- Do not modify unrelated phases.
- Run relevant tests after implementation.
- Keep implementation state synchronized with the vault.
- Never expose or commit API keys or secrets.
- Use Git commits as recovery points.

## Completion

A task is complete only after:

1. Implementation
2. Tests
3. Validation
4. State/handoff update
5. Git commit