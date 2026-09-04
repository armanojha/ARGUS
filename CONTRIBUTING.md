# Contributing to ARGUS

Thank you for your interest in contributing to ARGUS! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Include steps to reproduce, expected behavior, and actual behavior
- Include your Python version and OS

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add or update tests as needed
5. Run the test suite: `python -m pytest tests/ -v`
6. Run the linter: `ruff check app/`
7. Commit your changes with a clear message
8. Push to your fork and submit a Pull Request

### Code Style

- Use `from __future__ import annotations` in all modules
- Modern type hints (`dict[str, Any]`, `str | None`)
- Line length: 100 characters
- No hardcoded provider names, model IDs, or API keys
- LLMs are interpreters, not the database of record
- All evidence and provenance in deterministic stores

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting: `python -m pytest tests/ -v`
- Live LLM tests require `RUN_LIVE_LLM_TESTS=1` and valid API keys

### Architecture Principles

- Model assignment is explicit configuration only (`configs/model_policy.yaml`)
- ARGUS never autonomously discovers or ranks models
- Obsidian notes are personal claims, never automatically-trusted evidence
- Provider-agnostic LLM gateway — never hard-code around one provider

## Questions?

Open a GitHub Issue for questions about the project or contribution process.
