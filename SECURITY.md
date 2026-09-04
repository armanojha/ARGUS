# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in ARGUS, please report it responsibly:

1. **Do NOT** open a public GitHub Issue for security vulnerabilities
2. Email the maintainer directly (see README for contact)
3. Include a description of the vulnerability and steps to reproduce
4. Allow time for a fix before public disclosure

## API Key Security

ARGUS requires API keys for LLM providers (Groq, Gemini, Cerebras, Zen). These keys:

- Are stored in `.env` (gitignored, never committed)
- Are referenced via environment variables in configuration
- Should never be hardcoded in source code or configuration files
- Should be rotated regularly

### Best Practices

- Copy `.env.example` to `.env` and fill in your keys
- Never commit `.env` to version control
- Use different API keys for development and production
- Monitor your API usage for unexpected activity
- Set up spending limits where supported by the provider

## Data Security

- ARGUS stores evidence in a local SQLite database (`data/evidence.db`)
- Retrieval indexes are stored locally (`data/indexes/`)
- Telemetry data stays local (`data/telemetry/`)
- No data is sent to external services except the configured LLM providers

## LLM Provider Security

When using ARGUS with LLM providers:

- Your queries and evidence are sent to the configured provider's API
- Review the provider's data privacy policy
- Be cautious with sensitive or confidential information
- Consider using providers that do not retain query data for training

## Dependencies

ARGUS uses well-maintained open-source dependencies. Run `pip audit` regularly to check for known vulnerabilities.
