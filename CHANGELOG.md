# Changelog

All notable user-facing changes are documented here. This project follows semantic
versioning.

## 0.2.0 — 2026-09-23

### Added

- Seven-tool MCP server for structural search, repository orientation, symbol context,
  dependency inspection, impact analysis, change tracking, and session state.
- Ten-language tree-sitter indexing and function-level lexical, dense, and structural
  retrieval.
- Claude Code and Codex integration examples, evaluation harnesses, frozen task sets, and
  reproducible analysis scripts for the accompanying repair-agent study.
- `sg doctor`, incremental indexing, session deduplication, and PR blast-radius analysis.

### Changed

- Promoted semantic retrieval to a required dependency so installations do not silently
  fall back to a weaker retrieval configuration.
- Reduced the MCP surface from eleven overlapping tools to seven focused tools.
- Made the CLI and MCP handshake read the installed package version from distribution
  metadata, eliminating version drift between PyPI, `sg --version`, and the server.
- Restricted source distributions to runtime, tests, and build metadata; large evaluation
  records and documentation assets are released separately.

### Fixed

- Corrected stale version reporting that could advertise 0.1.0 from a 0.1.1 installation.
- Added stricter checkout cleanup and contamination auditing to the evaluation workflow.

## 0.1.1

- Packaging and installation fixes following the initial public release.

## 0.1.0

- Initial public release.
