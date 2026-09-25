# Changelog

All notable user-facing changes are documented here. This project follows semantic
versioning.

## 0.2.0 — 2026-09-25

### Added

- `SG_MCP_PLAIN=1`: serves the same retrieval with no behavioural guidance in tool
  descriptions or results, for use without the instructions `sg install` writes.
- The evaluation behind docs/RESULTS.md: frozen task sets, drivers for a ReAct loop,
  Claude Code and Codex CLI, execution-based verification, and the scripts that recompute
  every reported number and figure. Run records and transcripts are attached to this
  release.

### Changed

- Dense retrieval embeds in batches of 32 and frees the previous repository's vector store
  when switching repositories.
- `sg --version` and the MCP handshake read the version from the installed package, so they
  can no longer disagree with PyPI.
- The removed `mcp` extra was never needed; the server ships with the core package.
- Package description now matches what the tool is: zero-LLM structural code retrieval
  served over MCP.
- The source distribution contains only the package, its tests and build files.

### Documentation

- README and docs/RESULTS.md report the full study: the retriever finds the right file more
  often than agents' own search, but frontier agents read the right code on their first tool
  calls anyway, so the gain does not reach the fix; the token effect is a saving for agents
  that explore a lot and a small overhead for lean ones.

## 0.1.1

- Packaging and installation fixes following the initial public release.

## 0.1.0

- Initial public release.
