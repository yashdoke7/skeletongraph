# SkeletonGraph Setup for Cursor IDE

## Quick Start (No Manual Setup)

```bash
cd C:\Users\ASUS\Desktop\CS\Projects\skeletongraph
pip install -e .

cd <your-project-root>
python -m skeletongraph build --path . --auto-infer
```

That's it. Everything is auto-configured.

## What `--auto-infer` Does

On first build, automatically:
1. **Infers project description** from README/setup.py using LLM
2. **Detects constraints** from existing code patterns
3. **Determines phase** (active/maintenance/deprecated) from git history
4. **Finds architecture patterns** from module structure
5. **Creates** `.skeletongraph/project.md` and `.skeletongraph/architecture.md`
6. **Builds index** with docstring-first + BM25 enabled
7. **Extracts all docstrings** as summaries (no LLM summarization needed)

No prompts, no questions—fully autonomous.

## Manual Setup (If You Prefer Prompts)

```bash
cd <your-project-root>
python -m skeletongraph build --path .
# Will ask 4 questions, then build
```

## 1. Install SkeletonGraph Package

```bash
cd C:\Users\ASUS\Desktop\CS\Projects\skeletongraph
pip install -e .
```

This installs SkeletonGraph in development mode (editable), so changes to source code take effect immediately.

## 2. Verify Installation

```bash
python -m skeletongraph --help
```

Should show the CLI help menu. If not, Python/pip environment issue.

## 3. Initialize SkeletonGraph for Your Project

### Option A: Auto-Infer (Recommended)
```bash
cd <your-project-root>
python -m skeletongraph build --path . --auto-infer
```
- No manual input needed
- LLM infers project metadata automatically
- Creates .skeletongraph/ with config ready to go

### Option B: Interactive Setup
```bash
cd <your-project-root>
python -m skeletongraph build --path .
```
You'll be prompted for:
- Project description (1-2 sentences)
- Constraints to preserve
- Project phase (active/deprecated/etc)
- Choose IDE agent (Cursor, Claude Code, etc)

This will:
- Create `.skeletongraph/` directory
- Extract all functions/classes using tree-sitter
- Build the skeleton index
- Extract docstrings (docstring-first mode enabled by default)
- Set up config with BM25 fallback enabled

## 4. Verify Build Success

```bash
# Check if config was created
cat .skeletongraph/config.json | findstr "enable_bm25_fallback"
# Should show: "enable_bm25_fallback": true

# Check skeleton index
ls -la .skeletongraph/
# Should see: domain/, config.json, project.md, architecture.md
```

## 5. Test Query from CLI

```bash
# Query the index (tests docstring-first + BM25)
python -m skeletongraph query "authenticate user" --path .

# Should return ranked functions with:
# - Docstrings as summaries (not LLM summaries)
# - BM25 fallback if no direct entity match
```

## 6. MCP Server for Cursor Integration

```bash
# Start the MCP server (runs in background)
python -m skeletongraph server

# This exposes SkeletonGraph as an MCP tool for Cursor
# Cursor will auto-discover it in its agent tools
```

Once running, Cursor's agent can use:
- `query_context` - Retrieve skeleton context for a prompt
- `get_retrieval_context` - Get detailed context with reasoning
- `pack_context` - Assemble minimal context packet

## 7. Environment Variables (Optional)

```bash
# Enable all features (already defaults to true)
$env:SG_ENABLE_BM25_FALLBACK = "true"
$env:SG_SUMMARY_DOCSTRINGS = "true"
$env:SG_AUTO_REBUILD = "true"

# Set custom model for CLI queries
$env:SG_CLI_MODEL = "claude-opus-4-7"
```

## Agent Rules for Cursor

**Read [AGENT_RULES.md](AGENT_RULES.md) for mandatory agent rules.**

Key points:
- Always use `query_context` for codebase understanding (never manual search)
- Update docstrings when adding functions (not SG config)
- Update `.skeletongraph/project.md` when constraints change
- Call `report_completion()` after finishing tasks to rebuild index
- Never disable BM25 fallback or docstring-first mode

## Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -e .` (in skeletongraph dir) |
| Build (auto-infer) | `python -m skeletongraph build --path . --auto-infer` |
| Build (interactive) | `python -m skeletongraph build --path .` |
| Query | `python -m skeletongraph query "search term" --path .` |
| Start MCP server | `python -m skeletongraph server` |
| Check config | `cat .skeletongraph/config.json` |
| View help | `python -m skeletongraph --help` |

## Cursor Integration Checklist

- [ ] Run `pip install -e .` in SkeletonGraph project
- [ ] Run `python -m skeletongraph build --path . --auto-infer` in target project
- [ ] Verify `.skeletongraph/config.json` has `enable_bm25_fallback: true`
- [ ] Run `python -m skeletongraph server` to start MCP server
- [ ] In Cursor: Open any file in target project and use agent
- [ ] Agent should use `query_context` tool automatically
- [ ] Verify docstrings are returned as summaries (not LLM text)

## Docstring-First + BM25 Behavior in Cursor

When Cursor's agent queries SkeletonGraph:

1. **Entity Matching**: Agent asks for "authenticate" function
   - SG finds `authenticate_user` directly → returns with docstring as summary
   - ✅ Fast, no fallback needed

2. **BM25 Fallback**: Agent asks "how to verify credentials"
   - No direct function match for "verify credentials"
   - SG runs BM25 search over docstring corpus
   - Finds `verify_password_hash` (BM25 ranks it high)
   - Returns with docstring + BM25 match score
   - ✅ Semantic fallback without LLM overhead

3. **Comment Update**: Agent finds summary is inaccurate
   - Updates function docstring directly
   - Calls `report_completion()` to trigger auto-rebuild
   - Next query uses updated docstring
   - ✅ Agent-driven comment hygiene

## Troubleshooting

**`python -m skeletongraph` command not found**
```bash
# Reinstall package
pip install --force-reinstall --no-cache-dir -e C:\Users\ASUS\Desktop\CS\Projects\skeletongraph
```

**`.skeletongraph/` not created**
```bash
# Check Python version (needs 3.9+)
python --version

# Try explicit build
python -m skeletongraph build --path .
```

**Docstrings not extracted**
```bash
# Test extraction directly
python -c "from src.skeletongraph.parser.ast_extractor import extract_file; print(extract_file('yourfile.py', Path('.')).functions)"
```

**BM25 fallback not working**
```bash
# Verify config
cat .skeletongraph/config.json | findstr "enable_bm25_fallback"

# If false, enable it
# Edit .skeletongraph/config.json: "enable_bm25_fallback": true
```

**Auto-infer didn't work (LLM call failed)**
```bash
# Falls back gracefully to defaults
# Check your LLM model is available:
echo $env:SG_CLI_MODEL

# Try manual setup instead:
python -m skeletongraph build --path .
```
