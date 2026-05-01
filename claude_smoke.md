# Claude Code Smoke Test: SWE-bench `requests-1142`

This document outlines the complete workflow for benchmarking the **Claude Code** (Anthropic CLI) agent against the `requests` codebase for SWE-bench issue `requests-1142`.

---

## Phase 1: Environment Setup
1. `git clone https://github.com/psf/requests.git requests-smoke-test`
2. `cd requests-smoke-test`
3. `git checkout 22623bd8`
4. **Patch Python 3.11:** Change `collections.MutableMapping` to `collections.abc.MutableMapping` in `requests/packages/urllib3/_collections.py` and `requests/cookies.py`.
5. `pip install -e .` and `pip install pytest`
6. **Inject Test:** Add `test_no_content_length` to `test_requests.py`.
7. **Verify Bug:** `pytest test_requests.py::RequestsTestCase::test_no_content_length` (Should fail).

---

## Phase 2: Native Baseline (Claude Code)
1. Ensure SkeletonGraph is **OFF** (no MCP server configured).
2. Start Claude Code in the terminal: `claude`
3. Paste the prompt:
   > "Hi, It seems like that request.get always adds 'content-length' header to the request. I think that the right behavior is not to add this header automatically in GET requests or add the possibility to not send it. For example http://amazon.com returns 503 for every get request that contains 'content-length' header. Thanks, Oren"
4. Let Claude fix the issue and verify with `pytest`.
5. **Export:** Claude Code auto-saves sessions locally. Just run the parse command and it will auto-discover the latest session.
6. **Parse:** `skeletongraph eval-parse --agent claude_code --path .` (Notice no `--file` is needed, it auto-discovers).

---

## Phase 3: Preparing for SG
1. `git checkout requests/models.py` (Revert the fix)
2. `skeletongraph build` (Build the index)
3. `skeletongraph install claude` (This creates `CLAUDE.md` to force the workflow and configures the MCP server)

---

## Phase 4: SkeletonGraph Test (Claude Code)
1. Start Claude Code again: `claude`
2. Paste the prompt again.
3. Let Claude fix it (it should immediately call `query_context`).
4. **Export:** Again, it auto-saves.
5. **Parse:** 
   ```powershell
   skeletongraph eval-parse --agent claude_code --path .
   Move-Item .skeletongraph\eval\native_trace.json .skeletongraph\eval\sg_trace.json
   ```

---

## Phase 5: Report
```powershell
skeletongraph eval-compare --path . --output final_report.md
```
