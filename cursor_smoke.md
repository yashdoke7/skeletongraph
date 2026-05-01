# Cursor Smoke Test: SWE-bench `requests-1142`

This document outlines the complete workflow for benchmarking the **Cursor** agent against the `requests` codebase for SWE-bench issue `requests-1142`.

---

## Phase 1: Environment Setup
*(Run these in your terminal)*
1. `git clone https://github.com/psf/requests.git requests-smoke-test`
2. `cd requests-smoke-test`
3. `git checkout 22623bd8`
4. **Patch Python 3.11:** Change `collections.MutableMapping` to `collections.abc.MutableMapping` in `requests/packages/urllib3/_collections.py` and `requests/cookies.py`.
5. `pip install -e .` and `pip install pytest`
6. **Inject Test:** Add `test_no_content_length` to `test_requests.py`.
7. **Verify Bug:** `pytest test_requests.py::RequestsTestCase::test_no_content_length` (Should fail).

---

## Phase 2: Native Baseline (Cursor)
1. Ensure SkeletonGraph is **OFF** (or not configured in `.cursor/mcp.json`).
2. Open the Cursor Chat/Composer.
3. Paste the prompt:
   > "Hi, It seems like that request.get always adds 'content-length' header to the request. I think that the right behavior is not to add this header automatically in GET requests or add the possibility to not send it. For example http://amazon.com returns 503 for every get request that contains 'content-length' header. Thanks, Oren"
4. Let Cursor fix the issue and verify with `pytest`.
5. **Export:** Export the Cursor chat (usually via the chat menu -> Export as JSON/Markdown) and save it as `native_transcript.json` (or `.md`).
6. **Parse:** `skeletongraph eval-parse --agent cursor --file native_transcript.json --path .`

---

## Phase 3: Preparing for SG
1. `git checkout requests/models.py` (Revert the fix)
2. `skeletongraph build` (Build the index)
3. `skeletongraph install cursor` (This creates `.cursorrules` to force the workflow)

---

## Phase 4: SkeletonGraph Test (Cursor)
1. Ensure the SkeletonGraph MCP server is running and configured for Cursor.
2. Start a **NEW** Cursor chat (to pick up the `.cursorrules`).
3. Paste the prompt again.
4. Let Cursor fix it (it should immediately call `query_context`).
5. **Export:** Export the chat as `sg_transcript.json`.
6. **Parse:** 
   ```powershell
   skeletongraph eval-parse --agent cursor --file sg_transcript.json --path .
   Move-Item .skeletongraph\eval\native_trace.json .skeletongraph\eval\sg_trace.json
   ```

---

## Phase 5: Report
```powershell
skeletongraph eval-compare --path . --output final_report.md
```
