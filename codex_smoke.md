# Codex Smoke Test: SWE-bench `requests-1142`

This document outlines the complete workflow for benchmarking the **Codex** agent against the `requests` codebase for SWE-bench issue `requests-1142`.

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

## Phase 2: Native Baseline (Codex)
1. Ensure SkeletonGraph is **OFF**.
2. Open Codex Chat.
3. Paste the prompt:
   > "Hi, It seems like that request.get always adds 'content-length' header to the request. I think that the right behavior is not to add this header automatically in GET requests or add the possibility to not send it. For example http://amazon.com returns 503 for every get request that contains 'content-length' header. Thanks, Oren"
4. Let Codex fix the issue and verify with `pytest`.
5. **Export:** Export the Codex chat session to a file (e.g., `native_transcript.md`).
6. **Parse:** `skeletongraph eval-parse --agent codex --file native_transcript.md --path .`

---

## Phase 3: Preparing for SG
1. `git checkout requests/models.py` (Revert the fix)
2. `skeletongraph build` (Build the index)
3. `skeletongraph install codex` (This creates `AGENTS.md` to force the workflow)

---

## Phase 4: SkeletonGraph Test (Codex)
1. Ensure the SkeletonGraph MCP server is configured and active.
2. Start a **NEW** Codex chat (to pick up the `AGENTS.md` rules).
3. Paste the prompt again.
4. Let Codex fix it (it should immediately call `query_context`).
5. **Export:** Export the chat as `sg_transcript.md`.
6. **Parse:** 
   ```powershell
   skeletongraph eval-parse --agent codex --file sg_transcript.md --path .
   Move-Item .skeletongraph\eval\native_trace.json .skeletongraph\eval\sg_trace.json
   ```

---

## Phase 5: Report
```powershell
skeletongraph eval-compare --path . --output final_report.md
```
