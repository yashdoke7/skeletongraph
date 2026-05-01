# Antigravity Smoke Test: SWE-bench `requests-1142`

This document outlines the complete workflow for benchmarking the Antigravity agent against the `requests` codebase for SWE-bench issue `requests-1142` (Preventing Automatic Content-Length headers on GET requests). 

The goal is to compare the efficiency (tokens and tool calls) of a **Native Agent** (manual exploration) vs a **SkeletonGraph-Enhanced Agent** (graph-based context retrieval).

---

## Phase 1: Environment Setup

1. **Clone the legacy repository:**
   ```powershell
   git clone https://github.com/psf/requests.git requests-smoke-test
   cd requests-smoke-test
   git checkout 22623bd8  # The exact SWE-bench base commit
   ```

2. **Apply Python 3.11 Compatibility Fixes:**
   Because this code is from 2013, it will fail on modern Python. You must patch the `MutableMapping` imports:
   *   In `requests/packages/urllib3/_collections.py`, change `from collections import MutableMapping` to `from collections.abc import MutableMapping`.
   *   In `requests/cookies.py`, update `RequestsCookieJar` to use `collections.abc.MutableMapping`.

3. **Install the editable package & dependencies:**
   ```powershell
   pip install -e .
   pip install pytest
   ```

4. **Inject the SWE-bench Verification Test:**
   Create or append the following test to `test_requests.py` (or a dedicated `test_content_length.py`):
   ```python
   def test_no_content_length(self):
       get_req = requests.Request('GET', httpbin('get')).prepare()
       self.assertTrue('Content-Length' not in get_req.headers)
   ```

5. **Verify the Bug Exists:**
   ```powershell
   $env:PYTHONPATH="C:\path\to\requests-smoke-test"
   pytest test_requests.py::RequestsTestCase::test_no_content_length
   ```
   *Expected result: The test should **FAIL** (`AssertionError: False is not true`).*

---

## Phase 2: Native Agent Baseline Run

1. **Disable SkeletonGraph:** Ensure the SkeletonGraph MCP server is disabled in your IDE.
2. **Start the Agent:** Open a new chat session and paste the official SWE-bench prompt:
   > "Hi, It seems like that request.get always adds 'content-length' header to the request. I think that the right behavior is not to add this header automatically in GET requests or add the possibility to not send it. For example http://amazon.com returns 503 for every get request that contains 'content-length' header. Thanks, Oren"
3. **Verify Fix:** Wait for the agent to find and fix `models.py`. Run pytest again; it should now **PASS**.
4. **Export Transcript:** Save the agent's chat history as `native_transcript.md`.
5. **Parse Native Trace:**
   ```powershell
   skeletongraph eval-parse --agent antigravity --file native_transcript.md --path .
   ```
   *(This extracts tokens and tool calls and saves them to `.skeletongraph/eval/native_trace.json`)*

---

## Phase 3: Preparing for SkeletonGraph

We must revert the codebase to its buggy state while keeping our environment fixes.

1. **Revert the fix:**
   ```powershell
   git checkout requests/models.py
   ```
2. **Clean up spoiler files:**
   Delete any scratchpad scripts or diffs the Native agent left behind, so the SG agent doesn't get unfair clues.
   ```powershell
   Remove-Item native_transcript.md -ErrorAction SilentlyContinue
   ```
3. **Build the SkeletonGraph Index:**
   ```powershell
   skeletongraph build
   ```
4. **Install Agent Rules:**
   Force the agent to use the "Search-First" workflow by injecting the `.antigravity.md` rules file.
   ```powershell
   skeletongraph install antigravity
   ```

---

## Phase 4: SkeletonGraph Test Run

1. **Enable MCP:** Turn the SkeletonGraph MCP server back on.
2. **Start the Agent:** Open a **NEW** chat session so the agent reads the newly installed `.antigravity.md` rules.
3. **Run Prompt:** Paste the exact same Oren prompt.
4. **Observe:** The agent's *very first move* should be a `query_context` tool call, immediately locating `prepare_content_length`.
5. **Verify Fix:** Run pytest to ensure it passes.
6. **Export Transcript:** Save the chat history as `sg_transcript.md`.
7. **Parse SG Trace:**
   ```powershell
   skeletongraph eval-parse --agent antigravity --file sg_transcript.md --path .
   # Rename the output to prevent overwriting the native trace
   Move-Item .skeletongraph\eval\native_trace.json .skeletongraph\eval\sg_trace.json
   ```

---

## Phase 5: Generating the Final Report

With both `native_trace.json` and `sg_trace.json` in the `.skeletongraph/eval/` directory, run the comparison:

```powershell
skeletongraph eval-compare --path . --output final_report.md
```

### Expected Results
You should observe a massive reduction in exploration tool calls (e.g., 16 down to 4) and a ~17x reduction in retrieval tokens, proving that graph-based context assembly is vastly superior to manual LLM filesystem exploration.
