# SkeletonGraph Golden Dataset Master List (V1.0)

This dataset defines the official 25-task benchmark for the SkeletonGraph context engine. 

## 1. Metric Schema
- **TASKS:** 25 total.
- **CHATS:** 50 (Control vs Intervention).
- **METRICS:**
  - **TRR (Token Reduction Ratio):** (Native Tokens / SG Tokens). Target: > 4.0x.
  - **SR (Success Rate):** Did the AI fix correctly? (Pass/Fail).
  - **P-Score (Precision):** % of retrieved code used in the final fix.

---

## 2. The 25 Golden Tasks

### Python (The SWE-bench Core)
| Task ID | Repository | Issue / Topic | Commit / State | Human Prompt (Developer Style) |
| :--- | :--- | :--- | :--- | :--- |
| **PY-01** | `pallets/flask` | Routing `strict_slashes` | `Flask@4.0.x` | "Trailing slash routing is skipping global config. Fix the Blueprint init logic." |
| **PY-02** | `psf/requests` | SSL Timeout Logic | `Requests@2.25` | "SSL timeouts are being swallowed in `adapters.py`. Ensure the exception propagates." |
| **PY-03** | `django/django` | Model Inheritance | `Django@3.2` | "Recursive Relation lookups are failing in multi-table inheritance. Audit the SQL generator." |
| **PY-04** | `flask` | Teardown Race | `Flask@3.0` | "The `teardown_request` function is firing before the context is closed. Sync the lifecycle." |
| **PY-05** | `requests` | Header 'None' Case | `Requests v2.x`| "Implicitly passing None to headers causes a crash in the preparation loop. Handle it." |
| **PY-06** | `django` | Sqlite JSON Field | `Django @4.0` | "SQLite JSON field lookups are ignoring nested keys. Fix the indexer." |
| **PY-07** | `flask` | Cookie Expiration | `Flask @2.x` | "Session cookies aren't respecting the `PERMANENT_SESSION_LIFETIME` across blueprints." |
| **PY-08** | `scikit-learn` | Array Input Logic | `Sklearn 1.x` | "Inconsistent array input checks across the SVC model. Standardize the validation call." |
| **PY-09** | `requests` | Stream Chunking | `Requests 2.x` | "Chunked transfer encoding is failing when the stream is empty. Fix the yield loop." |
| **PY-10** | `django` | Migration Rename | `Django 3.x` | "Renaming a M2M field in migrations fails to update the junction table. Fix the operation." |

### JavaScript & TypeScript
| Task ID | Repository | Issue / Topic | Commit / State | Human Prompt (Developer Style) |
| :--- | :--- | :--- | :--- | :--- |
| **JS-01** | `expressjs/express`| Router Param Order | `Express 4.x` | "Parameters in the router URL are being parsed in the wrong order for nested routers." |
| **JS-02** | `facebook/react` | Fiber Work Loop | `React 18.x` | "Task priorities are leaking from the high-priority queue into the idle loop. Audit the scheduler." |
| **JS-03** | `expressjs` | Body-Parser Order | `Express 4.x` | "Middleware registered after body-parser is losing request body context. Fix the stack flow." |
| **JS-04** | `typescript` | Generic Inference | `TS 5.x` | "Nested generic inference is failing for deep interfaces. Investigate the type checker." |
| **JS-05** | `nodejs/node` | HTTP2 Reset | `Node v20.x` | "The HTTP2 stream reset is causing a memory leak in the handle pool. Close the reference." |

### Go (Golang)
| Task ID | Repository | Issue / Topic | Commit / State | Human Prompt (Developer Style) |
| :--- | :--- | :--- | :--- | :--- |
| **GO-01** | `gin-gonic/gin` | Context Data Race | `Gin v1.7` | "The context pool is racing during concurrent resets. Add a mutex lock to the recycler." |
| **GO-02** | `gin-gonic/gin` | Logger Path Fix | `Gin v1.6` | "The logger is outputting raw escape codes in Windows terminals. Force ASCII fallback." |
| **GO-03** | `golang/go` | Garbage Collect | `Go 1.22` | "Minor heap leak in the net/http transport layer. Ensure the connection is fully drained." |

### Rust, Java, C++, C#
| Task ID | Repository | Language | Issue / Topic | Human Prompt |
| :--- | :--- | :--- | :--- | :--- |
| **RS-01** | `tokio-rs/tokio` | Rust | Task Budget | "Task budget is being ignored in the local-set scheduler. Fix the increment logic." |
| **RS-02** | `rust-lang/rust` | Rust | Lifetime Inference | "Implicit lifetime inference fails on trait objects with multiple bounds. Fix the resolver."|
| **JV-01** | `spring-boot` | Java | Prop Binding | "YAML property binding is failing for nested lists in the configuration processor." |
| **JV-02** | `apache/commons` | Java | NullPointer | "Standardize the defensive null checks across the Lang library's StringUtils." |
| **CPP-01** | `llvm/llvm` | C++ | Pass Manager | "Out-of-line method definitions in the new PassManager are missing from the build. Merged FQN." |
| **CS-01** | `dotnet/runtime` | C# | Async State | "Async state machine is leaking memory in the WebHost loop. Fix the awaiter cleanup." |
| **PHP-01** | `symfony/symfony` | PHP | Proxy Refresh | "Lazy-loaded proxies are failing to refresh the cache after entity updates. Clear the metadata." |

---

## 3. The 4-Metric Evaluation Sheet (Gemini Judge)

For every task, record the following into your Judge Prompt:

1. **SR (Success Rate):** Did the Agent solve the PR correctly with the context? (0/1)
2. **Native Tokens:** total tokens used by the agent using `grep/read`.
3. **SG Tokens:** total tokens used by the agent using `resolve_context`.
4. **Multiplier (TRR):** `Native Tokens / SG Tokens`.

**Goal:** Achieve an average **TRR of >4.0x** while maintaining **SR > 0.90**. 
