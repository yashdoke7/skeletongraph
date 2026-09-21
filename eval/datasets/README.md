# Task sets

The frozen task sets every published number was measured on. They are kept
verbatim, so their `repo_path` fields point at the machine the study ran on;
use `eval/scripts/relocate_tasks.py` to write a copy for your own machine
(see `eval/README.md`).

| file | tasks | used by |
|---|---|---|
| `swebench_100.jsonl` | 100 SWE-bench Verified | Claude Code and Codex CLI runs |
| `graphify_100.jsonl` | the same 100 | retrieval-only evaluation and the ReAct loop |
| `swebench_100_prose_stripped.jsonl` | the same 100, code blocks and tracebacks removed from the issue | cue-removal check |
| `swe_rebench_100.jsonl` | 100 SWE-rebench; the study used the first 50 | Claude Code on SWE-rebench |
| `swe_rebench_100_prose.jsonl` | the same, code removed from the issue | cue-removal check on SWE-rebench |
| `main100_task_ids.txt` | the 100 SWE-bench Verified task ids | cross-check |

Each row carries the task id, repository, base commit, issue text (`query`),
and the gold files and functions derived from the reference patch. The
prose-only variants are produced deterministically by
`eval/scripts/make_prose_stripped.py`; no language model is involved.

## Sources and licences

- **SWE-bench Verified** — Jimenez et al., *SWE-bench: Can Language Models Resolve
  Real-World GitHub Issues?* (ICLR 2024), human-validated subset released by
  OpenAI. SWE-bench is MIT-licensed. <https://github.com/SWE-bench/SWE-bench>
- **SWE-rebench** — Badertdinov et al., *SWE-rebench: An Automated Pipeline for Task
  Collection and Decontaminated Evaluation of Software Engineering Agents* (2025),
  released under CC-BY-4.0. <https://huggingface.co/datasets/nebius/SWE-rebench>

Issue text belongs to the upstream projects' GitHub issues.
