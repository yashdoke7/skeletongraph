# SkeletonGraph Agent Rules

SkeletonGraph (SG) is a context assembler for coding agents. Its goal is to
reduce exploratory turns by returning the project briefing and an edit-ready
task packet before you start reading files manually.

## Required Workflow

1. Call `sg_overview` first at the start of a task.
   It is the project briefing: project purpose, important structure, active
   constraints, recent turns/decisions, and index status.

2. Call `sg_search` once with the whole task or symptom.
   Treat it as a task-context assembler, not as grep. For coding/debug tasks it
   returns likely edit targets, imports/prelude, helper bodies, graph neighbors,
   and likely tests. Do not decompose one task into many symbol searches unless
   confidence is LOW/MISS or the target is absent.

3. Use `sg_get` and `sg_expand` only for exact follow-up.
   Expand a specific FQN when you are about to edit it and its body was not
   already returned by `sg_search`. Do not read MCP `content.txt` result files;
   they duplicate SG tool output and increase token cost.

4. Use native file reads only after SG has identified a concrete file/range to
   edit or verify. Prefer small ranges over full files.

5. Check `sg_constraint` before architectural changes, migrations, dependency
   changes, or broad refactors.

6. Use `sg_log` or `sg_decision` for relevant project memory when the task
   depends on prior decisions.

## Good Query Shape

Good:

```text
sg_search("Header.fromstring should accept bytes like Card.fromstring; fix FITS header parsing", intent="debug_targeted")
```

Avoid:

```text
sg_search("Header.fromstring")
sg_search("Card.fromstring")
sg_search("decode_ascii")
sg_search("_pad")
sg_search("test_header.py")
```

The second pattern recreates manual grep and defeats SG's purpose.
