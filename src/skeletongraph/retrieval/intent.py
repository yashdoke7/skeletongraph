"""
Intent analysis: extract entities and classify task type from user prompts.

Multi-signal weighted classification — not just keywords, but entity context,
file mentions, and error patterns.

v4 changes:
  - 10 task types (added TEST, DOCUMENT, MIGRATE, ARCHITECTURE)
  - Two-tier stop-word filter (hard stops always filtered, soft stops only
    filtered when hard entities exist)
  - Extended file pattern for non-code files (.json, .md, .yaml, .toml, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple


class TaskType(Enum):
    """Classification of user intent — 10 types for v4."""
    DEBUG = "debug"              # Fix a bug, resolve an error
    CREATE = "create"            # Add new feature, write new code
    EDIT = "edit"                # Modify existing code
    REFACTOR = "refactor"        # Restructure without changing behavior
    EXPLAIN = "explain"          # Understand code, ask a question
    REVIEW = "review"            # Code review, audit, lint
    TEST = "test"                # Write tests, add coverage
    DOCUMENT = "document"        # Write docstrings, API docs, README
    MIGRATE = "migrate"          # Upgrade APIs, replace deprecated code
    ARCHITECTURE = "architecture"  # Design decisions, system architecture


@dataclass
class Entity:
    """An entity extracted from the user's prompt."""
    value: str
    entity_type: str  # "file_path", "function_name", "class_name", "error_message", "line_number"
    confidence: float = 1.0


@dataclass
class Intent:
    """Parsed intent from a user prompt."""
    task_type: TaskType
    entities: List[Entity]
    file_paths: List[str]       # Mentioned file paths
    function_names: List[str]   # Mentioned function/class names
    error_message: Optional[str] = None
    line_number: Optional[int] = None
    raw_prompt: str = ""


def analyze_intent(prompt: str, known_files: Set[str] = frozenset(),
                   known_fqns: Set[str] = frozenset()) -> Intent:
    """Extract entities and classify task type from a user prompt.

    Args:
        prompt: The user's natural language request.
        known_files: Set of file paths in the project (for matching).
        known_fqns: Set of FQNs in the project (for matching).

    Returns:
        An Intent with classified task type and extracted entities.
    """
    entities: List[Entity] = []
    file_paths: List[str] = []
    function_names: List[str] = []
    error_message = None
    line_number = None

    # ── Entity Extraction ──────────────────────────────────────────────

    # 1. File paths (explicit mentions)
    # Match: "middleware.py", "auth/middleware.py", "src/utils.ts"
    file_pattern = re.compile(
        r'[\w./\\-]+\.(?:py|js|ts|tsx|jsx|mjs|cjs|java|go|rs|cpp|cs|rb|php'
        r'|json|yaml|yml|toml|md|txt|cfg|ini|env)'
    )
    for match in file_pattern.finditer(prompt):
        candidate = match.group().replace("\\", "/")
        entities.append(Entity(candidate, "file_path"))

        # Try to match against known files
        if candidate in known_files:
            file_paths.append(candidate)
        else:
            # Try partial match (basename)
            basename = candidate.split("/")[-1]
            for kf in known_files:
                if kf.endswith(candidate) or kf.endswith("/" + basename):
                    file_paths.append(kf)
                    break
            else:
                file_paths.append(candidate)  # Keep as-is, may resolve later

    # 2. Function/class names (identifiers that match known FQNs)
    # Match: snake_case, camelCase, PascalCase identifiers
    ident_pattern = re.compile(r'\b([a-zA-Z_]\w{2,})\b')
    prompt_lower = prompt.lower()

    # Track whether we found any hard entities (file paths, exact FQN matches)
    has_hard_entities = bool(file_paths)

    for match in ident_pattern.finditer(prompt):
        name = match.group(1)
        name_lower = name.lower()
        # Always skip hard stop words (language keywords, articles, etc.)
        if name_lower in _HARD_STOPS:
            continue
        # Skip soft stop words ONLY if we already have hard entities
        # If we have NO hard entities, keep soft stops as potential search terms
        if has_hard_entities and name_lower in _SOFT_STOPS:
            continue

        # Check if it matches a known file (basename match)
        found_file = False
        for kf in known_files:
            kf_base = kf.split("/")[-1]
            kf_name = kf_base.split(".")[0]
            if kf_name == name.lower() or kf_base == name.lower():
                entities.append(Entity(name, "file_path", confidence=0.85))
                file_paths.append(kf)
                found_file = True
                break
        
        if found_file:
            continue

        # Check if it matches a known FQN suffix (case-insensitive)
        for fqn in known_fqns:
            short = fqn.split("::")[-1] if "::" in fqn else fqn
            if short.lower() == name.lower() or short.endswith(f".{name}"):
                entities.append(Entity(name, "function_name", confidence=0.95))
                function_names.append(name)
                break

    # 3. Error messages
    error_patterns = [
        re.compile(r'(?:Error|Exception|Traceback)[\s:]+(.+)', re.IGNORECASE),
        re.compile(r'traceback.*?:\s*(.+)', re.IGNORECASE),
        re.compile(r'"([^"]*(?:Error|Exception)[^"]*)"'),
    ]
    for pat in error_patterns:
        m = pat.search(prompt)
        if m:
            error_message = m.group(1).strip()[:200]
            entities.append(Entity(error_message, "error_message"))
            break

    # 4. Line numbers
    line_pattern = re.compile(r'(?:line\s+|L|:)(\d+)')
    line_match = line_pattern.search(prompt)
    if line_match:
        line_number = int(line_match.group(1))
        entities.append(Entity(str(line_number), "line_number"))

    # ── Task Classification ────────────────────────────────────────────

    task_type = _classify_task(prompt_lower, entities)

    return Intent(
        task_type=task_type,
        entities=entities,
        file_paths=file_paths,
        function_names=function_names,
        error_message=error_message,
        line_number=line_number,
        raw_prompt=prompt,
    )


def _classify_task(prompt_lower: str, entities: List[Entity]) -> TaskType:
    """Multi-signal weighted task classification."""

    scores = {task: 0.0 for task in TaskType}

    # Signal 1: Keywords
    _KEYWORD_SIGNALS = {
        TaskType.DEBUG: [
            "fix", "bug", "error", "broken", "crash", "fail",
            "not working", "issue", "traceback", "exception",
            "wrong", "incorrect", "doesn't work", "debug",
        ],
        TaskType.CREATE: [
            "add", "create", "implement", "build", "new",
            "feature", "write", "generate", "scaffold",
        ],
        TaskType.EDIT: [
            "change", "modify", "update", "edit", "set",
            "replace", "rename", "adjust", "configure",
        ],
        TaskType.REFACTOR: [
            "refactor", "restructure", "move", "extract",
            "split", "merge", "clean", "simplify", "optimize",
            "decouple", "reorganize",
        ],
        TaskType.EXPLAIN: [
            "explain", "how does", "why does", "what does",
            "understand", "describe", "show me", "walk through",
            "how to", "what is", "tell me about",
        ],
        TaskType.REVIEW: [
            "review", "check", "audit", "lint", "feedback",
            "improve", "suggest", "vulnerability", "security",
        ],
        TaskType.TEST: [
            "test", "tests", "spec", "coverage", "assert",
            "unit test", "integration test", "write test",
            "add test", "test case", "fixture", "mock",
        ],
        TaskType.DOCUMENT: [
            "docstring", "document", "jsdoc", "readme",
            "api doc", "documentation", "describe", "annotate",
            "type hint", "comment",
        ],
        TaskType.MIGRATE: [
            "migrate", "upgrade", "deprecated", "replace all",
            "migration", "port", "convert", "switch from",
            "move from", "update all", "breaking change",
        ],
        TaskType.ARCHITECTURE: [
            "architecture", "design", "system design",
            "how should", "approach", "strategy", "pattern",
            "component", "service", "module design",
        ],
    }

    for task_type, keywords in _KEYWORD_SIGNALS.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                scores[task_type] += 1.0

    # Signal 2: Entity context
    for entity in entities:
        if entity.entity_type == "error_message":
            scores[TaskType.DEBUG] += 3.0  # Strong debug signal
        if entity.entity_type == "line_number":
            scores[TaskType.DEBUG] += 1.0
            scores[TaskType.EDIT] += 1.0

    # Signal 3: Question marks suggest EXPLAIN
    if "?" in prompt_lower:
        scores[TaskType.EXPLAIN] += 1.5

    # Default: if no strong signal, assume EDIT
    max_score = max(scores.values())
    if max_score == 0:
        return TaskType.EDIT

    # Tie-breaking priority (lower index = higher priority)
    priority = [
        TaskType.DEBUG, TaskType.EDIT, TaskType.CREATE,
        TaskType.REFACTOR, TaskType.EXPLAIN, TaskType.REVIEW,
        TaskType.TEST, TaskType.DOCUMENT, TaskType.MIGRATE,
        TaskType.ARCHITECTURE,
    ]

    # Sort by (score DESC, priority_index ASC)
    best = min(
        (t for t in TaskType if scores[t] == max_score),
        key=lambda t: priority.index(t) if t in priority else 99,
    )
    return best


# ── Two-Tier Stop-Word System (v4) ───────────────────────────────────
#
# HARD stops: language keywords, articles, pronouns — ALWAYS filtered.
#   These are never useful as entity matches regardless of context.
#
# SOFT stops: domain terms that CAN be entity matches.
#   Only filtered when we already have hard entities (file paths, exact FQN matches).
#   When no hard entities are found, soft stops are KEPT as search terms.
#   This fixes the critical v3 bug where "content", "length", "header" were
#   always filtered, causing MISS on NLP prompts like
#   "Content-Length is always being sent on GET requests".

_HARD_STOPS = frozenset({
    # Articles, pronouns, conjunctions
    "the", "this", "that", "with", "from", "into", "when",
    "then", "than", "have", "has", "had", "was", "were",
    "will", "would", "could", "should", "can", "may",
    "not", "all", "any", "each", "every", "some",
    "also", "just", "only", "more", "most", "less",
    "here", "there", "where", "what", "which", "who",
    "does", "did", "done", "been", "being",
    "but", "and", "for", "are", "isn", "don",
    "about", "after", "before", "between", "during",
    # Python/JS keywords (never entity names)
    "def", "class", "import", "return", "yield", "async",
    "await", "try", "except", "finally", "raise", "pass",
    "true", "false", "none", "null", "undefined",
    "const", "let", "var", "function", "interface",
    # Generic verbs that are never entity targets
    "make", "take", "use", "using", "used", "like",
    "need", "want", "know", "see", "look", "give",
    "tell", "keep", "let",
    # Meta-terms about code (not code entities themselves)
    "file", "code", "method", "module", "package",
    "line", "lines", "block", "section",
    # Past tenses of common verbs
    "added", "disabled", "enabled", "automatically",
    "especially", "skipped", "missing", "found", "working",
})

_SOFT_STOPS = frozenset({
    # Domain terms that COULD be entity names in the right project.
    # Only filtered when better entities are already found.
    "get", "set", "add", "run", "put", "pop", "send",
    "read", "post", "head", "delete", "patch",
    "content", "length", "header", "headers",
    "request", "response", "type", "value", "data",
    "name", "path", "body", "url", "item", "items",
    "list", "dict", "string", "number", "result",
    "error", "status", "check", "test", "fix", "find",
    "call", "model", "query", "session", "token",
    "config", "handler", "middleware", "route", "view",
})
