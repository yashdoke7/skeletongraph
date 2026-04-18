"""
Intent analysis: extract entities and classify task type from user prompts.

Multi-signal weighted classification — not just keywords, but entity context,
file mentions, and error patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple


class TaskType(Enum):
    """Classification of user intent."""
    DEBUG = "debug"          # Fix a bug, resolve an error
    CREATE = "create"        # Add new feature, write new code
    EDIT = "edit"            # Modify existing code
    REFACTOR = "refactor"    # Restructure without changing behavior
    EXPLAIN = "explain"      # Understand code, ask a question
    REVIEW = "review"        # Code review, audit, lint


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
    file_pattern = re.compile(r'[\w./\\-]+\.(?:py|js|ts|tsx|jsx|mjs|cjs)')
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

    for match in ident_pattern.finditer(prompt):
        name = match.group(1)
        # Skip common English words
        if name.lower() in _COMMON_WORDS:
            continue

        # Check if it matches a known FQN suffix
        for fqn in known_fqns:
            short = fqn.split("::")[-1] if "::" in fqn else fqn
            if short == name or short.endswith(f".{name}"):
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

    # Tie-breaking priority: DEBUG > EDIT > CREATE > REFACTOR > EXPLAIN > REVIEW
    priority = [TaskType.DEBUG, TaskType.EDIT, TaskType.CREATE,
                TaskType.REFACTOR, TaskType.EXPLAIN, TaskType.REVIEW]

    best = max(scores, key=lambda t: (scores[t], -priority.index(t)))
    return best


# Common words to skip during function name extraction
_COMMON_WORDS = frozenset({
    "the", "this", "that", "with", "from", "into", "when",
    "then", "than", "have", "has", "had", "was", "were",
    "will", "would", "could", "should", "can", "may",
    "not", "all", "any", "each", "every", "some",
    "also", "just", "only", "more", "most", "less",
    "make", "take", "use", "using", "used", "like",
    "need", "want", "know", "see", "look", "find",
    "give", "tell", "call", "try", "keep", "let",
    "file", "code", "function", "class", "method",
    "here", "there", "where", "what", "which", "who",
    "does", "did", "done", "been", "being",
    "but", "and", "for", "are", "isn", "don",
    "about", "after", "before", "between", "during",
})
