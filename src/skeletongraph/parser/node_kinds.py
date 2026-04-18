"""
Unified node kind enumeration covering all supported languages.

Every code entity (function, class, struct, trait, etc.) maps to exactly one NodeKind.
This is the language-agnostic intermediate representation that the rest of the pipeline
operates on — parser modules translate language-specific AST nodes into these kinds.
"""

from enum import Enum


class NodeKind(Enum):
    """Classification of code entities across all supported languages.

    Grouped by semantic category. Each kind carries retrieval semantics:
    - CONSTRUCTOR: always included when parent class is in context
    - PROPERTY: included as signature-only (low token cost)
    - TOP_LEVEL: grouped per-file, not per-function
    - LAMBDA: addressed by parent FQN + line number
    """

    # ── Universal ──────────────────────────────────────────────────────────
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"

    # ── OOP ────────────────────────────────────────────────────────────────
    CLASS = "class"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    PROPERTY = "property"
    ABSTRACT_CLASS = "abstract_class"

    # ── Interface / Protocol ───────────────────────────────────────────────
    INTERFACE = "interface"       # Go, TS, Java, C#, Kotlin
    PROTOCOL = "protocol"        # Swift, Python typing.Protocol

    # ── Struct / Value Types ───────────────────────────────────────────────
    STRUCT = "struct"             # Go, Rust, C, C++, Swift
    ENUM = "enum"
    UNION = "union"              # C/C++, Rust tagged unions
    TYPE_ALIAS = "type_alias"    # TS type, Rust type, Go type, C typedef

    # ── Rust-specific ──────────────────────────────────────────────────────
    TRAIT = "trait"
    IMPL_BLOCK = "impl_block"
    MACRO = "macro"

    # ── C/C++-specific ─────────────────────────────────────────────────────
    HEADER_DECL = "header_decl"
    TEMPLATE = "template"
    NAMESPACE = "namespace"

    # ── Module / Package ───────────────────────────────────────────────────
    MODULE = "module"
    PACKAGE = "package"

    # ── Async / Concurrent ─────────────────────────────────────────────────
    ASYNC_FUNCTION = "async_function"
    GENERATOR = "generator"

    # ── Special ────────────────────────────────────────────────────────────
    LAMBDA = "lambda"
    TOP_LEVEL = "top_level"      # Script-level code outside any function
    DECORATOR_DEF = "decorator_def"
    FIXTURE = "fixture"          # pytest fixtures, test setup methods

    # ── Callable check ─────────────────────────────────────────────────────

    @property
    def is_callable(self) -> bool:
        """True if this kind represents something that can be invoked."""
        return self in _CALLABLE_KINDS

    @property
    def is_type_definition(self) -> bool:
        """True if this kind defines a type (class, struct, interface, etc.)."""
        return self in _TYPE_DEF_KINDS

    @property
    def is_container(self) -> bool:
        """True if this kind can contain other skeletons (class, impl, module)."""
        return self in _CONTAINER_KINDS

    @property
    def auto_include_constructor(self) -> bool:
        """True if including this type should also pull in its constructor."""
        return self in _AUTO_CONSTRUCTOR_KINDS


# Pre-computed sets (avoid repeated set creation on every property call)
_CALLABLE_KINDS = frozenset({
    NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CONSTRUCTOR,
    NodeKind.STATIC_METHOD, NodeKind.CLASS_METHOD, NodeKind.PROPERTY,
    NodeKind.ASYNC_FUNCTION, NodeKind.GENERATOR, NodeKind.LAMBDA,
    NodeKind.MACRO, NodeKind.DECORATOR_DEF, NodeKind.FIXTURE,
})

_TYPE_DEF_KINDS = frozenset({
    NodeKind.CLASS, NodeKind.ABSTRACT_CLASS, NodeKind.INTERFACE,
    NodeKind.PROTOCOL, NodeKind.STRUCT, NodeKind.ENUM, NodeKind.UNION,
    NodeKind.TYPE_ALIAS, NodeKind.TRAIT, NodeKind.TEMPLATE,
})

_CONTAINER_KINDS = frozenset({
    NodeKind.CLASS, NodeKind.ABSTRACT_CLASS, NodeKind.INTERFACE,
    NodeKind.PROTOCOL, NodeKind.STRUCT, NodeKind.TRAIT,
    NodeKind.IMPL_BLOCK, NodeKind.MODULE, NodeKind.NAMESPACE,
    NodeKind.PACKAGE,
})

_AUTO_CONSTRUCTOR_KINDS = frozenset({
    NodeKind.CLASS, NodeKind.ABSTRACT_CLASS, NodeKind.STRUCT,
})
