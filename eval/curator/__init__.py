"""Learned curator for SkeletonGraph (eval-only).

A small classifier that predicts the retrieval *mode* for a query, replacing the
rule-based router. Used by the `sg-learned` arm. See docs/CURATOR.md for the full
design, training procedure (disjoint data — no leakage), and honest assessment.
"""
