"""SkeletonGraph agentic evaluation harness.

A controlled ReAct loop over an OpenAI-compatible model endpoint (vLLM).
One fixed model, one fixed prompt; only the retrieval backend is swapped
between arms. See STAGES.md for the staged run plan.
"""
