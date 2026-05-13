#!/usr/bin/env python3
"""Test docstring-first summary picker logic."""

from src.skeletongraph.assembly.prompt_builder import _pick_summary

# Simulate skeleton with docstring
class MockSkeleton:
    def __init__(self, fqn, docstring, summary):
        self.fqn = fqn
        self.docstring = docstring

# Simulate store with summaries
class MockStore:
    def __init__(self):
        self.summaries = {}

# Test 1: Docstring exists and should be preferred
print('=== DOCSTRING-FIRST SUMMARY PICKER TEST ===\n')
sk1 = MockSkeleton('func_a', 'Authenticate user against database', None)
store1 = MockStore()
store1.summaries['func_a'] = 'Old LLM summary that is wrong'
result1 = _pick_summary(sk1, store1)
print('Test 1 (Docstring prioritized over summary):')
print(f'  Docstring: "Authenticate user against database"')
print(f'  LLM Summary: "Old LLM summary that is wrong"')
print(f'  Result: "{result1}"')
print(f'  ✅ PASS - Used docstring' if 'Authenticate' in result1 else f'  ❌ FAIL')
print()

# Test 2: No docstring, use summary fallback
sk2 = MockSkeleton('func_b', None, None)
store2 = MockStore()
store2.summaries['func_b'] = 'Fallback summary from LLM'
result2 = _pick_summary(sk2, store2)
print('Test 2 (Summary fallback when no docstring):')
print(f'  Docstring: None')
print(f'  LLM Summary: "Fallback summary from LLM"')
print(f'  Result: "{result2}"')
print(f'  ✅ PASS - Used summary' if 'Fallback' in result2 else f'  ❌ FAIL')
print()

# Test 3: Both missing
sk3 = MockSkeleton('func_c', None, None)
store3 = MockStore()
result3 = _pick_summary(sk3, store3)
print('Test 3 (Empty when both missing):')
print(f'  Docstring: None')
print(f'  LLM Summary: (not in store)')
print(f'  Result: "{result3}"')
print(f'  ✅ PASS - Empty string' if result3 == '' else f'  ❌ FAIL')
