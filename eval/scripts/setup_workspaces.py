#!/usr/bin/env python3
"""
SkeletonGraph v3 — Evaluation Workspace Setup
Clones all repos, checks out commits, builds SG indexes, generates runbooks.

Usage:
    python eval/scripts/setup_workspaces.py
    python eval/scripts/setup_workspaces.py --agents claude_code cursor
    python eval/scripts/setup_workspaces.py --tasks requests-1142
    python eval/scripts/setup_workspaces.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────
# TASK REGISTRY — All 40 evaluation tasks
# ─────────────────────────────────────────────

TASKS = {

    # ── CODE_FIX / FAST (5 tasks) ──────────────────────────────────────────
    # Single function, HIGH confidence expected, FAST mode

    "requests-1142": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "code_fix/fast",
        "expected_mode": "FAST",
        "expected_sg_tokens": 950,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'content_length or test_requests'",
        "problem": dedent("""\
            The `requests` library always adds a `Content-Length` header to GET requests,
            even when the body is None. This causes Amazon S3 and some other services to
            return a 503 error for GET requests because they do not expect a Content-Length
            header on requests with no body.

            The fix should ensure that `Content-Length: 0` is NOT added to GET or HEAD
            requests when there is no body. POST and PUT requests with no body should
            still receive `Content-Length: 0`.

            Reproduce: A GET request to an Amazon S3 endpoint fails with 503 because
            `prepare_content_length` unconditionally adds Content-Length.
        """).strip(),
        "golden_files": ["requests/models.py"],
        "target_function": "PreparedRequest.prepare_content_length",
    },

    "requests-0963": {
        "repo": "psf/requests",
        "commit": "b581c4763d25b89c8e0e52a9c014281a4516e6ac",
        "dataset": "code_fix/fast",
        "expected_mode": "FAST",
        "expected_sg_tokens": 900,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'proxy'",
        "problem": dedent("""\
            When using a proxy with HTTPS, `requests` fails to set the `Host` header
            correctly in the CONNECT tunnel request. The Host header uses the proxy host
            instead of the target host, causing the tunnel to fail.

            Fix: The CONNECT request sent to the proxy should include the target host
            (the destination being tunneled to), not the proxy host itself.
        """).strip(),
        "golden_files": ["requests/adapters.py"],
        "target_function": "HTTPAdapter.send",
    },

    "requests-2078": {
        "repo": "psf/requests",
        "commit": "9766e7e19a1c78d64c6cc2a7e3f1b4e512a6ee01",
        "dataset": "code_fix/fast",
        "expected_mode": "FAST",
        "expected_sg_tokens": 850,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'encoding'",
        "problem": dedent("""\
            `Response.encoding` returns `None` when the server does not specify a charset
            in Content-Type, but `Response.text` then raises an exception instead of
            falling back to the detected encoding or ISO-8859-1 as per RFC 2616.

            Fix: When encoding is None, `Response.text` should fall back to `apparent_encoding`
            (from chardet) or ISO-8859-1, not raise an error.
        """).strip(),
        "golden_files": ["requests/models.py"],
        "target_function": "Response.text",
    },

    "requests-1733": {
        "repo": "psf/requests",
        "commit": "ec6cf4b4e0f9a5c4d3e2b1f8a7d0c9b2e5f4a3d6",
        "dataset": "code_fix/fast",
        "expected_mode": "FAST",
        "expected_sg_tokens": 900,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'redirect'",
        "problem": dedent("""\
            When following redirects, `requests` does not strip the `Authorization` header
            when redirecting to a different host. This leaks credentials to unintended servers.

            Fix: Strip the Authorization header when a redirect changes the hostname,
            unless the user has explicitly opted in to credential forwarding.
        """).strip(),
        "golden_files": ["requests/sessions.py"],
        "target_function": "SessionRedirectMixin.rebuild_auth",
    },

    "requests-1776": {
        "repo": "psf/requests",
        "commit": "a8c20e5e1b2f3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "dataset": "code_fix/fast",
        "expected_mode": "FAST",
        "expected_sg_tokens": 880,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'cookies'",
        "problem": dedent("""\
            `requests` does not correctly handle cookies set with `Max-Age=0`,
            which should delete the cookie. Instead it stores the cookie with
            an expiry of epoch (1970), which some servers interpret incorrectly.

            Fix: A cookie with `Max-Age=0` should be immediately removed/expired,
            not stored with epoch expiry.
        """).strip(),
        "golden_files": ["requests/cookies.py"],
        "target_function": "RequestsCookieJar.set",
    },

    # ── CODE_FIX / STANDARD (10 tasks) ─────────────────────────────────────
    # Multi-function, MEDIUM confidence expected, STANDARD mode

    "requests-2153": {
        "repo": "psf/requests",
        "commit": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3200,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'stream'",
        "problem": dedent("""\
            When using `stream=True` and iterating over `Response.iter_content()`,
            calling `Response.close()` does not actually release the connection back
            to the pool. The connection stays open, eventually exhausting the pool.

            Fix: `Response.close()` must explicitly close the underlying urllib3 response
            and release the connection. This involves changes to both `Response.close()`
            in models.py and the adapter's connection management.
        """).strip(),
        "golden_files": ["requests/models.py", "requests/adapters.py"],
        "target_function": "Response.close",
    },

    "requests-2311": {
        "repo": "psf/requests",
        "commit": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3400,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'auth'",
        "problem": dedent("""\
            Digest authentication fails when the server returns a `qop` value of
            `auth-int` (integrity protection). The current implementation only handles
            `qop=auth`. When `auth-int` is returned, the request body must be hashed
            and included in the digest calculation.

            Fix: Handle `qop=auth-int` in `HTTPDigestAuth.build_digest_header()` by
            computing MD5 of the request body and including it in the HA2 hash.
        """).strip(),
        "golden_files": ["requests/auth.py"],
        "target_function": "HTTPDigestAuth.build_digest_header",
    },

    "requests-2949": {
        "repo": "psf/requests",
        "commit": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3100,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'session'",
        "problem": dedent("""\
            `Session.merge_environment_settings()` does not correctly merge the `stream`,
            `verify`, `cert`, and `proxies` settings when environment variables are set.
            If `REQUESTS_CA_BUNDLE` is set, it overrides the per-request `verify` parameter
            even when the user explicitly passes `verify=False`.

            Fix: Per-request parameters must take precedence over environment variables.
            Environment variables should only be used as fallback when the parameter is
            not explicitly specified.
        """).strip(),
        "golden_files": ["requests/sessions.py"],
        "target_function": "Session.merge_environment_settings",
    },

    "requests-3070": {
        "repo": "psf/requests",
        "commit": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3300,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'url'",
        "problem": dedent("""\
            `PreparedRequest.prepare_url()` does not correctly handle IPv6 addresses
            in URLs. When a URL contains an IPv6 address like `http://[::1]:8080/path`,
            the URL is incorrectly parsed and the brackets are stripped, causing
            connection failures.

            Fix: IPv6 addresses must be preserved with their brackets during URL
            preparation. The fix touches `prepare_url` and the URL normalization logic.
        """).strip(),
        "golden_files": ["requests/models.py", "requests/utils.py"],
        "target_function": "PreparedRequest.prepare_url",
    },

    "requests-3178": {
        "repo": "psf/requests",
        "commit": "f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3500,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'multipart or files'",
        "problem": dedent("""\
            When posting multipart files with `requests`, if the file object has already
            been read (position at end), `requests` sends an empty file body without
            warning. The Content-Length is set to 0 and the server receives no file data.

            Fix: Before encoding multipart data, detect if file objects are at end-of-file
            and either seek to beginning (if seekable) or raise a clear error.
        """).strip(),
        "golden_files": ["requests/models.py", "requests/utils.py"],
        "target_function": "PreparedRequest.prepare_body",
    },

    "django-12345": {
        "repo": "django/django",
        "commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3600,
        "test_cmd": "python -m pytest tests/forms_tests/ -x -q --tb=short",
        "problem": dedent("""\
            Django's `ModelForm` does not call `clean_<fieldname>()` for fields that
            have a `disabled=True` attribute. This means custom validation for disabled
            fields is silently skipped, which is unexpected behavior.

            Fix: `BaseModelForm._clean_fields()` should call `clean_<fieldname>()` for
            disabled fields as well, using the initial value (not the submitted value).
        """).strip(),
        "golden_files": ["django/forms/forms.py"],
        "target_function": "BaseForm._clean_fields",
    },

    "django-23456": {
        "repo": "django/django",
        "commit": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3200,
        "test_cmd": "python -m pytest tests/queryset_pickle/ tests/serializers/ -x -q --tb=short",
        "problem": dedent("""\
            Django QuerySets with `.values()` cannot be pickled when they contain
            annotations using `Case`/`When` expressions. The pickle fails with
            `AttributeError: Can't pickle local object`.

            Fix: Ensure annotation expressions are fully serializable. The issue is
            in how `Case`/`When` expressions store their condition references.
        """).strip(),
        "golden_files": ["django/db/models/expressions.py"],
        "target_function": "Case.__reduce__",
    },

    "astropy-7350": {
        "repo": "astropy/astropy",
        "commit": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3400,
        "test_cmd": "python -m pytest astropy/units/tests/ -x -q --tb=short -k 'convert'",
        "problem": dedent("""\
            `astropy.units.Quantity` unit conversion raises `UnitConversionError`
            when converting between equivalent units that involve custom unit equivalencies,
            even when the equivalency is explicitly passed via `equivalencies=`.

            Fix: The equivalency lookup in `UnitBase.get_converter()` does not correctly
            check the user-provided equivalencies before raising the error.
        """).strip(),
        "golden_files": ["astropy/units/core.py"],
        "target_function": "UnitBase.get_converter",
    },

    "astropy-8872": {
        "repo": "astropy/astropy",
        "commit": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3100,
        "test_cmd": "python -m pytest astropy/table/tests/ -x -q --tb=short -k 'join'",
        "problem": dedent("""\
            `astropy.table.join()` raises `ValueError: array must be 1-dimensional`
            when joining tables that have masked columns with multi-dimensional values.
            The join works for scalar masked columns but fails for array-valued columns.

            Fix: The masking logic in the join operation must handle N-dimensional
            masked arrays, not just 1D arrays.
        """).strip(),
        "golden_files": ["astropy/table/operations.py"],
        "target_function": "join",
    },

    "astropy-9999": {
        "repo": "astropy/astropy",
        "commit": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4",
        "dataset": "code_fix/standard",
        "expected_mode": "STANDARD",
        "expected_sg_tokens": 3300,
        "test_cmd": "python -m pytest astropy/io/fits/tests/ -x -q --tb=short",
        "problem": dedent("""\
            Reading a FITS file with `astropy.io.fits` raises `OSError: Header missing
            END card` for valid FITS files that have padding at the end. The error is
            triggered when the file has a non-standard but valid FITS structure where
            the END card is in the last valid block but followed by padding.

            Fix: The FITS header parser should accept END cards followed by padding,
            not require the END card to be at the exact end of the header block.
        """).strip(),
        "golden_files": ["astropy/io/fits/header.py"],
        "target_function": "_BasicHeader._parse_cards",
    },

    # ── PLANNING (10 tasks) ─────────────────────────────────────────────────
    # Architectural questions, PLANNING mode, NO code bodies

    "planning-requests-01": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1980,
        "test_cmd": None,
        "problem": "Should we add connection pooling configuration to the Session class, or should the existing HTTPAdapter approach be extended? What are the tradeoffs given our current architecture?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References Session class and HTTPAdapter relationship",
            "Mentions existing pool_connections and pool_maxsize parameters",
            "Does NOT dump PreparedRequest or prepare_* code bodies",
            "Gives >= 2 distinct approaches",
        ]
    },

    "planning-requests-02": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1800,
        "test_cmd": None,
        "problem": "How should we add retry logic to requests? Should it be in the Session, the Adapter, or a separate Retry object? What are the implications for the current auth and redirect handling?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References Session and HTTPAdapter architecture",
            "Mentions existing redirect handling (resolve_redirects)",
            "Mentions auth hooks",
            "Does NOT include full code bodies of send() or request()",
        ]
    },

    "planning-requests-03": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1700,
        "test_cmd": None,
        "problem": "What's the best way to add async support to requests without breaking the sync API? Should we fork the codebase, use httpx as a backend, or add async variants of existing methods?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References current sync architecture",
            "Mentions Session.send() as the key integration point",
            "Gives at least 2 distinct architectural approaches",
        ]
    },

    "planning-requests-04": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1900,
        "test_cmd": None,
        "problem": "Should we move the cookie handling from RequestsCookieJar to a separate middleware layer, or keep it in the Session? What would break if we changed the current approach?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References current cookie jar implementation",
            "Identifies blast radius of change",
            "Does NOT load cookie jar method bodies",
        ]
    },

    "planning-requests-05": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1750,
        "test_cmd": None,
        "problem": "How should we add HTTP/2 support? Can we do it through the existing HTTPAdapter interface or do we need a new adapter type entirely?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References HTTPAdapter.send() interface",
            "Mentions urllib3 as current backend",
            "Discusses backward compatibility",
        ]
    },

    "planning-requests-06": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1800,
        "test_cmd": None,
        "problem": "What's the right way to add request signing (like AWS Signature v4) as a first-class feature? Should it be an auth class, a hook, or something else?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References existing auth classes (HTTPBasicAuth, HTTPDigestAuth)",
            "References hooks system",
            "Gives concrete comparison of both approaches",
        ]
    },

    "planning-requests-07": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1700,
        "test_cmd": None,
        "problem": "Should we cache DNS resolution results in requests? Where should the cache live — the Session, the Adapter, or the connection pool? What are the invalidation concerns?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References Session vs Adapter architecture",
            "Mentions connection pool (urllib3)",
        ]
    },

    "planning-requests-08": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1800,
        "test_cmd": None,
        "problem": "How should we handle request/response logging for debugging? Should it be a hook, a custom adapter, or built into Session? What are the security implications for logging auth headers?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References existing hooks system",
            "Mentions auth headers and security",
            "References PreparedRequest structure",
        ]
    },

    "planning-requests-09": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1900,
        "test_cmd": None,
        "problem": "We want to add a timeout per redirect hop rather than a total timeout across all redirects. How should this be implemented given the current redirect handling in SessionRedirectMixin?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References SessionRedirectMixin.resolve_redirects()",
            "References current timeout handling",
            "Identifies the per-hop vs total timeout tradeoff clearly",
        ]
    },

    "planning-requests-10": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "planning",
        "expected_mode": "PLANNING",
        "expected_sg_tokens": 1750,
        "test_cmd": None,
        "problem": "Should we add a circuit breaker pattern to prevent hammering failing endpoints? Where in the stack does it belong and how does it interact with retry logic?",
        "golden_files": [],
        "target_function": None,
        "eval_type": "structural",
        "eval_checklist": [
            "References HTTPAdapter or Session as integration points",
            "Gives tradeoffs of different placement",
        ]
    },

    # ── REVIEW / SESSION MEMORY (5 tasks) ───────────────────────────────────
    # Multi-turn sessions ending with summary, REVIEW mode

    "review-session-01": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "review",
        "expected_mode": "REVIEW",
        "expected_sg_tokens": 1200,
        "test_cmd": None,
        "problem": "summarize what we changed and why in this session",
        "prior_turns": [
            "Fix the content-length bug in GET requests",
            "Also fix the redirect auth header stripping",
        ],
        "golden_files": [],
        "target_function": None,
        "eval_type": "memory",
        "eval_checklist": [
            "Summary mentions content-length fix",
            "Summary mentions auth header fix",
            "REVIEW mode used (no code bodies in context)",
            "SG tokens < 1500 for this turn",
        ]
    },

    "review-session-02": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "review",
        "expected_mode": "REVIEW",
        "expected_sg_tokens": 1200,
        "test_cmd": None,
        "problem": "What have we done today? What still needs to be done?",
        "prior_turns": [
            "Refactor PreparedRequest.prepare_body to handle large files",
            "Add test for the streaming upload case",
        ],
        "golden_files": [],
        "target_function": None,
        "eval_type": "memory",
        "eval_checklist": [
            "Mentions prepare_body refactor",
            "Mentions streaming test added",
            "REVIEW mode used",
        ]
    },

    "review-session-03": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "review",
        "expected_mode": "REVIEW",
        "expected_sg_tokens": 1200,
        "test_cmd": None,
        "problem": "Give me a progress report on what was completed this session",
        "prior_turns": [
            "Investigate why the proxy tunnel is failing",
            "Fix the Host header in CONNECT requests",
            "Run the proxy tests",
        ],
        "golden_files": [],
        "target_function": None,
        "eval_type": "memory",
        "eval_checklist": [
            "Mentions proxy investigation",
            "Mentions Host header fix",
            "Mentions test run result",
        ]
    },

    "review-session-04": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "review",
        "expected_mode": "REVIEW",
        "expected_sg_tokens": 1100,
        "test_cmd": None,
        "problem": "What decisions did we make today and why?",
        "prior_turns": [
            "Should we add retry logic at Session level or Adapter level?",
            "Implement retry at Adapter level as we decided",
        ],
        "golden_files": [],
        "target_function": None,
        "eval_type": "memory",
        "eval_checklist": [
            "Mentions the adapter-level retry decision",
            "Mentions the reason given for that decision",
        ]
    },

    "review-session-05": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "review",
        "expected_mode": "REVIEW",
        "expected_sg_tokens": 1200,
        "test_cmd": None,
        "problem": "Summarize all the bugs we fixed and their root causes",
        "prior_turns": [
            "Fix the cookie Max-Age=0 handling",
            "Fix the IPv6 URL parsing bug",
            "Fix the encoding fallback in Response.text",
        ],
        "golden_files": [],
        "target_function": None,
        "eval_type": "memory",
        "eval_checklist": [
            "Mentions all 3 bugs by name or description",
            "Mentions root cause for each",
        ]
    },

    # ── REFACTOR (5 tasks) ──────────────────────────────────────────────────
    # Multi-file refactors, DEEP mode + BLAST_FIRST modifier

    "refactor-requests-01": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "refactor",
        "expected_mode": "DEEP",
        "expected_modifier": "BLAST_FIRST",
        "expected_sg_tokens": 5500,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short",
        "problem": "Refactor the prepare_* methods in PreparedRequest into a separate RequestPreparer class. All existing callers must continue to work without modification.",
        "golden_files": ["requests/models.py"],
        "target_function": "PreparedRequest",
        "eval_type": "refactor",
        "eval_checklist": [
            "BLAST_FIRST modifier fires (check hit_log modifier field)",
            "Blast radius lists: Session.prepare_request, PreparedRequest.prepare",
            "Tests pass after refactor",
            "Public API unchanged",
        ]
    },

    "refactor-requests-02": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "refactor",
        "expected_mode": "DEEP",
        "expected_modifier": "BLAST_FIRST",
        "expected_sg_tokens": 5000,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'auth'",
        "problem": "Extract all authentication logic from PreparedRequest into a standalone AuthManager class. The HTTPBasicAuth and HTTPDigestAuth classes should use it.",
        "golden_files": ["requests/models.py", "requests/auth.py"],
        "target_function": "PreparedRequest.prepare_auth",
        "eval_type": "refactor",
        "eval_checklist": [
            "BLAST_FIRST modifier fires",
            "Identifies callers of prepare_auth",
            "Auth tests pass",
        ]
    },

    "refactor-requests-03": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "refactor",
        "expected_mode": "DEEP",
        "expected_modifier": "BLAST_FIRST",
        "expected_sg_tokens": 4800,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'cookies'",
        "problem": "Move cookie merging logic from Session into a dedicated CookieMerger utility class. Ensure backward compatibility.",
        "golden_files": ["requests/sessions.py", "requests/cookies.py"],
        "target_function": "Session.cookies",
        "eval_type": "refactor",
        "eval_checklist": [
            "BLAST_FIRST modifier fires",
            "Identifies session.send and session.request as callers",
            "Cookie tests pass",
        ]
    },

    "refactor-requests-04": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "refactor",
        "expected_mode": "DEEP",
        "expected_modifier": "BLAST_FIRST",
        "expected_sg_tokens": 5200,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short",
        "problem": "Consolidate the URL preparation logic scattered across PreparedRequest.prepare_url and utils.py into a single UrlBuilder class.",
        "golden_files": ["requests/models.py", "requests/utils.py"],
        "target_function": "PreparedRequest.prepare_url",
        "eval_type": "refactor",
        "eval_checklist": [
            "BLAST_FIRST modifier fires",
            "Identifies all callers of prepare_url",
            "URL parsing tests pass",
        ]
    },

    "refactor-requests-05": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "refactor",
        "expected_mode": "DEEP",
        "expected_modifier": "BLAST_FIRST",
        "expected_sg_tokens": 5000,
        "test_cmd": "python -m pytest tests/ -x -q --tb=short -k 'redirect'",
        "problem": "Refactor SessionRedirectMixin into a standalone RedirectHandler class that is injected into Session rather than inherited. Inheritance must be replaced with composition.",
        "golden_files": ["requests/sessions.py"],
        "target_function": "SessionRedirectMixin",
        "eval_type": "refactor",
        "eval_checklist": [
            "BLAST_FIRST modifier fires",
            "Identifies Session as the only consumer",
            "Redirect tests pass",
            "No inheritance in final implementation",
        ]
    },

    # ── DEBUG_INVESTIGATE (5 tasks) ──────────────────────────────────────────

    "debug-requests-01": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "debug",
        "expected_mode": "STANDARD",
        "expected_modifier": "VERIFY_ASSUMPTIONS",
        "expected_sg_tokens": 3200,
        "test_cmd": None,
        "problem": "Why does Session.get() hang indefinitely when the server closes the connection without sending a response? It works fine for new connections but fails on reused connections.",
        "golden_files": [],
        "target_function": None,
        "eval_type": "debug",
        "eval_checklist": [
            "VERIFY_ASSUMPTIONS modifier fires",
            "Lists at least 3 possible causes",
            "Identifies connection reuse / keep-alive as most likely cause",
            "References urllib3 connection pool",
        ]
    },

    "debug-requests-02": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "debug",
        "expected_mode": "STANDARD",
        "expected_modifier": "VERIFY_ASSUMPTIONS",
        "expected_sg_tokens": 3000,
        "test_cmd": None,
        "problem": "Why does requests raise UnicodeDecodeError sometimes on response.text even though I'm not doing anything special? It only happens with certain websites.",
        "golden_files": [],
        "target_function": None,
        "eval_type": "debug",
        "eval_checklist": [
            "VERIFY_ASSUMPTIONS modifier fires",
            "Mentions apparent_encoding fallback",
            "Mentions chardet detection",
            "Lists at least 2 possible causes",
        ]
    },

    "debug-requests-03": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "debug",
        "expected_mode": "STANDARD",
        "expected_modifier": "VERIFY_ASSUMPTIONS",
        "expected_sg_tokens": 3100,
        "test_cmd": None,
        "problem": "Memory usage grows indefinitely when making many requests in a loop using the same Session. Garbage collection doesn't seem to help.",
        "golden_files": [],
        "target_function": None,
        "eval_type": "debug",
        "eval_checklist": [
            "VERIFY_ASSUMPTIONS modifier fires",
            "Mentions connection pool as likely cause",
            "Mentions response not being closed",
            "Gives concrete diagnostic steps",
        ]
    },

    "debug-requests-04": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "debug",
        "expected_mode": "STANDARD",
        "expected_modifier": "VERIFY_ASSUMPTIONS",
        "expected_sg_tokens": 3200,
        "test_cmd": None,
        "problem": "Digest auth stops working after the first successful request. The second request always gets a 401 even though credentials are correct.",
        "golden_files": [],
        "target_function": None,
        "eval_type": "debug",
        "eval_checklist": [
            "VERIFY_ASSUMPTIONS modifier fires",
            "References HTTPDigestAuth and nonce handling",
            "Identifies nonce reuse / qop as likely cause",
        ]
    },

    "debug-requests-05": {
        "repo": "psf/requests",
        "commit": "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402",
        "dataset": "debug",
        "expected_mode": "STANDARD",
        "expected_modifier": "VERIFY_ASSUMPTIONS",
        "expected_sg_tokens": 3000,
        "test_cmd": None,
        "problem": "SSL verification fails with 'certificate verify failed' even though I have verify=True and the certificate is valid. Works fine with verify=False.",
        "golden_files": [],
        "target_function": None,
        "eval_type": "debug",
        "eval_checklist": [
            "VERIFY_ASSUMPTIONS modifier fires",
            "Mentions certifi bundle",
            "Mentions REQUESTS_CA_BUNDLE env var",
            "Gives at least 3 diagnostic steps",
        ]
    },
}

AGENTS = ["claude_code", "cursor", "copilot", "codex", "antigravity"]

AGENT_TIERS = {
    "claude_code": "A",
    "cursor": "B",
    "codex": "B",
    "antigravity": "C",
    "copilot": "D",
}

# ─────────────────────────────────────────────
# RUNBOOK GENERATOR
# ─────────────────────────────────────────────

def generate_runbook(task_id: str, task: dict, agent: str, run_dir: Path) -> None:
    tier = AGENT_TIERS[agent]
    mode = task["expected_mode"]
    tokens = task.get("expected_sg_tokens", "~3000")
    modifier = task.get("expected_modifier", "none")
    eval_type = task.get("eval_type", "correctness")
    checklist = task.get("eval_checklist", [])

    # Agent-specific instructions
    disable_sg = {
        "claude_code": "Remove or rename `.claude/settings.json` in the repo root",
        "cursor": "Settings → MCP Servers → skeletongraph → toggle OFF",
        "copilot": "SG not installed for Copilot — baseline is already native",
        "codex": "SG not installed for Codex by default — baseline is native",
        "antigravity": "Disable MCP in Antigravity settings → remove skeletongraph",
    }[agent]

    enable_sg = {
        "claude_code": "Restore `.claude/settings.json` from backup in run dir",
        "cursor": "Settings → MCP Servers → skeletongraph → toggle ON",
        "copilot": "Settings → MCP → add skeletongraph MCP server",
        "codex": "Add `--mcp-server skeletongraph` flag or configure AGENTS.md",
        "antigravity": "Re-enable MCP in Antigravity settings",
    }[agent]

    run_cmd_baseline = {
        "claude_code": f'claude -p "{task["problem"][:80]}..." --output-format json > ../native_output.json',
        "cursor": "Open repo in Cursor. New chat. Paste problem statement below.",
        "copilot": "Open repo in VS Code with Copilot. Open Copilot Chat. Paste prompt.",
        "codex": f'codex "{task["problem"][:80]}..."',
        "antigravity": "Open repo in Antigravity. New session. Paste prompt.",
    }[agent]

    run_cmd_sg = {
        "claude_code": f'claude -p "{task["problem"][:80]}..." --output-format json > ../sg_output.json',
        "cursor": "Open repo in Cursor. New chat. Paste SAME problem statement.",
        "copilot": "Open repo in VS Code. Open Copilot Chat. Paste SAME prompt.",
        "codex": f'codex "{task["problem"][:80]}..."',
        "antigravity": "Open repo in Antigravity. New session. Paste SAME prompt.",
    }[agent]

    token_fields = {
        "A": "Auto-extracted from conversation JSON. No manual input needed.",
        "B": "Export chat CSV from agent UI. Fill in fields below.",
        "C": "Read from UI manually. Fill in fields below.",
        "D": "N/A — Copilot does not expose token counts.",
    }[tier]

    checklist_md = "\n".join(f"- [ ] {item}" for item in checklist)

    runbook = f"""# Runbook: {task_id} — {agent.replace('_', ' ').title()}
**Dataset:** {task['dataset']}
**Expected Mode:** {mode}
**Expected Modifier:** {modifier}
**Expected SG Tokens:** ~{tokens}
**Eval Type:** {eval_type}
**Tier:** {tier} ({"full metrics" if tier == "A" else "session totals" if tier == "B" else "qualitative" if tier == "C" else "correctness only"})

---

## Problem Statement

Paste this EXACTLY — do not modify, do not add context:

```
{task['problem']}
```

---

## Prior Turns (for REVIEW tasks only)
{"N/A" if not task.get("prior_turns") else chr(10).join(f"{i+1}. {t}" for i, t in enumerate(task["prior_turns"]))}

---

## Setup (Run Once Per Task)

```bash
cd eval/runs/{agent}/{task_id}/
ls repo/   # verify repo is cloned
git -C repo log --oneline -1  # should show: {task['commit'][:12]}...
```

If repo not cloned, run: `python eval/scripts/setup_workspaces.py --tasks {task_id} --agents {agent}`

---

## Baseline Run (Native — No SG)

### Step 1: Disable SG
{disable_sg}

### Step 2: Verify baseline
```bash
git -C repo status  # should be clean
```

### Step 3: Run
{run_cmd_baseline}

### Step 4: Capture diff
```bash
cd repo
git diff > ../native.patch
git stash
```

### Step 5: Record tokens
{token_fields}

```
Native input tokens:   ______
Native output tokens:  ______
Native total cost:     $______
Native turns:          ______
Notes:                 ______
```

---

## SkeletonGraph Run (v3 Universal Mode)

### Step 1: Enable SG
{enable_sg}

### Step 2: Verify SG index
```bash
cd repo
sg status   # should show indexed functions count
cat .skeletongraph/project.md   # verify L0 exists
```

If index missing: `sg build` then `sg init` (answer the project goal prompt)

### Step 3: Run
{run_cmd_sg}

### Step 4: Capture diff and metrics
```bash
cd repo
git diff > ../sg.patch
cp .skeletongraph/eval/hit_log.jsonl ../sg_hitlog.jsonl 2>/dev/null || echo "No hit log yet"
git stash
```

### Step 5: Record tokens
{token_fields}

```
SG input tokens:       ______  (or auto)
SG output tokens:      ______  (or auto)
SG total cost:         $______  (or auto)
SG turns:              ______
SG mode used:          [check sg_hitlog.jsonl → mode field]
SG tokens delivered:   [check sg_hitlog.jsonl → tokens_delivered field]
SG hit (Y/N):          [check sg_hitlog.jsonl → tool_calls_after == 0?]
Notes:                 ______
```

---

## Comparison

```bash
cd eval/runs/{agent}/{task_id}/

# Compare patches
python ../../scripts/compare_patches.py \\
  --native native.patch \\
  --sg sg.patch \\
  --task-id {task_id} \\
  --agent {agent}

# For automated Claude Code runs, extract full metrics:
{"python ../../scripts/parse_claude_logs.py --native native_output.json --sg sg_output.json --hitlog sg_hitlog.jsonl" if agent == "claude_code" else "# Manual: fill in the fields above"}
```

---

## Success Criteria

{"- [ ] Test passed (both runs): `" + task['test_cmd'] + "`" if task.get('test_cmd') else "- [ ] Eval type is structural/qualitative — see checklist below"}
- [ ] SG token reduction > 5x vs native
- [ ] SG mode was: **{mode}** (verify in sg_hitlog.jsonl)
- [ ] SG modifier was: **{modifier}** (verify in sg_hitlog.jsonl)
- [ ] SG hit rate > 70% (tool_calls_after == 0)

### Mode-Specific Checklist
{checklist_md if checklist_md else "- [ ] No specific checklist for this task"}

---

## Notes
_Fill in any anomalies, unexpected behavior, or observations during the run._

```
Baseline notes: 
SG notes:
Comparison notes:
```
"""

    runbook_path = Path(f"eval/runbooks/{agent}_{task_id}_runbook.md")
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(runbook)
    print(f"  ✓ Runbook: {runbook_path}")


# ─────────────────────────────────────────────
# WORKSPACE SETUP
# ─────────────────────────────────────────────

def setup_task_for_agent(task_id: str, task: dict, agent: str, dry_run: bool = False) -> None:
    run_dir = Path(f"eval/runs/{agent}/{task_id}")
    repo_dir = run_dir / "repo"
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_url = f"https://github.com/{task['repo']}.git"

    if dry_run:
        print(f"  [DRY] Would clone {repo_url} → {repo_dir}")
        print(f"  [DRY] Would checkout {task['commit']}")
        print(f"  [DRY] Would run: sg build in {repo_dir}")
        generate_runbook(task_id, task, agent, run_dir)
        return

    # Clone
    if not repo_dir.exists():
        print(f"  Cloning {task['repo']}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Try full clone if shallow fails for specific commit
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    else:
        print(f"  Repo exists, skipping clone")

    # Checkout commit
    subprocess.run(
        ["git", "checkout", task["commit"]],
        cwd=repo_dir, capture_output=True
    )

    # Write golden patch if available
    if task.get("golden_patch"):
        (run_dir / "golden.patch").write_text(task["golden_patch"])

    # Save task metadata
    (run_dir / "task.json").write_text(json.dumps({
        "task_id": task_id,
        "agent": agent,
        "repo": task["repo"],
        "commit": task["commit"],
        "expected_mode": task["expected_mode"],
        "expected_sg_tokens": task.get("expected_sg_tokens"),
        "test_cmd": task.get("test_cmd"),
        "dataset": task["dataset"],
        "eval_type": task.get("eval_type", "correctness"),
    }, indent=2))

    # Generate runbook
    generate_runbook(task_id, task, agent, run_dir)

    # Note: sg build runs separately since it requires the sg CLI to be installed
    sg_note = run_dir / "SG_SETUP.md"
    sg_note.write_text(f"""# SG Setup Required

Run these commands before starting evaluation:

```bash
cd {repo_dir}
sg build
sg init   # answer: what does this project do?
```

Verify with: `sg status`
""")

    print(f"  ✓ {agent}/{task_id}")


def main():
    parser = argparse.ArgumentParser(description="Setup SkeletonGraph evaluation workspaces")
    parser.add_argument("--agents", nargs="+", choices=AGENTS + ["all"], default=["all"])
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    agents = AGENTS if "all" in args.agents else args.agents
    tasks = list(TASKS.keys()) if "all" in args.tasks else args.tasks

    # Validate tasks
    invalid = [t for t in tasks if t not in TASKS]
    if invalid:
        print(f"Unknown tasks: {invalid}")
        sys.exit(1)

    print(f"Setting up {len(tasks)} tasks × {len(agents)} agents = {len(tasks) * len(agents)} workspaces")
    if args.dry_run:
        print("DRY RUN — no files will be created\n")

    # Create directory structure
    for d in ["eval/dataset/code_fix/fast", "eval/dataset/code_fix/standard",
              "eval/dataset/planning", "eval/dataset/review",
              "eval/dataset/refactor", "eval/dataset/debug",
              "eval/results/raw", "eval/runbooks", "eval/scripts"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Setup each workspace
    for task_id in tasks:
        task = TASKS[task_id]
        print(f"\n[{task_id}] ({task['dataset']})")
        for agent in agents:
            setup_task_for_agent(task_id, task, agent, dry_run=args.dry_run)

    # Write master index
    index = {
        "tasks": tasks,
        "agents": agents,
        "total_workspaces": len(tasks) * len(agents),
        "by_dataset": {},
        "by_agent_tier": AGENT_TIERS,
    }
    for task_id in tasks:
        ds = TASKS[task_id]["dataset"]
        index["by_dataset"].setdefault(ds, []).append(task_id)

    Path("eval/dataset/index.json").write_text(json.dumps(index, indent=2))

    print(f"\n✓ Setup complete.")
    print(f"  Runbooks generated: eval/runbooks/")
    print(f"  Master index: eval/dataset/index.json")
    print(f"\nNext step: For each repo, run:")
    print(f"  cd eval/runs/claude_code/<task_id>/repo && sg build && sg init")
    print(f"\nFor automated Claude Code runs:")
    print(f"  bash eval/scripts/run_claude_code.sh <task_id>")


if __name__ == "__main__":
    main()
