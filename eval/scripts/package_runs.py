"""Package the study's run records and transcripts for a release.

    python -m eval.scripts.package_runs --out <dir>/skeletongraph-runs.zip

Copies the run tags the published results were computed from (plus the
retrieval-only result files), redacts the local username from file paths, checks
that every JSON and JSONL file still parses after redaction, and writes a zip with
a README and a SHA-256 manifest. The originals under eval/results/ are never
modified.

Redaction replaces the account name in Windows user paths (``C:\\Users\\<name>\\``,
in any escaping) and in Claude Code's project-folder slugs (``C--Users-<name>-``)
with ``user``. Nothing else is changed: agent transcripts still include each run's
tool list and the agent's own messages.

Unzip into eval/results/ to rerun eval.scripts.paper_v2_analysis against the
records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "eval" / "results"
TAGS = ["nemotron_v2", "nemotron_v4", "claude_v7", "claude_v7_rep2",
        "claude_v7_rep2_retry25", "claude_v8", "claude_rebench_v1",
        "claude_rebench_prose_v1", "claude_v7_prose", "codex_v1"]
RETRIEVAL_ONLY = "paper_verified_*_*.json"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt"}

README = """SkeletonGraph study — run records and transcripts
================================================

agent/<tag>/*.json              one record per run: task, arm, CLI version, tokens,
                                turns, tool calls, retrieval hits, patch, verdict
agent/<tag>/_claude_transcripts Claude Code stream-json transcripts
agent/<tag>/_codex_transcripts  Codex CLI JSON event streams
agent/<tag>/_quarantine_*       runs excluded for contamination (kept for audit)
retrieval/                      retrieval-only results (no agent)
MANIFEST.sha256                 checksum of every file

Tags and settings are listed in eval/README.md of the repository. Unzip into
eval/results/ and run `python -m eval.scripts.paper_v2_analysis` to recompute
every number in docs/RESULTS.md.

Local usernames in file paths are replaced with "user"; nothing else is altered.
"""

_USER_PATH = re.compile(r"(Users(?:\\+|/+))(?!user\b)([A-Za-z0-9._-]+)")
_SLUG = re.compile(r"(C--Users-)(?!user-)([A-Za-z0-9._]+)(-)")


def redact(text: str, name: str) -> tuple[str, int]:
    n = 0

    def sub_path(m):
        nonlocal n
        if m.group(2) == name:
            n += 1
            return m.group(1) + "user"
        return m.group(0)

    def sub_slug(m):
        nonlocal n
        if m.group(2) == name:
            n += 1
            return m.group(1) + "user" + m.group(3)
        return m.group(0)

    text = _USER_PATH.sub(sub_path, text)
    text = _SLUG.sub(sub_slug, text)
    return text, n


def check_parses(arc: str, data: str) -> None:
    if arc.endswith(".json"):
        json.loads(data)
    elif arc.endswith(".jsonl"):
        for line in data.splitlines():
            if line.strip():
                json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--user", default=Path.home().name,
                    help="account name to redact from paths (default: this machine's)")
    args = ap.parse_args()

    files: list[tuple[Path, str]] = []
    for tag in TAGS:
        base = ROOT / "agent" / tag
        if not base.is_dir():
            raise SystemExit(f"missing run tag: {base}")
        files += [(f, f"agent/{tag}/{f.relative_to(base).as_posix()}")
                  for f in sorted(base.rglob("*")) if f.is_file()]
    files += [(f, f"retrieval/{f.name}") for f in sorted(ROOT.glob(RETRIEVAL_ONLY))]

    manifest, redactions = [], 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for src, arc in files:
            if src.suffix in TEXT_SUFFIXES:
                text = src.read_text(encoding="utf-8", errors="surrogateescape")
                text, n = redact(text, args.user)
                redactions += n
                check_parses(arc, text)
                data = text.encode("utf-8", errors="surrogateescape")
            else:
                data = src.read_bytes()
            z.writestr(arc, data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")
        z.writestr("README.txt", README)
        z.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    print(f"{len(files)} files, {redactions} path redactions -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
