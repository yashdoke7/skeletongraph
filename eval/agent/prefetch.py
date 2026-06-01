"""Pre-pull SWE-bench prebuilt instance images for a fixed task set.

PULL-ONLY: each prebuilt instance image on Docker Hub already contains the
base+env+instance layers, so one `docker pull` per task fetches everything the
verifier needs — no local base/env builds (which is what `prepare_images` does,
and what fails on flaky setup_env.sh). Run this during a good-data window; then
`verify --cache-level instance` runs fully offline.

    python -m eval.agent.prefetch \
        --tasks /path/swebench_100.jsonl --workers 8

Names/tags match the harness defaults (namespace=swebench, tags=latest) so the
images land under the exact keys `verify` (run_evaluation) looks up.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _make_test_spec():
    # import path moved across swebench versions
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except Exception:
        from swebench.harness.test_spec import make_test_spec  # type: ignore
    return make_test_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, type=Path,
                    help="dataset jsonl with task_id per line (e.g. swebench_100.jsonl)")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--namespace", default="swebench")
    ap.add_argument("--tag", default="latest", help="instance image tag")
    ap.add_argument("--env-image-tag", default="latest")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ids = [json.loads(l)["task_id"]
           for l in args.tasks.read_text(encoding="utf-8").splitlines() if l.strip()]

    import docker
    from swebench.harness.utils import load_swebench_dataset
    make_test_spec = _make_test_spec()

    dataset = load_swebench_dataset(args.dataset, args.split, ids)
    client = docker.from_env()

    keys = []
    for inst in dataset:
        spec = make_test_spec(inst, namespace=args.namespace,
                              instance_image_tag=args.tag,
                              env_image_tag=args.env_image_tag)
        keys.append(spec.instance_image_key)
    print(f"{len(keys)} instance images to ensure (namespace={args.namespace}, "
          f"tag={args.tag})")

    def pull(key: str):
        try:
            client.images.get(key)
            return (key, "cached")
        except Exception:
            pass
        repo, tag = key.rsplit(":", 1) if ":" in key else (key, "latest")
        client.images.pull(repo, tag=tag)
        return (key, "pulled")

    ok = fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pull, k): k for k in keys}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                kk, st = fut.result()
                ok += 1
                print(f"[{ok+fail}/{len(keys)}] {st}: {kk}")
            except Exception as e:
                fail += 1
                failures.append((k, str(e)[:120]))
                print(f"[{ok+fail}/{len(keys)}] FAIL: {k}: {str(e)[:120]}")

    print(f"\ndone: {ok} ready, {fail} failed")
    if failures:
        print("failed (these will build/pull on demand during verify):")
        for k, e in failures:
            print(f"  - {k}: {e}")


if __name__ == "__main__":
    main()
