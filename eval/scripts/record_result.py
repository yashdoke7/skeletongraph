#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Record agent evaluation result.")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--task", required=True, help="Task ID")
    parser.add_argument("--mode", default="sg", choices=["baseline", "sg"], help="Result mode")
    parser.add_argument("--success", required=True, type=int, choices=[0, 1], help="1 for success, 0 for failure")
    args = parser.parse_args()

    results_file = Path("eval/results.json")
    
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
    else:
        results = {}

    if args.agent not in results:
        results[args.agent] = {}

    if args.task not in results[args.agent]:
        results[args.agent][args.task] = {}

    results[args.agent][args.task][args.mode] = args.success

    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    status = "PASSED" if args.success == 1 else "FAILED"
    print(f"Recorded: {args.agent} on {args.task} ({args.mode}) -> {status}")

if __name__ == "__main__":
    main()
