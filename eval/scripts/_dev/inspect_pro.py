"""Deep inspection: BM25 search results on Pro."""
import json, glob, os

result_dir = r"C:\Users\ASUS\Desktop\CS\Projects\skeletongraph\eval\results\agent\nemotron_pro"

for path in sorted(glob.glob(os.path.join(result_dir, "*__bm25__*.json"))):
    fname = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    repo = d.get("repo", "?")
    stopped = d.get("stopped", "?")
    gold = d.get("gold_files", [])
    hit = d.get("retrieval_hit", False)
    prec = d.get("retrieval_precision", 0)

    print(f"\n{'='*80}")
    print(f"REPO: {repo}")
    print(f"GOLD FILES: {gold}")
    print(f"STOPPED: {stopped} | HIT: {hit} | PREC: {prec}")

    for i, t in enumerate(d.get("turns", [])):
        for tc in t.get("tool_calls", []):
            if tc.get("name") == "search_code" and i < 6:
                result = tc.get("result", "")
                print(f"  T{i} search_code({tc.get('args',{}).get('query','?')[:60]})")
                print(f"     -> {result[:250]}")
