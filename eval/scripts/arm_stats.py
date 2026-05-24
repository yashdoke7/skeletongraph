"""Quick arm stats from 7B results."""
import json, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else 'eval/results/agent_7b'
arms = {}
for f in sorted(os.listdir(d)):
    if not f.endswith('.json') or f.startswith('_'): continue
    r = json.loads(open(os.path.join(d, f)).read())
    arm = r.get('arm', '?')
    stopped = r.get('stopped', '')
    if stopped not in ('submit', 'max_turns'): continue  # exclude errors
    if arm not in arms:
        arms[arm] = {'tok': [], 'search': [], 'files': [], 'prec': [], 'turns': [], 'cost': [], 'hit': []}
    arms[arm]['tok'].append(r.get('billed_input', 0))
    arms[arm]['search'].append(r.get('n_search_calls', 0))
    arms[arm]['files'].append(r.get('unique_files_retrieved_total', 0))
    arms[arm]['prec'].append(r.get('retrieval_precision', 0) or 0)
    arms[arm]['turns'].append(r.get('n_turns', 0))
    arms[arm]['cost'].append(r.get('imputed_cost', 0) or 0)
    arms[arm]['hit'].append(1 if r.get('retrieval_hit') else 0)

def avg(xs):
    return round(sum(xs) / len(xs), 2) if xs else 0

print(f"{'Arm':20s} {'n':>3s} {'avgTok':>8s} {'searches':>8s} {'files':>6s} {'prec':>6s} {'hit':>5s} {'turns':>6s} {'cost$':>8s}")
print("-" * 80)
for arm in sorted(arms):
    a = arms[arm]
    n = len(a['tok'])
    print(f"{arm:20s} {n:3d} {avg(a['tok']):8.0f} {avg(a['search']):8.1f} {avg(a['files']):6.1f} {avg(a['prec']):6.3f} {avg(a['hit']):5.2f} {avg(a['turns']):6.1f} {avg(a['cost']):8.4f}")

# SG vs sg-nograph detailed comparison
print("\n--- SG vs sg-nograph detail ---")
for arm in ['sg', 'sg-nograph']:
    a = arms.get(arm, {})
    if not a: continue
    print(f"\n{arm}: tok/search ratio = {avg(a['tok'])/max(avg(a['search']),0.1):.0f} tokens per search call")
    print(f"  files per search = {avg(a['files'])/max(avg(a['search']),0.1):.2f}")
