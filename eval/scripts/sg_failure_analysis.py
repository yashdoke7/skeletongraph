"""Deep dive into SG failure cases — what went wrong in search?"""
import json, glob, os
from collections import defaultdict

d = 'eval/results/agent_7b'

# Load all results
results = {}
for f in sorted(glob.glob(f'{d}/*.json')):
    if os.path.basename(f).startswith('_'): continue
    try:
        r = json.loads(open(f).read())
        key = (r.get('task_id'), r.get('arm'))
        results[key] = r
    except: pass

# Focus: tasks where SG misses but a baseline hits
sg_losses = {}
for (task, arm), r in results.items():
    if arm != 'sg': continue
    if r.get('retrieval_hit'): continue  # SG found it, skip
    # Check if any baseline found it
    for other in ['bm25', 'grep', 'hybrid']:
        other_r = results.get((task, other))
        if other_r and other_r.get('retrieval_hit'):
            if task not in sg_losses:
                sg_losses[task] = {'sg': r, 'winners': []}
            sg_losses[task]['winners'].append((other, other_r))

print("=" * 80)
print(f"SG FAILURE ANALYSIS — {len(sg_losses)} tasks where SG misses but baselines hit")
print("=" * 80)

for task in sorted(sg_losses):
    info = sg_losses[task]
    sg_r = info['sg']
    
    # Get gold files from task dataset
    gold = sg_r.get('first_search_hits', [])
    gold_files = []
    # Try to get gold from dataset
    for df in glob.glob('eval/datasets/*.jsonl'):
        for line in open(df):
            try:
                d_item = json.loads(line)
                if d_item['task_id'] == task:
                    gold_files = d_item.get('gold_files', [])
                    break
            except: pass
        if gold_files: break
    
    print(f"\n{'-'*70}")
    print(f"TASK: {task}")
    print(f"  Gold files: {gold_files}")
    print(f"  SG first_search_hits: {sg_r.get('first_search_hits', [])[:5]}")
    print(f"  SG turns: {sg_r.get('n_turns',0)}, stopped: {sg_r.get('stopped','?')}")
    print(f"  SG search_calls: {sg_r.get('n_search_calls',0)}")
    print(f"  SG embeddings_used: {sg_r.get('embeddings_used','?')}")
    
    # Show what baselines returned
    for other_arm, other_r in info['winners']:
        print(f"  {other_arm} first_search_hits: {other_r.get('first_search_hits', [])[:5]}")
        print(f"    {other_arm} turns: {other_r.get('n_turns',0)}, stopped: {other_r.get('stopped','?')}")

# Now show SG wins — where SG finds but baselines miss
print(f"\n\n{'='*80}")
print("SG WIN ANALYSIS — tasks where SG hits but ALL flat baselines miss")
print("=" * 80)

sg_unique_wins = {}
for (task, arm), r in results.items():
    if arm != 'sg': continue
    if not r.get('retrieval_hit'): continue
    # Check if ALL baselines missed
    all_missed = True
    for other in ['bm25', 'grep', 'hybrid']:
        other_r = results.get((task, other))
        if other_r and other_r.get('retrieval_hit'):
            all_missed = False
            break
    if all_missed:
        sg_unique_wins[task] = r

for task in sorted(sg_unique_wins):
    r = sg_unique_wins[task]
    print(f"\n  {task}")
    print(f"    SG hits: {r.get('first_search_hits', [])[:3]}")
    print(f"    Precision: {r.get('retrieval_precision', 0)}")

# Failure mode analysis
print(f"\n\n{'='*80}")
print("MAX_TURNS ANALYSIS — why do runs exhaust 40 turns?")
print("=" * 80)

for arm in ['sg', 'bm25', 'grep', 'hybrid', 'none']:
    max_runs = [(t, r) for (t, a), r in results.items() 
                if a == arm and r.get('stopped') == 'max_turns']
    if not max_runs:
        continue
    print(f"\n{arm}: {len(max_runs)} max_turns runs")
    for task, r in max_runs[:3]:
        counts = {}
        for t in r.get('turns', []):
            for tc in t.get('tool_calls', []):
                c = tc['name']
                counts[c] = counts.get(c, 0) + 1
        print(f"  {task}: tools={counts}")
