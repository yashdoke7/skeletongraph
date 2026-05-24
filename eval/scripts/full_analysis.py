"""Aggregate whatever results exist across all result directories and produce a combined analysis."""
import json, glob, os, sys
from collections import defaultdict

dirs = {
    '7b_v2': 'eval/results/agent_7b_v2',
    'nim70b': 'eval/results/agent',  # NIM results still in default dir
    '7b_orig': 'eval/results/agent_7b',
}

for label, d in dirs.items():
    if not os.path.isdir(d):
        print(f"\n=== {label}: directory not found ({d}) ===")
        continue
    
    files = [f for f in glob.glob(f'{d}/*.json') if not os.path.basename(f).startswith('_') and 'summary' not in f.lower()]
    if not files:
        print(f"\n=== {label}: no result files ===")
        continue
    
    arms = defaultdict(lambda: {
        'n': 0, 'complete': 0, 'error': 0,
        'hits': [], 'prec': [], 'gold_edit': [],
        'turns': [], 'tok': [], 'cost': [],
        'search_calls': [], 'edit_attempts': [],
        'max_turns_count': 0,
    })
    
    task_results = defaultdict(dict)  # task_id -> {arm: result_dict}
    
    for f in sorted(files):
        try:
            r = json.loads(open(f).read())
        except:
            continue
        arm = r.get('arm', '?')
        task = r.get('task_id', '?')
        stopped = r.get('stopped', '?')
        a = arms[arm]
        a['n'] += 1
        
        if stopped in ('submit', 'max_turns'):
            a['complete'] += 1
            a['hits'].append(1 if r.get('retrieval_hit') else 0)
            a['prec'].append(r.get('retrieval_precision', 0) or 0)
            a['gold_edit'].append(1 if r.get('edited_gold_file') else 0)
            a['turns'].append(r.get('n_turns', 0))
            a['tok'].append(r.get('billed_input', 0))
            a['cost'].append(r.get('imputed_cost', 0) or 0)
            a['search_calls'].append(r.get('n_search_calls', 0))
            
            # Count edit attempts from tool_calls
            edits = 0
            for t in r.get('turns', []):
                for tc in t.get('tool_calls', []):
                    if tc.get('name') == 'edit_file':
                        edits += 1
            a['edit_attempts'].append(edits)
            
            if stopped == 'max_turns':
                a['max_turns_count'] += 1
            
            task_results[task][arm] = {
                'hit': r.get('retrieval_hit', False),
                'gold': r.get('edited_gold_file', False),
                'turns': r.get('n_turns', 0),
                'stopped': stopped,
                'prec': r.get('retrieval_precision', 0) or 0,
            }
        else:
            a['error'] += 1
    
    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0
    
    print(f"\n{'='*80}")
    print(f"=== {label}: {d} ({sum(a['n'] for a in arms.values())} files) ===")
    print(f"{'='*80}")
    print(f"{'Arm':15s} {'n':>3s} {'ok':>3s} {'err':>3s} {'hit':>5s} {'prec':>6s} {'gold':>5s} {'turns':>6s} {'tok':>7s} {'edits':>6s} {'max40':>5s}")
    print("-" * 80)
    for arm in sorted(arms):
        a = arms[arm]
        print(f"{arm:15s} {a['n']:3d} {a['complete']:3d} {a['error']:3d} "
              f"{avg(a['hits']):5.2f} {avg(a['prec']):6.3f} {avg(a['gold_edit']):5.2f} "
              f"{avg(a['turns']):6.1f} {avg(a['tok']):7.0f} {avg(a['edit_attempts']):6.1f} "
              f"{a['max_turns_count']:5d}")
    
    # SG wins/losses analysis
    if 'sg' in arms and len(task_results) > 0:
        print(f"\n--- Where SG wins vs loses (tasks with both sg + baselines) ---")
        sg_only_wins = []
        sg_only_loses = []
        for task in sorted(task_results):
            tr = task_results[task]
            if 'sg' not in tr:
                continue
            sg_hit = tr['sg']['hit']
            for other_arm in ['bm25', 'grep', 'hybrid', 'none']:
                if other_arm not in tr:
                    continue
                other_hit = tr[other_arm]['hit']
                if sg_hit and not other_hit:
                    sg_only_wins.append((task, other_arm))
                elif other_hit and not sg_hit:
                    sg_only_loses.append((task, other_arm))
        
        print(f"  SG finds gold but baseline misses ({len(sg_only_wins)} cases):")
        for task, other in sg_only_wins[:10]:
            short = task.split('__')[-1] if '__' in task else task
            print(f"    {short:30s} vs {other}")
        
        print(f"  Baseline finds gold but SG misses ({len(sg_only_loses)} cases):")
        for task, other in sg_only_loses[:10]:
            short = task.split('__')[-1] if '__' in task else task
            print(f"    {short:30s} vs {other}")
        
        # Tasks where NOBODY finds gold
        all_miss = []
        for task in sorted(task_results):
            tr = task_results[task]
            if all(not v['hit'] for v in tr.values()):
                all_miss.append(task)
        print(f"\n  Tasks where NO arm finds gold ({len(all_miss)}):")
        for task in all_miss[:10]:
            short = task.split('__')[-1] if '__' in task else task
            print(f"    {short}")
        
        # Tasks where SG hits AND edits gold
        sg_full_win = [t for t, tr in task_results.items() 
                       if 'sg' in tr and tr['sg']['hit'] and tr['sg']['gold']]
        print(f"\n  Tasks where SG finds AND edits gold ({len(sg_full_win)}):")
        for task in sg_full_win[:10]:
            short = task.split('__')[-1] if '__' in task else task
            print(f"    {short}")

print("\n\nDone.")
