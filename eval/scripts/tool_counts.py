import json, glob, os
d = 'eval/results/agent'
print(f'--- Tool distributions in {d} ---')
for f in sorted(glob.glob(f'{d}/*.json')):
    try:
        r = json.loads(open(f).read())
        arm = r.get('arm','?')
        task = r.get('task_id','?')
        stopped = r.get('stopped','?')
        counts = {}
        for t in r.get('turns',[]):
            for tc in t.get('tool_calls',[]):
                c = tc['name']
                counts[c] = counts.get(c,0) + 1
        print(f"{arm:10s} {task:30s} {stopped:10s} turns={r.get('n_turns',0):2d} gold={r.get('edited_gold_file',False)} tools={counts}")
    except Exception as e:
        print('Error reading', f, e)
