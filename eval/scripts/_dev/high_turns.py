import json, glob
print('--- High turn or edit count runs ---')
for f in sorted(glob.glob('eval/results/agent/*.json')):
    try:
        r = json.loads(open(f).read())
        turns = r.get('n_turns',0)
        counts = {}
        for t in r.get('turns',[]):
            for tc in t.get('tool_calls',[]):
                c = tc['name']
                counts[c] = counts.get(c,0) + 1
        
        if turns >= 20 or counts.get('edit_file',0) > 5:
            print(f"{r.get('arm','?'):10s} {r.get('task_id','?'):30s} {r.get('stopped','?'):10s} turns={turns:2d} edits={counts.get('edit_file',0)}")
    except: pass
