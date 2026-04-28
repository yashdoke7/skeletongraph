"""Show the user EXACTLY what data the JSON export contains for each tool call.

This dumps the full breakdown so they can visually verify that the numbers
are real, not assumed.
"""
import json

d = json.loads(open(r'eval_logs\copilot\fastapi\native_export.json', 'r', encoding='utf-8').read())
req = d['requests'][0]
meta = req['result']['metadata']
rounds = meta['toolCallRounds']
tcr = meta['toolCallResults']

def extract_text(node):
    texts = []
    if isinstance(node, dict):
        if 'text' in node and isinstance(node['text'], str):
            texts.append(node['text'])
        if 'value' in node and isinstance(node['value'], str):
            texts.append(node['value'])
        for child in node.get('children', []):
            texts.append(extract_text(child))
        if 'node' in node and isinstance(node['node'], (dict, list)):
            texts.append(extract_text(node['node']))
    elif isinstance(node, list):
        for item in node:
            texts.append(extract_text(item))
    return '\n'.join(t for t in texts if t)

print("=" * 90)
print("FULL TOOL-BY-TOOL BREAKDOWN: What the JSON ACTUALLY contains")
print("=" * 90)

total_retrieval = 0
files_modified = []

for i, rnd in enumerate(rounds):
    for tc in rnd.get('toolCalls', []):
        call_id = tc['id']
        name = tc['name']
        args = json.loads(tc.get('arguments', '{}')) if isinstance(tc.get('arguments'), str) else tc.get('arguments', {})
        
        # Get actual result
        result = tcr.get(call_id, {})
        content_list = result.get('content', [])
        actual_text = ''
        for c in content_list:
            if isinstance(c, dict):
                val = c.get('value')
                if isinstance(val, str):
                    actual_text += val
                elif isinstance(val, (dict, list)):
                    actual_text += extract_text(val)

        chars = len(actual_text)
        lines_in_result = actual_text.count('\n') + 1 if actual_text else 0
        
        print(f"\n{'-'*90}")
        print(f"Round {i:2d} | Tool: {name}")
        print(f"         | Args: {json.dumps(args)[:120]}")
        print(f"         | Result: {chars} chars, {lines_in_result} lines")
        
        # Show first few lines of actual content
        if actual_text:
            preview_lines = actual_text.split('\n')[:5]
            for line in preview_lines:
                print(f"         |   {line[:100]}")
            if lines_in_result > 5:
                print(f"         |   ... ({lines_in_result - 5} more lines)")
        
        # Track
        if name in ('read_file', 'grep_search'):
            total_retrieval += chars
        
        if name == 'apply_patch':
            # Extract file from patch content
            patch_input = args.get('input', '')
            if 'Update File:' in patch_input:
                file_match = patch_input.split('Update File:')[1].strip().split('\n')[0].strip()
                files_modified.append(file_match)
            elif 'filePath' in args:
                files_modified.append(args['filePath'])

print(f"\n{'='*90}")
print(f"TOTALS")
print(f"{'='*90}")
print(f"Total retrieval content: {total_retrieval} chars")
print(f"Files modified from patches: {files_modified}")
print(f"completionTokens (from API): {req['completionTokens']}")

# Also check: what does the 'response' array look like for the thinking steps?
print(f"\n{'='*90}")
print(f"THINKING/REASONING BLOCKS IN RESPONSE ARRAY")
print(f"{'='*90}")
response_parts = req.get('response', [])
for i, part in enumerate(response_parts):
    if not isinstance(part, dict):
        continue
    kind = part.get('kind')
    if kind == 'thinking':
        val = part.get('value', '')
        title = part.get('generatedTitle', '')
        print(f"\n  Thinking #{i}: title='{title}', len={len(val)} chars")
        if val:
            print(f"    First 200 chars: {val[:200]}")
