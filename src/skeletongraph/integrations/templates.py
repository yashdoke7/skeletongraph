# SkeletonGraph — Agent Integration Templates
#
# Ready-to-copy config files for each supported coding agent.
# Used by `sg init --agent <name>` to bootstrap integration.

CLAUDE_CODE_HOOKS = '''#!/bin/bash
# SkeletonGraph Claude Code integration hooks
# Install: cp this to ~/.claude/hooks/ and enable in Claude Code settings

SG_ROOT="{project_root}"
SG_MCP="python -m skeletongraph.server.mcp --path $SG_ROOT"

# PreToolUse hook — inject context before tool execution
hook_pre_tool_use() {{
    local tool_name="$1"
    local tool_input="$2"
    
    # Only intercept edit/write tools
    case "$tool_name" in
        Edit|Write|MultiEdit|CreateFile)
            # Get relevant context from SG
            local prompt="Editing: $(echo $tool_input | jq -r '.file_path // .path // empty')"
            if [ -n "$prompt" ]; then
                echo "$prompt" | $SG_MCP query_context 2>/dev/null
            fi
            ;;
    esac
}}

# PostToolUse hook — update session after tool execution  
hook_post_tool_use() {{
    local tool_name="$1"
    local tool_result="$2"
    
    case "$tool_name" in
        Edit|Write|MultiEdit|CreateFile)
            # Record the change in session
            echo "session_update" | $SG_MCP 2>/dev/null
            ;;
    esac
}}
'''

CLAUDE_MD = '''# CLAUDE.md — SkeletonGraph v4 Integration

## Context Retrieval
Before any code task, query SkeletonGraph for targeted context:

```
Use the `query_context` MCP tool with the user's request as prompt.
The response includes:
- Attention-optimized context (target code, callers, constraints)
- Confidence level (HIGH/MEDIUM/LOW/MISS)
- Recommended model tier (slm/mlm/llm)
- Cost transparency (SLM tokens used, savings vs raw)
```

## Model Routing
When SG recommends a model tier:
- **slm**: Simple lookup/navigation — current model is fine
- **mlm**: Standard coding task — use Sonnet-level model
- **llm**: Complex architecture/debugging — consider `/model claude-opus-4-7`

## Page Faults
If you need more detail on a specific function, use `sg_expand`:
```
sg_expand(target="file.py::ClassName.method_name", include_neighbors=true)
```

## Rules
1. ALWAYS call query_context before editing code
2. Trust SG's confidence: if MISS, explore freely via file reading
3. Use sg_expand for back-routing instead of reading entire files
4. Do NOT re-read files that SG already provided in context
5. If a provided summary/docstring is inaccurate for a function you modify,
    update its docstring/comment to match actual behavior
6. After finishing, call report_completion with files_modified so SG can
    refresh the index and summaries
'''

CURSOR_RULES = '''# .cursorrules — SkeletonGraph v4 Integration

## Required First Step
Before making any code changes, ALWAYS call the `query_context` MCP tool
with the user's request. This provides:
- Pre-assembled, attention-optimized context
- Only the relevant functions, not entire files
- Session memory (what was already shown/changed)

## Context Quality
- HIGH confidence: trust the context, make changes directly
- MEDIUM confidence: context is good but verify with quick file scan
- LOW/MISS confidence: SG couldn't find relevant code, explore freely

## Model Recommendations
When the response includes `recommended_model`:
- Consider switching to the recommended model in the model selector
- Architecture tasks benefit from stronger models (Opus/GPT-5.5)

## Back-routing
If you need more detail on a function mentioned in context:
- Use `sg_expand` instead of reading the entire file
- This costs zero tokens and returns just the function body

## Comment Hygiene
If a provided summary/docstring is inaccurate for a function you modify,
update its docstring/comment to match actual behavior.
After finishing, call report_completion with files_modified.

## Cost Transparency  
Each response includes cost data:
- `slm_cost_usd`: how much the SLM retrieval cost
- `context_tokens`: tokens in the assembled context
- `saved_vs_raw_tokens`: tokens saved vs reading raw files
'''

CODEX_AGENTS_MD = '''# AGENTS.md — SkeletonGraph v4 Integration for Codex

## Workflow
1. For every user request, first call `query_context` via MCP
2. Use the returned context instead of reading files directly
3. If more detail needed, use `sg_expand` for specific functions
4. Trust confidence levels:
   - HIGH: proceed with changes
   - MEDIUM: verify with targeted file reads
   - LOW/MISS: explore freely

## Model Profiles
SG recommends model tiers per query:
- `slm` tier → use `--profile fast` (mini models)
- `mlm` tier → use default profile
- `llm` tier → use `--profile deep` (strongest model)

## Comment Hygiene
If a provided summary/docstring is inaccurate for a function you modify,
update its docstring/comment to match actual behavior.
After finishing, call report_completion with files_modified.

## Available MCP Tools
- `query_context(prompt)`: Main entry — returns assembled context
- `sg_expand(target, type, include_neighbors)`: Get full function body
- `search_index(query)`: Keyword search across all functions
'''

ANTIGRAVITY_MCP_CONFIG = '''{
  "name": "skeletongraph",
  "description": "Graph-driven context retrieval for coding agents",
  "command": "python",
  "args": ["-m", "skeletongraph.server.mcp", "--path", "{project_root}"],
  "env": {},
  "tools": {
    "query_context": {
      "description": "Get attention-optimized context for a coding task",
      "parameters": {
        "prompt": {"type": "string", "required": true}
      }
    },
    "sg_expand": {
      "description": "Get full body of a function, class, file, or directory",
      "parameters": {
        "target": {"type": "string", "required": true},
        "type": {"type": "string", "enum": ["auto", "function", "class", "file", "range", "directory"]},
        "include_neighbors": {"type": "boolean", "default": false}
      }
    }
  }
}
'''

COPILOT_EXTENSION_SNIPPET = '''// VS Code extension snippet for GitHub Copilot integration
// Place in your VS Code extension's activate() function

const { exec } = require('child_process');

function getSkeletonGraphContext(prompt, projectRoot) {
    return new Promise((resolve, reject) => {
        const cmd = `python -m skeletongraph.server.mcp --path "${projectRoot}"`;
        const input = JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params: { name: "query_context", arguments: { prompt } }
        });
        
        const proc = exec(cmd, { timeout: 5000 }, (err, stdout) => {
            if (err) return reject(err);
            try {
                const result = JSON.parse(stdout);
                resolve(result?.result?.content?.[0]?.text || "");
            } catch (e) {
                reject(e);
            }
        });
        proc.stdin.write(input + "\\n");
        proc.stdin.end();
    });
}

// Prepend SG context to Copilot prompt
vscode.chat.registerChatParticipant('sg', {
    async provideResponse(request, context, response, token) {
        const sgContext = await getSkeletonGraphContext(
            request.prompt,
            vscode.workspace.rootPath
        );
        response.markdown(sgContext);
    }
});
'''


def get_template(agent: str, project_root: str = ".") -> str:
    """Get the integration template for a specific agent.
    
    Args:
        agent: One of "claude_code", "cursor", "codex", "copilot", "antigravity"
        project_root: Project root path for template substitution
    
    Returns:
        Template string ready to write to disk.
    """
    templates = {
        "claude_code": CLAUDE_MD,
        "cursor": CURSOR_RULES,
        "codex": CODEX_AGENTS_MD,
        "copilot": COPILOT_EXTENSION_SNIPPET,
        "antigravity": ANTIGRAVITY_MCP_CONFIG,
    }
    
    template = templates.get(agent, "")
    return template.replace("{project_root}", project_root)


def get_hook_script(project_root: str = ".") -> str:
    """Get the Claude Code hook script."""
    return CLAUDE_CODE_HOOKS.replace("{project_root}", project_root)
