# SkeletonGraph v3 — Architecture Diagram Prompt

Use this prompt with an image generation model (Nano Banana Pro, Midjourney, DALL-E, etc.) to create a professional, visually striking pipeline diagram suitable for LinkedIn, Twitter/X, and GitHub README.

---

## Prompt (for text-to-image model)

```
Create a professional, modern, visually stunning technical architecture diagram for a software system called "SkeletonGraph v3". This should look like it belongs in a top-tier tech company's engineering blog or a Y Combinator demo day slide. Use a dark theme background (#0d1117 or very dark navy) with vibrant accent colors (electric blue #58a6ff, purple #bc8cff, green #3fb950, amber #d29922).

The diagram shows a PIPELINE flowing LEFT to RIGHT with these exact stages, each in a rounded rectangle node with subtle glow effects:

STAGE 1 (Left, input): "User Prompt" — shown as a chat bubble icon with the text: "fix the content-length bug"

STAGE 2: "Intent Extraction" — small node, icon: magnifying glass. Label below: "Regex + AST entity matching, ~2ms"

STAGE 3: "Query Classifier" — medium node, icon: routing/fork symbol. Shows 7 query types branching: CODE_FIX, NEW_FEATURE, REFACTOR, DEBUG, PLANNING, SUMMARY, GENERAL. Label: "Pure Python, <1ms, deterministic"

STAGE 4: "Resolver" — medium node, icon: graph/network. Label: "FQN matching → keyword fallback → semantic search". Shows a small code graph visualization with nodes and edges.

STAGE 5: "Confidence Scorer" — small node showing a gauge/meter icon with the needle pointing to "MEDIUM". Shows 5 factors as tiny pills: entity, coverage, ambiguity, depth, cross_file. Outputs: HIGH/MEDIUM/LOW/MISS

STAGE 6: "Mode Router" — KEY node, larger, with a glowing border. Shows 5 output lanes like highway exits:
  - FAST (~950 tokens) — green, thin lane
  - STANDARD (~3,500 tokens) — blue, medium lane  
  - DEEP (~6,500 tokens) — purple, wide lane
  - PLANNING (~2,000 tokens, NO CODE) — amber lane
  - REVIEW (~1,200 tokens) — teal lane
  Label: "The primary cost lever"

STAGE 7: "Layer Assembly" — shows 5 stacked horizontal bars representing layers:
  - L0: Project DNA (always loaded, small, red/orange)
  - L1: Architecture Map (medium, blue)
  - L2: Domain Context (conditional, purple)
  - L3: Session Memory (tiered: current/recent/log, green)
  - L4: Target Code (dynamic, largest bar, white)
  Label: "Attention-optimal ordering: Lost in the Middle"

STAGE 8 (Right, output): Shows the assembled context being delivered to 6 different IDE/agent icons arranged in a fan:
  - Claude Code (hook icon + shadow files)
  - Cursor (MCP icon)
  - Copilot (MCP icon)  
  - Codex (MCP icon)
  - Antigravity (MCP icon)
  - Claude.ai / ChatGPT (clipboard icon, labeled "sg prompt")

At the BOTTOM of the diagram, show a horizontal comparison bar:
  LEFT side: "Native Agent: ~45,000 tokens" with a very long red bar
  RIGHT side: "SkeletonGraph v3: ~3,500 tokens" with a very short green bar
  Label in the middle: "12.8x reduction"

STYLE REQUIREMENTS:
- Dark background with subtle grid pattern
- Nodes have subtle glassmorphism (frosted glass effect with blur)
- Connecting arrows are gradient-colored flowing lines (not plain straight arrows)
- Each stage number shown as a small circle badge
- Typography: clean sans-serif (Inter or similar), white text, subtle secondary text in gray
- Overall feel: premium, technical, polished — like a Vercel or Linear engineering diagram
- NO clipart, NO cartoon style — professional technical illustration only
- Include a subtle "SkeletonGraph" wordmark/logo in the top-left corner
- Bottom right: "v3 Pipeline — Universal Mode" in small text

Aspect ratio: 16:9 landscape for social media posting
Resolution: high (suitable for 4K display)
```

---

## Alternative: Simplified Version (for Twitter/X where detail gets lost)

```
Create a sleek, minimal, dark-themed (background #0d1117) technical diagram showing a software pipeline called "SkeletonGraph v3".

Show 4 main stages flowing left to right with glowing neon connections:

1. "Prompt" (chat bubble) →
2. "Classify" (shows: CODE_FIX, PLANNING, REFACTOR as labels) →  
3. "Route" (shows 5 colored lanes: FAST/STANDARD/DEEP/PLANNING/REVIEW with token counts) →
4. "Deliver" (fan out to 6 agent icons: Claude, Cursor, Copilot, Codex, Antigravity, ChatGPT)

Below the pipeline, a dramatic comparison:
- Red bar stretching full width: "Native: 45,000 tokens"
- Tiny green bar: "SG v3: 3,500 tokens"  
- Big bold text: "12.8x cheaper"

Style: glassmorphism, dark theme, electric blue and purple accents, premium feel like a Vercel product page. Clean Inter font. 16:9 landscape.
```

---

## Alternative: README Badge/Hero Image

```
Create a wide hero banner image for a GitHub README. Dark theme (#0d1117 background).

Center: The text "SkeletonGraph" in a modern, bold, clean font with a subtle gradient (electric blue to purple). Below it: "Intelligent Context Assembly for AI Coding Agents" in smaller gray text.

Below the text, show a horizontal flow diagram with 3 key concepts as frosted glass cards:
- Card 1: "Parse" — icon of code tree, subtitle "AST-aware indexing"
- Card 2: "Route" — icon of branching paths, subtitle "5 context modes"  
- Card 3: "Deliver" — icon of multiple screens, subtitle "6 IDE integrations"

At the very bottom, a subtle stat bar: "12.8x token reduction • 7 query types • 5 context modes • <70ms latency"

Style: premium, minimal, dark. Suitable as the first thing someone sees on a GitHub repo page. No busy details. Clean and confident.
```
