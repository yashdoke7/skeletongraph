#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def parse_trace(trace_path: Path):
    if not trace_path.exists():
        return None

    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        total_input = 0
        total_output = 0
        turns = 0

        # Traverse OpenTelemetry format
        for rs in data.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    # We are looking for chat completion spans
                    if span.get("name", "").startswith("chat:"):
                        input_toks = 0
                        output_toks = 0
                        for attr in span.get("attributes", []):
                            key = attr.get("key")
                            val = attr.get("value", {})
                            if key == "gen_ai.usage.input_tokens":
                                input_toks = int(val.get("intValue", val.get("stringValue", 0)))
                            elif key == "gen_ai.usage.output_tokens":
                                output_toks = int(val.get("intValue", val.get("stringValue", 0)))
                        
                        if input_toks > 0 or output_toks > 0:
                            total_input += input_toks
                            total_output += output_toks
                            turns += 1

        return {
            "total_tokens": total_input + total_output,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "turns": turns
        }
    except Exception as e:
        print(f"Error parsing trace {trace_path}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_ide_trace.py <path_to_trace.json>")
        sys.exit(1)
    
    trace_path = Path(sys.argv[1])
    res = parse_trace(trace_path)
    if res:
        print(f"Parsed {trace_path.name}:")
        print(json.dumps(res, indent=2))
    else:
        print("No valid trace data found.")

if __name__ == "__main__":
    main()
