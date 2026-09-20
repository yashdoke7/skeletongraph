import os
import sys
from pathlib import Path

# Important to set BEFORE import if anything else relies on it, 
# but we will manually re-assign mpf.OUT below to change it dynamically.
os.environ["SG_FIG_OUT"] = "docs/paper/figures_v2"

from eval.scripts import make_paper_figures as mpf

ds = "eval/datasets/swebench_100_prose_stripped.jsonl"

def generate_for(tag, nat_arm, sg_arm, out_folder):
    print(f"\n--- Generating figures for {tag} ({nat_arm} vs {sg_arm}) ---")
    out_dir = Path("docs/paper/figures_v2") / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    mpf.OUT = out_dir
    mpf._style()
    
    nat, sg = mpf.paired(tag, nat_arm, sg_arm)
    print(f"Paired tasks: {len(nat)}")
    
    mpf.fig_tail(nat, sg)
    mpf.fig_scatter(nat, sg)
    
    try:
        mpf.fig_retrieval(nat, sg, ds)
    except Exception as e:
        print("Skipping fig_retrieval:", e)
        
    try:
        mpf.fig_tools(nat, sg)
    except Exception as e:
        print("Skipping fig_tools:", e)
        
    try:
        mpf.fig_context_curve(nat, sg)
    except Exception as e:
        print("Skipping fig_context_curve:", e)
        
    try:
        mpf.fig_mechanism(nat, sg)
    except Exception as e:
        print("Skipping fig_mechanism:", e)

generate_for("claude_v7_rep2", "native", "sg-fusion-plain", "claude_v7_rep2")
generate_for("codex_v1", "codex-native", "codex-sg-plain", "codex_v1")
