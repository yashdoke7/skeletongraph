import json
import sys
from pathlib import Path

# Add eval/scripts to path to import setup_workspaces
sys.path.append(str(Path("eval/scripts").resolve()))
from setup_workspaces import TASKS

with open("eval/tasks.json", "w", encoding="utf-8") as f:
    json.dump(TASKS, f, indent=2)

print("Extracted tasks to eval/tasks.json")
