# unpause.py
import json
from pathlib import Path
import sys

def unpause_timelapse():
    state_file = Path("timelapse_state.json")
    
    if not state_file.exists():
        print("Error: Timelapse is not running (state file not found)")
        sys.exit(1)
        
    try:
        with state_file.open('r') as f:
            state = json.load(f)
            
        if not state.get("paused", False):
            print("Timelapse is already running")
            return
            
        state["paused"] = False
        
        with state_file.open('w') as f:
            json.dump(state, f)
            
        print("Timelapse resumed")
        
    except Exception as e:
        print(f"Error: Failed to resume timelapse: {e}")
        sys.exit(1)

if __name__ == "__main__":
    unpause_timelapse()