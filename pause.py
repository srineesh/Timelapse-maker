# pause.py
import json
from pathlib import Path
import sys

def pause_timelapse():
    state_file = Path("timelapse_state.json")
    
    if not state_file.exists():
        print("Error: Timelapse is not running (state file not found)")
        sys.exit(1)
        
    try:
        with state_file.open('r') as f:
            state = json.load(f)
            
        if state.get("paused", False):
            print("Timelapse is already paused")
            return
            
        state["paused"] = True
        
        with state_file.open('w') as f:
            json.dump(state, f)
            
        print("Timelapse paused")
        
    except Exception as e:
        print(f"Error: Failed to pause timelapse: {e}")
        sys.exit(1)

if __name__ == "__main__":
    pause_timelapse()

