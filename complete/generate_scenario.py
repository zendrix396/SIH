# File: generate_scenario.py

import json
import random
from pathlib import Path

def generate_head_on_conflict_scenario(num_stations=10, num_trains=15):
    """
    Generates a dataset with a GUARANTEED head-on conflict to force a looping decision.
    """
    print(f"--- Generating scenario with a guaranteed head-on conflict ---")
    
    random.seed(42)
    
    stations = [f"Station_{i:02d}" for i in range(num_stations)]
    section_data = {'stations': stations, 'segments': {}}
    current_pos = 0
    single_track_segment_name = None
    # Ensure the network has a clear single-track bottleneck
    for i in range(num_stations - 1):
        segment_name = f"{stations[i]}_to_{stations[i+1]}"
        length = random.randint(100, 150)
        # Force a single track segment in the middle of the network
        track_type = 'single' if 2 < i < 5 else 'double'
        if track_type == 'single' and not single_track_segment_name:
            single_track_segment_name = segment_name
        
        section_data['segments'][segment_name] = {
            'start_pos': current_pos, 'end_pos': current_pos + length,
            'length': length, 'type': track_type
        }
        current_pos += length

    train_data = {}
    
    # --- INJECT THE HEAD-ON CONFLICT PAIR ---
    print(f"Injecting a head-on conflict on the single-track section around {single_track_segment_name}")
    # High-priority "Up" train that will arrive at the bottleneck mid-journey
    train_data["UP_Express"] = {
        'priority': 10, 'base_speed_kph': 130, 'dep_A_mins': 50,
        'stop_penalty_mins': 5, 'direction': 'UP'
    }
    # Low-priority "Down" train scheduled to meet it head-on
    train_data["DOWN_Goods"] = {
        'priority': 1, 'base_speed_kph': 70, 'dep_A_mins': 0,
        'stop_penalty_mins': 10, 'direction': 'DOWN'
    }

    # Generate other "filler" trains that are out of the way of the main conflict
    for i in range(num_trains - 2):
        train_id = f"Filler_Train_{i:02d}"
        priority = random.choice([1, 3, 5])
        base_speed = random.randint(80, 110)
        # Schedule them much later to not interfere
        scheduled_departure = 200 + (i * 25)
        
        train_data[train_id] = {
            'priority': priority, 'base_speed_kph': base_speed,
            'dep_A_mins': scheduled_departure,
            'stop_penalty_mins': random.randint(5, 8),
            'direction': 'UP' # All other filler trains go UP for simplicity
        }
        
    op_params = {
        'loop_lines_at_stations': 1, 'headway_mins': 3,
        'hold_penalty_per_min': 0.1, 'loop_penalty': 5.0,
    }

    print(f"Generated {num_stations} stations and {num_trains} trains.\n")
    return train_data, section_data, op_params

if __name__ == "__main__":
    train_data, section_data, op_params = generate_head_on_conflict_scenario()

    scenario = {
        "scenario_name": "Head-On Conflict Scenario",
        "train_data": train_data,
        "section_data": section_data,
        "op_params": op_params
    }

    output_path = Path("input_scenario.json")
    with open(output_path, 'w') as f:
        json.dump(scenario, f, indent=4)
        
    print(f"Scenario data with guaranteed head-on conflict saved to '{output_path}'")