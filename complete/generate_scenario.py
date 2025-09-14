# File: generate_scenario.py

import json
import random
from pathlib import Path

def generate_complex_scenario(num_stations=10, num_trains=15, seed=42):
    """
    Generates a more complex and realistic railway scenario with a mix of track types and train priorities.
    """
    print(f"--- Generating complex scenario with {num_stations} stations and {num_trains} trains ---")
    random.seed(seed)

    stations = [f"Station_{i:02d}" for i in range(num_stations)]
    section_data = {'stations': stations, 'segments': {}}
    current_pos = 0

    # Create a network with a mix of single and double tracks
    for i in range(num_stations - 1):
        segment_name = f"{stations[i]}_to_{stations[i+1]}"
        length = random.randint(80, 160)
        # Create a bottleneck of single track in the middle
        track_type = 'single' if 3 < i < 6 else 'double'
        
        section_data['segments'][segment_name] = {
            'start_pos': current_pos, 'end_pos': current_pos + length,
            'length': length, 'type': track_type
        }
        current_pos += length

    train_data = {}
    
    # Generate a diverse set of trains
    for i in range(num_trains):
        train_id = f"Train_{i:02d}"
        priority = random.choices([1, 3, 5, 8, 10], weights=[0.2, 0.3, 0.3, 0.1, 0.1], k=1)[0]
        base_speed = random.randint(70, 140)
        scheduled_departure = random.randint(0, 200) + (i * 10)
        direction = random.choice(['UP', 'DOWN'])

        train_data[train_id] = {
            'priority': priority, 
            'base_speed_kph': base_speed,
            'dep_A_mins': scheduled_departure,
            'stop_penalty_mins': random.randint(5, 12),
            'direction': direction
        }
        
    op_params = {
        'loop_lines_at_stations': 2, 'headway_mins': 4,
        'hold_penalty_per_min': 0.15, 'loop_penalty': 7.0,
    }

    print(f"Generated {num_stations} stations and {num_trains} trains with varied characteristics.\n")
    return train_data, section_data, op_params

if __name__ == "__main__":
    train_data, section_data, op_params = generate_complex_scenario()

    scenario = {
        "scenario_name": "Complex Mixed Traffic Scenario",
        "train_data": train_data,
        "section_data": section_data,
        "op_params": op_params
    }

    output_path = Path("input_scenario.json")
    with open(output_path, 'w') as f:
        json.dump(scenario, f, indent=4)
        
    print(f"Complex scenario data saved to '{output_path}'")