# File: run_optimizer.py

import json
import pulp
import pandas as pd
from pathlib import Path

def run_optimization(scenario_data):
    """Solves a large-scale, bi-directional scheduling problem."""
    
    train_data = scenario_data["train_data"]
    section_data = scenario_data["section_data"]
    op_params = scenario_data["op_params"]

    print("--- Preparing and running the bi-directional optimization model ---")

    STATIONS = section_data['stations']
    for tid, train in train_data.items():
        # Assign origin and destination based on direction
        origin = STATIONS[0] if train['direction'] == 'UP' else STATIONS[-1]
        destination = STATIONS[-1] if train['direction'] == 'UP' else STATIONS[0]
        train['origin'] = origin
        train['destination'] = destination

        # Calculate scheduled arrival based on path
        train['scheduled_arrival_final'] = train['dep_A_mins']
        path = STATIONS if train['direction'] == 'UP' else list(reversed(STATIONS))
        for i in range(len(path) - 1):
            start_s, end_s = path[i], path[i+1]
            # Segment names are always Station_A_to_Station_B format
            segment_name = f"{min(start_s, end_s)}_to_{max(start_s, end_s)}"
            segment = section_data['segments'][segment_name]
            train['scheduled_arrival_final'] += (segment['length'] / train['base_speed_kph']) * 60

    prob = pulp.LpProblem("BiDirectionalSchedule", pulp.LpMinimize)
    
    TRAINS = list(train_data.keys())
    SEGMENTS = list(section_data['segments'].keys())
    TRAIN_PAIRS = [(t1, t2) for t1 in TRAINS for t2 in TRAINS if t1 < t2]

    arrival_time = pulp.LpVariable.dicts("ArrTime", (TRAINS, STATIONS), lowBound=None)
    departure_time = pulp.LpVariable.dicts("DepTime", (TRAINS, STATIONS), lowBound=None)
    uses_loop = pulp.LpVariable.dicts("UsesLoop", (TRAINS, STATIONS), cat='Binary')
    hold_at_origin = pulp.LpVariable.dicts("HoldAtOrigin", TRAINS, lowBound=0)
    
    # This variable decides which train gets priority on a single-track segment
    gets_block_first = pulp.LpVariable.dicts("GetsBlockFirst", (TRAIN_PAIRS, SEGMENTS), cat='Binary')

    weighted_delay = pulp.lpSum(
        (arrival_time[t][train_data[t]['destination']] - train_data[t]['scheduled_arrival_final']) * train_data[t]['priority'] for t in TRAINS
    )
    hold_cost = pulp.lpSum(hold_at_origin[t] * op_params['hold_penalty_per_min'] for t in TRAINS)
    loop_cost = pulp.lpSum(uses_loop[t][s] * op_params['loop_penalty'] for t in TRAINS for s in STATIONS)
    
    prob += weighted_delay + hold_cost + loop_cost, "Minimize_Total_Cost"

    M = 100000 
    HEADWAY = op_params['headway_mins']

    for t in TRAINS:
        train = train_data[t]
        origin = train['origin']
        path = STATIONS if train['direction'] == 'UP' else list(reversed(STATIONS))
        
        prob += departure_time[t][origin] == train['dep_A_mins'] + hold_at_origin[t], f"InitialDep_{t}"

        for i in range(len(path)):
            station_name = path[i]
            prob += departure_time[t][station_name] >= arrival_time[t][station_name] + (uses_loop[t][station_name] * train['stop_penalty_mins']), f"Dwell_{t}_{station_name}"
            if i < len(path) - 1:
                start_s, end_s = path[i], path[i+1]
                segment_name = f"{min(start_s, end_s)}_to_{max(start_s, end_s)}"
                segment = section_data['segments'][segment_name]
                travel_time = (segment['length'] / train['base_speed_kph']) * 60
                prob += arrival_time[t][end_s] >= departure_time[t][start_s] + travel_time, f"RunTime_{t}_{segment_name}"

    for i in range(len(STATIONS) - 1):
        station_name = STATIONS[i]
        prob += pulp.lpSum(uses_loop[t][station_name] for t in TRAINS) <= op_params['loop_lines_at_stations'], f"LoopCapacity_{station_name}"

    # --- NEW: CONFLICT CONSTRAINTS FOR ALL PAIRS ON SINGLE TRACKS ---
    for segment_name, segment in section_data['segments'].items():
        if segment['type'] == 'single':
            for t1, t2 in TRAIN_PAIRS:
                start_s = segment_name.split('_to_')[0]
                end_s = segment_name.split('_to_')[1]

                dir1 = train_data[t1]['direction']
                dir2 = train_data[t2]['direction']

                # Case 1: Same direction on single track (overtake prevention)
                if dir1 == dir2:
                    dep_s = start_s if dir1 == 'UP' else end_s
                    arr_s = end_s if dir1 == 'UP' else start_s
                    prob += departure_time[t2][dep_s] >= departure_time[t1][dep_s] - M * (1 - gets_block_first[t1,t2][segment_name])
                    prob += departure_time[t1][dep_s] >= departure_time[t2][dep_s] - M * gets_block_first[t1,t2][segment_name]
                    prob += arrival_time[t2][arr_s] >= arrival_time[t1][arr_s] + HEADWAY - M * (1 - gets_block_first[t1,t2][segment_name])
                    prob += arrival_time[t1][arr_s] >= arrival_time[t2][arr_s] + HEADWAY - M * gets_block_first[t1,t2][segment_name]
                
                # Case 2: Opposite direction on single track (head-on prevention)
                else:
                    up_train = t1 if dir1 == 'UP' else t2
                    down_train = t2 if dir1 == 'UP' else t1
                    # One must clear the block before the other enters.
                    # If UP train gets block first: DOWN train must depart its entry station AFTER UP train arrives there.
                    prob += departure_time[down_train][end_s] >= arrival_time[up_train][end_s] - M * (1 - gets_block_first[t1,t2][segment_name])
                    # If DOWN train gets block first: UP train must depart its entry station AFTER DOWN train arrives there.
                    prob += departure_time[up_train][start_s] >= arrival_time[down_train][start_s] - M * gets_block_first[t1,t2][segment_name]

    print("Solving with HiGHS... This may take a couple of minutes.")
    solver = pulp.HiGHS(timeLimit=120, msg=False)
    prob.solve(solver)

    results = {
        "solver_status": pulp.LpStatus[prob.status],
        "total_cost_objective": pulp.value(prob.objective),
        "train_results": {}
    }
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"Solver found a valid solution with status: {pulp.LpStatus[prob.status]}")
        for t in TRAINS:
            res = {
                "priority": train_data[t]['priority'], "direction": train_data[t]['direction'],
                "scheduled_departure": train_data[t]['dep_A_mins'],
                "scheduled_final_arrival": train_data[t]['scheduled_arrival_final'],
                "optimal_departure": departure_time[t][train_data[t]['origin']].value(),
                "optimal_final_arrival": arrival_time[t][train_data[t]['destination']].value(),
                "final_delay_mins": arrival_time[t][train_data[t]['destination']].value() - train_data[t]['scheduled_arrival_final'],
                "actions": {
                    "hold_at_origin_mins": hold_at_origin[t].value(),
                    "loops_used_at": {s: True for s in STATIONS if uses_loop[t][s].value() > 0.5}
                },
                "full_schedule_mins": {
                    "arrivals": {s: arrival_time[t][s].value() for s in STATIONS},
                    "departures": {s: departure_time[t][s].value() for s in STATIONS}
                }
            }
            results["train_results"][t] = res
    else:
        results["remarks"] = ["Solver could not find a feasible or optimal solution."]

    return results

if __name__ == "__main__":
    input_path = Path("input_scenario.json")
    if not input_path.exists():
        print(f"Error: '{input_path}' not found. Please run generate_scenario.py first.")
    else:
        with open(input_path, 'r') as f:
            scenario_data = json.load(f)
        
        optimization_results = run_optimization(scenario_data)

        output_path = Path("optimization_result.json")
        with open(output_path, 'w') as f:
            json.dump(optimization_results, f, indent=4)
        
        print(f"\nOptimization complete. Technical results saved to '{output_path}'")