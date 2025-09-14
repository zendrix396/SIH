import pandas as pd
import pulp
import random

def generate_complex_data(num_stations=3, num_trains=10):
    """Generates a large, complex dataset for the scheduling problem."""
    print("--- Generating Complex Scenario Data ---")
    
    stations = [f"Station_{i:02d}" for i in range(num_stations)]
    section_data = {'stations': stations, 'segments': {}}
    current_pos = 0
    for i in range(num_stations - 1):
        segment_name = f"{stations[i]}_to_{stations[i+1]}"
        length = random.randint(80, 150)
        track_type = 'double' if random.random() < 0.3 else 'single'
        section_data['segments'][segment_name] = {
            'start_pos': current_pos, 'end_pos': current_pos + length,
            'length': length, 'type': track_type
        }
        current_pos += length

    train_data = {}
    for i in range(num_trains):
        train_id = f"Train_{i:02d}"
        priority = random.choice([1, 1, 3, 3, 5, 7, 7, 10])
        base_speed = random.randint(70, 140)
        scheduled_departure = i * 15
        operational_delay = random.randint(0, 20) if random.random() < 0.25 else 0
        
        train_data[train_id] = {
            'priority': priority, 'base_speed_kph': base_speed,
            'dep_A_mins': scheduled_departure + operational_delay,
            'stop_penalty_mins': random.randint(5, 10)
        }
        
    op_params = {
        'loop_lines_at_stations': 1, 'headway_mins': 3,
        'hold_penalty_per_min': 0.1, 'loop_penalty': 5.0,
    }

    print(f"Generated {num_stations} stations and {num_trains} trains.\n")
    return train_data, section_data, op_params

def generate_controller_briefing(train_data, section_data, results, op_params):
    """Translates the numerical solution into a detailed, natural language briefing."""
    
    print("\n==========================================================")
    print("CONTROLLER'S BRIEFING AND OPERATIONAL PLAN")
    print("==========================================================")
    
    # 1. Summarize the key strategic decisions
    print("\n--- I. Key Strategic Interventions ---")
    actions_taken = False
    for train_id, res in results.items():
        if res['HoldAtA'] > 1 or res['LoopsUsed'] > 0:
            actions_taken = True
            details = []
            if res['HoldAtA'] > 1:
                details.append(f"held at origin for {res['HoldAtA']:.0f} mins")
            if res['LoopsUsed'] > 0:
                looped_at = [s for s, looped in res['LoopStations'].items() if looped]
                details.append(f"routed via loop at {', '.join(looped_at)}")
            print(f"  - {train_id} (P{train_data[train_id]['priority']}): " + ", ".join(details) + ".")

    if not actions_taken:
        print("  - No significant holds or loop maneuvers were executed. Conflicts were managed via sequencing.")

    # 2. Detailed report on high-priority trains
    print("\n--- II. High-Priority Train Performance Analysis (Priority >= 7) ---")
    high_priority_trains = sorted(
        [t for t, d in train_data.items() if d['priority'] >= 7],
        key=lambda t: train_data[t]['dep_A_mins']
    )
    for train_id in high_priority_trains:
        res = results[train_id]
        delay = res['FinalDelay']
        status = "ON TIME"
        if delay > 15: status = f"SIGNIFICANT DELAY ({delay:.0f} mins)"
        elif delay > 1: status = f"MINOR DELAY ({delay:.0f} mins)"
        
        print(f"\n  - {train_id} (P{train_data[train_id]['priority']}): Final Status - {status}.")
        
        # Segment-by-segment delay analysis
        cumulative_delay = 0
        initial_delay = res['ActualDepartureA'] - res['ScheduledDepartureA']
        if initial_delay > 1:
            print(f"    - Incurred {initial_delay:.0f} min delay at origin due to a planned hold.")
            cumulative_delay = initial_delay

        for i in range(len(section_data['stations']) - 1):
            start_station = section_data['stations'][i]
            end_station = section_data['stations'][i+1]
            
            run_time = res['Arrivals'][end_station] - res['Departures'][start_station]
            scheduled_run_time = res['ScheduledRunTimes'][i]
            run_delay = run_time - scheduled_run_time
            
            dwell_time = res['Departures'][end_station] - res['Arrivals'][end_station]
            
            if run_delay > op_params['headway_mins']:
                print(f"    - Lost {run_delay:.0f} mins on the segment to {end_station}, likely trailing a slower train.")
            if dwell_time > train_data[train_id]['stop_penalty_mins'] + 1 and res['LoopsUsed'] == 0:
                 print(f"    - Held at {end_station} for an extra {dwell_time:.0f} mins for traffic to pass.")

    # 3. Explain the primary cause of delays
    print("\n--- III. Primary Cause of Delays ---")
    # Find train with the largest delay
    most_delayed_train = max(results.keys(), key=lambda t: results[t]['FinalDelay'])
    delay_amount = results[most_delayed_train]['FinalDelay']

    if delay_amount > 15:
        print(f"  - The most significant delays (e.g., {most_delayed_train} delayed by {delay_amount:.0f} mins) were primarily caused by network congestion on single-track sections.")
        print("  - The optimal schedule minimized the impact on high-priority services by forcing lower-priority trains to absorb these cascading headway delays.")
    else:
        print("  - The network operated with high fluidity. All delays were minor and managed through routine sequencing.")

    print("\n--- End of Briefing ---")


def solve_large_scale_schedule(scenario_name, train_data, section_data, op_params):
    """Solves the large-scale, multi-feature train scheduling problem."""
    print(f"\n==========================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"==========================================================")

    for tid, train in train_data.items():
        train['scheduled_arrival_final'] = train['dep_A_mins']
        for segment in section_data['segments'].values():
            train['scheduled_arrival_final'] += (segment['length'] / train['base_speed_kph']) * 60

    prob = pulp.LpProblem("LargeScaleSchedule", pulp.LpMinimize)
    
    TRAINS = list(train_data.keys())
    STATIONS = section_data['stations']
    SEGMENTS = list(section_data['segments'].keys())
    TRAIN_PAIRS = [(t1, t2) for t1 in TRAINS for t2 in TRAINS if t1 < t2]

    arrival_time = pulp.LpVariable.dicts("ArrTime", (TRAINS, STATIONS), lowBound=None)
    departure_time = pulp.LpVariable.dicts("DepTime", (TRAINS, STATIONS), lowBound=None)
    uses_loop = pulp.LpVariable.dicts("UsesLoop", (TRAINS, STATIONS), cat='Binary')
    departs_segment_first = pulp.LpVariable.dicts("DepartsSegFirst", (TRAIN_PAIRS, SEGMENTS), cat='Binary')
    hold_at_A = pulp.LpVariable.dicts("HoldAtA", TRAINS, lowBound=0)

    weighted_delay = pulp.lpSum(
        (arrival_time[t][STATIONS[-1]] - train_data[t]['scheduled_arrival_final']) * train_data[t]['priority'] for t in TRAINS
    )
    hold_cost = pulp.lpSum(hold_at_A[t] * op_params['hold_penalty_per_min'] for t in TRAINS)
    loop_cost = pulp.lpSum(uses_loop[t][s] * op_params['loop_penalty'] for t in TRAINS for s in STATIONS)
    
    prob += weighted_delay + hold_cost + loop_cost, "Minimize_Total_Cost"

    M = 100000 
    HEADWAY = op_params['headway_mins']

    for t in TRAINS:
        train = train_data[t]
        prob += departure_time[t][STATIONS[0]] == train['dep_A_mins'] + hold_at_A[t], f"InitialDep_{t}"
        for i, station_name in enumerate(STATIONS):
            prob += departure_time[t][station_name] >= arrival_time[t][station_name] + (uses_loop[t][station_name] * train['stop_penalty_mins']), f"Dwell_{t}_{station_name}"
            if i < len(STATIONS) - 1:
                start_station, end_station = STATIONS[i], STATIONS[i+1]
                segment_name = f"{start_station}_to_{end_station}"
                segment = section_data['segments'][segment_name]
                travel_time = (segment['length'] / train['base_speed_kph']) * 60
                prob += arrival_time[t][end_station] >= departure_time[t][start_station] + travel_time, f"RunTime_{t}_{segment_name}"

    for i in range(len(STATIONS) - 1):
        start_station, end_station = STATIONS[i], STATIONS[i+1]
        segment_name = f"{start_station}_to_{end_station}"
        segment = section_data['segments'][segment_name]
        prob += pulp.lpSum(uses_loop[t][start_station] for t in TRAINS) <= op_params['loop_lines_at_stations'], f"LoopCapacity_{start_station}"

        if segment['type'] == 'single':
            for t1, t2 in TRAIN_PAIRS:
                prob += departure_time[t2][start_station] >= departure_time[t1][start_station] - M * (1 - departs_segment_first[t1, t2][segment_name]), f"SegOrderDep_{t1}_{t2}_{segment_name}"
                prob += departure_time[t1][start_station] >= departure_time[t2][start_station] - M * departs_segment_first[t1, t2][segment_name], f"SegOrderDep_{t2}_{t1}_{segment_name}"
                prob += arrival_time[t2][end_station] >= arrival_time[t1][end_station] + HEADWAY - M * (1 - departs_segment_first[t1, t2][segment_name]), f"SegHeadway_{t1}_{t2}_{segment_name}"
                prob += arrival_time[t1][end_station] >= arrival_time[t2][end_station] + HEADWAY - M * departs_segment_first[t1, t2][segment_name], f"SegHeadway_{t2}_{t1}_{segment_name}"

    print("Solving large-scale schedule with PuLP and HiGHS...")
    solver = pulp.HiGHS(timeLimit=120, msg=True) 
    prob.solve(solver)

    print("\n--- Technical Results ---")
    print(f"Solver Status: {pulp.LpStatus[prob.status]}")
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"Total Cost (Weighted Delay + Penalties): {pulp.value(prob.objective):.2f}")
        
        # --- NEW: Process detailed results for the better briefing ---
        results_summary = {}
        for t in TRAINS:
            res = {
                'HoldAtA': hold_at_A[t].value() or 0,
                'LoopsUsed': sum(uses_loop[t][s].value() for s in STATIONS if uses_loop[t][s].value() is not None),
                'LoopStations': {s: (uses_loop[t][s].value() > 0.5) for s in STATIONS},
                'FinalDelay': arrival_time[t][STATIONS[-1]].value() - train_data[t]['scheduled_arrival_final'],
                'ScheduledArrival': train_data[t]['scheduled_arrival_final'],
                'ActualArrival': arrival_time[t][STATIONS[-1]].value(),
                'ScheduledDepartureA': train_data[t]['dep_A_mins'],
                'ActualDepartureA': departure_time[t][STATIONS[0]].value(),
                'Arrivals': {s: arrival_time[t][s].value() for s in STATIONS},
                'Departures': {s: departure_time[t][s].value() for s in STATIONS},
                'ScheduledRunTimes': []
            }
            for i in range(len(STATIONS) - 1):
                start_station, end_station = STATIONS[i], STATIONS[i+1]
                segment = section_data['segments'][f"{start_station}_to_{end_station}"]
                res['ScheduledRunTimes'].append((segment['length'] / train_data[t]['base_speed_kph']) * 60)
            results_summary[t] = res
        
        generate_controller_briefing(train_data, section_data, results_summary, op_params)
        
    else:
        print("\nCould not find a feasible or optimal solution within the time limit.")

if __name__ == "__main__":
    train_data, section_data, op_params = generate_complex_data(num_stations=3, num_trains=10)
    solve_large_scale_schedule(
        "Complex Multi-Train, Multi-Segment Scenario",
        train_data, section_data, op_params
    )