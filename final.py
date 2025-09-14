import pandas as pd
import pulp
import random
import itertools

def generate_bidirectional_network_data(num_main_stations=3, num_loop_stations=2, num_trains=20):
    """Generates data for a network with bidirectional tracks and a loop."""
    print("--- Generating Bidirectional Network and Scenario Data ---")

    stations = [f"Station_{i}" for i in range(num_main_stations)]
    loop_stations = [f"Loop_Station_{i}" for i in range(num_loop_stations)]
    junction_in = "Junction_A"
    junction_out = "Junction_B"

    network = {
        'stations': stations + loop_stations + [junction_in, junction_out],
        'segments': {
            ("Station_0", junction_in): {'length': 80, 'type': 'single'},
            # *** FIX IS HERE: Added the missing segment connecting the two junctions ***
            (junction_in, junction_out): {'length': 10, 'type': 'single'}, 
            (junction_out, "Station_1"): {'length': 90, 'type': 'single'},
            (junction_in, "Loop_Station_0"): {'length': 60, 'type': 'single'},
            ("Loop_Station_0", junction_out): {'length': 70, 'type': 'single'},
        }
    }

    train_data = {}
    possible_routes = [
        ["Station_0", junction_in, junction_out, "Station_1"],
        ["Station_0", junction_in, "Loop_Station_0", junction_out, "Station_1"],
        ["Station_1", junction_out, junction_in, "Station_0"]
    ]

    for i in range(num_trains):
        train_id = f"Train_{i:02d}"
        priority = random.choice([1, 5, 10])
        base_speed = random.randint(80, 120)
        scheduled_departure = i * 25 + random.randint(0, 10)
        route = random.choice(possible_routes)
        
        scheduled_arrival = scheduled_departure
        for j in range(len(route) - 1):
            u, v = route[j], route[j+1]
            segment_key = (u, v) if (u, v) in network['segments'] else (v, u)
            segment = network['segments'][segment_key] # This line will now work
            scheduled_arrival += (segment['length'] / base_speed) * 60
            if j < len(route) - 2:
                scheduled_arrival += 5 

        train_data[train_id] = {
            'priority': priority, 'base_speed_kph': base_speed,
            'departure_time': scheduled_departure, 'origin': route[0],
            'destination': route[-1], 'route': route, 'stop_penalty_mins': 5,
            'scheduled_arrival': scheduled_arrival
        }

    op_params = {'headway_mins': 4}
    print(f"Generated a network with {len(network['stations'])} nodes and {num_trains} trains.\n")
    return train_data, network, op_params

def generate_controller_briefing_bidirectional(train_data, results):
    """Translates the bidirectional solution into a natural language briefing."""
    print("\n==========================================================")
    print("CONTROLLER'S BRIEFING AND OPERATIONAL PLAN")
    print("==========================================================")

    print("\n--- I. Train Performance Summary ---")
    
    sorted_trains = sorted(results.keys(), key=lambda t: train_data[t]['departure_time'])
    
    for train_id in sorted_trains:
        res = results[train_id]
        delay = res['FinalDelay']
        status = "ON TIME"
        if delay > 10: status = f"SIGNIFICANT DELAY ({delay:.0f} mins)"
        elif delay > 1: status = f"MINOR DELAY ({delay:.0f} mins)"
        
        print(f"\n  - {train_id} (P{train_data[train_id]['priority']}): Final Status - {status}.")
        print(f"    - Route: {' -> '.join(train_data[train_id]['route'])}")
        print(f"    - Scheduled: Depart {res['ScheduledDeparture']:.0f}, Arrive {res['ScheduledArrival']:.0f}")
        print(f"    - Actual:    Depart {res['ActualDeparture']:.0f}, Arrive {res['ActualArrival']:.0f}")

        if delay > 1:
            initial_delay = res['ActualDeparture'] - res['ScheduledDeparture']
            if initial_delay > 1:
                print(f"    - Primarily delayed by an initial hold of {initial_delay:.0f} mins at origin to resolve conflicts.")
            else:
                 print(f"    - Incurred delays en-route due to network congestion.")
    
    print("\n--- II. Primary Cause of Delays ---")
    most_delayed_train = max(results.keys(), key=lambda t: results[t]['FinalDelay'])
    delay_amount = results[most_delayed_train]['FinalDelay']

    if delay_amount > 10:
        print(f"  - The most significant delays (e.g., {most_delayed_train} delayed by {delay_amount:.0f} mins) were caused by resolving conflicts on single-track sections with opposing traffic.")
        print("  - The schedule prioritized higher-priority trains, forcing lower-priority services to wait at stations or junctions.")
    else:
        print("  - The network operated with high fluidity. All delays were minor and managed through routine sequencing.")

    print("\n--- End of Briefing ---")

def solve_bidirectional_schedule(scenario_name, train_data, network, op_params):
    """Solves the bidirectional scheduling problem using the HiGHS solver."""
    print(f"\n==========================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"==========================================================")

    prob = pulp.LpProblem("BidirectionalTrainSchedule", pulp.LpMinimize)

    TRAINS, NODES = list(train_data.keys()), network['stations']
    TRAIN_PAIRS = list(itertools.combinations(TRAINS, 2))
    M, HEADWAY = 10000, op_params['headway_mins']

    arrival_time = pulp.LpVariable.dicts("ArrTime", (TRAINS, NODES), lowBound=0)
    departure_time = pulp.LpVariable.dicts("DepTime", (TRAINS, NODES), lowBound=0)
    order_indices = [(p, s) for p in TRAIN_PAIRS for s in network['segments'].keys()]
    train_order = pulp.LpVariable.dicts("TrainOrder", order_indices, cat='Binary')

    prob += pulp.lpSum(
        (arrival_time[t][train_data[t]['destination']] - train_data[t]['scheduled_arrival']) * train_data[t]['priority']
        for t in TRAINS
    ), "Minimize_Weighted_Delay"

    for t in TRAINS:
        train = train_data[t]
        prob += departure_time[t][train['origin']] >= train['departure_time'], f"MinDep_{t}"
        prob += arrival_time[t][train['origin']] == departure_time[t][train['origin']], f"OriginArr_{t}"
        for i in range(len(train['route']) - 1):
            u, v = train['route'][i], train['route'][i+1]
            key = (u, v) if (u, v) in network['segments'] else (v, u)
            travel_time = (network['segments'][key]['length'] / train['base_speed_kph']) * 60
            prob += arrival_time[t][v] >= departure_time[t][u] + travel_time, f"Travel_{t}_{u}_{v}"
            if v != train['destination']:
                prob += departure_time[t][v] >= arrival_time[t][v] + train['stop_penalty_mins'], f"Dwell_{t}_{v}"

    def find_subpath(route, path): return any((route[i], route[i+1]) == path for i in range(len(route) - 1))

    for (t1, t2), seg_key in order_indices:
        u, v = seg_key
        order = train_order[((t1, t2), seg_key)]
        t1_uv, t1_vu = find_subpath(train_data[t1]['route'], (u,v)), find_subpath(train_data[t1]['route'], (v,u))
        t2_uv, t2_vu = find_subpath(train_data[t2]['route'], (u,v)), find_subpath(train_data[t2]['route'], (v,u))

        if (t1_uv or t1_vu) and (t2_uv or t2_vu):
            if t1_uv and t2_uv:
                prob += departure_time[t2][u] >= departure_time[t1][u] + HEADWAY - M*(1-order), f"H_Same_UV_{t1}_{t2}_{u}_{v}"
                prob += departure_time[t1][u] >= departure_time[t2][u] + HEADWAY - M*order, f"H_Same_UV_{t2}_{t1}_{u}_{v}"
            elif t1_vu and t2_vu:
                prob += departure_time[t2][v] >= departure_time[t1][v] + HEADWAY - M*(1-order), f"H_Same_VU_{t1}_{t2}_{v}_{u}"
                prob += departure_time[t1][v] >= departure_time[t2][v] + HEADWAY - M*order, f"H_Same_VU_{t2}_{t1}_{v}_{u}"
            elif t1_uv and t2_vu:
                prob += departure_time[t1][u] >= arrival_time[t2][u] - M*order, f"C_Opp_{t1}_{t2}_{u}_{v}"
                prob += departure_time[t2][v] >= arrival_time[t1][v] - M*(1-order), f"C_Opp_{t2}_{t1}_{u}_{v}"
            elif t1_vu and t2_uv:
                prob += departure_time[t2][u] >= arrival_time[t1][u] - M*order, f"C_Opp_{t2}_{t1}_{v}_{u}"
                prob += departure_time[t1][v] >= arrival_time[t2][v] - M*(1-order), f"C_Opp_{t1}_{t2}_{v}_{u}"
    
    print("Solving bidirectional schedule with PuLP and HiGHS...")
    solver = pulp.HiGHS(timeLimit=120, msg=True) 
    prob.solve(solver)

    print("\n--- Technical Results ---")
    print(f"Solver Status: {pulp.LpStatus[prob.status]}")
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"Total Cost (Weighted Delay): {pulp.value(prob.objective):.2f}")
        
        results_summary = {}
        for t in TRAINS:
            train = train_data[t]
            actual_arrival = arrival_time[t][train['destination']].value()
            results_summary[t] = {
                'FinalDelay': actual_arrival - train['scheduled_arrival'],
                'ScheduledArrival': train['scheduled_arrival'],
                'ActualArrival': actual_arrival,
                'ScheduledDeparture': train['departure_time'],
                'ActualDeparture': departure_time[t][train['origin']].value(),
            }
        
        generate_controller_briefing_bidirectional(train_data, results_summary)
    else:
        print("\nCould not find a feasible or optimal solution.")

if __name__ == "__main__":
    train_data, network, op_params = generate_bidirectional_network_data()
    solve_bidirectional_schedule(
        "Bidirectional and Looping Network Scenario",
        train_data, network, op_params
    )