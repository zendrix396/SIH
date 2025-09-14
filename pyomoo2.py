import pandas as pd
import pulp

def solve_complex_schedule(scenario_name, train_data, section_data, lock_departures=False):
    """
    Solves a complex multi-train scheduling problem using PuLP with realistic constraints.
    """
    print(f"\n==========================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"==========================================================")

    # --- 1. PREPARE THE DATA ---
    for tid, train in train_data.items():
        train['travel_time_A_B'] = (section_data['dist_A_B'] / train['speed_kph']) * 60
        train['travel_time_B_C'] = (section_data['dist_B_C'] / train['speed_kph']) * 60
        train['scheduled_arrival_C'] = train['dep_A_mins'] + train['travel_time_A_B'] + train['travel_time_B_C']

    df = pd.DataFrame.from_dict(train_data, orient='index')
    print("Input Train Data:")
    print(df[['priority', 'speed_kph', 'dep_A_mins', 'stop_penalty_mins']])
    print(f"\nDeparture Times Locked: {lock_departures}")
    print("-" * 30)

    # --- 2. CREATE PULP MODEL ---
    prob = pulp.LpProblem("ComplexTrainSchedule_PuLP", pulp.LpMinimize)
    
    TRAINS = list(train_data.keys())
    STATIONS = ['A', 'B', 'C']
    TRAIN_PAIRS = [(t1, t2) for t1 in TRAINS for t2 in TRAINS if t1 < t2]

    # --- 4. DEFINE VARIABLES ---
    arrival_time = pulp.LpVariable.dicts("ArrTime", (TRAINS, STATIONS), lowBound=None) # Allow negative times initially
    departure_time = pulp.LpVariable.dicts("DepTime", (TRAINS, STATIONS), lowBound=None)
    uses_loop = pulp.LpVariable.dicts("UsesLoop", TRAINS, cat='Binary')
    departs_A_first = pulp.LpVariable.dicts("DepartsAFirst", (TRAIN_PAIRS), cat='Binary')
    departs_B_first = pulp.LpVariable.dicts("DepartsBFirst", (TRAIN_PAIRS), cat='Binary') # NEW: Order for B->C segment
    
    # Controller Action Variables
    hold_at_A = pulp.LpVariable.dicts("HoldA", TRAINS, lowBound=0)
    speed_adj_AB = pulp.LpVariable.dicts("SpeedAdjAB", TRAINS, lowBound=-section_data['max_speed_adj_mins'], upBound=section_data['max_speed_adj_mins'])
    speed_adj_BC = pulp.LpVariable.dicts("SpeedAdjBC", TRAINS, lowBound=-section_data['max_speed_adj_mins'], upBound=section_data['max_speed_adj_mins'])

    # --- 5. DEFINE OBJECTIVE FUNCTION ---
    weighted_delay = pulp.lpSum(
        (arrival_time[t]['C'] - train_data[t]['scheduled_arrival_C']) * train_data[t]['priority'] for t in TRAINS
    )
    
    # NEW: Linearize absolute value for penalties
    abs_speed_adj_AB = pulp.LpVariable.dicts("AbsSpeedAdjAB", TRAINS, lowBound=0)
    abs_speed_adj_BC = pulp.LpVariable.dicts("AbsSpeedAdjBC", TRAINS, lowBound=0)
    for t in TRAINS:
        prob += abs_speed_adj_AB[t] >= speed_adj_AB[t], f"Abs_SA_AB_pos_{t}"
        prob += abs_speed_adj_AB[t] >= -speed_adj_AB[t], f"Abs_SA_AB_neg_{t}"
        prob += abs_speed_adj_BC[t] >= speed_adj_BC[t], f"Abs_SA_BC_pos_{t}"
        prob += abs_speed_adj_BC[t] >= -speed_adj_BC[t], f"Abs_SA_BC_neg_{t}"

    # Penalties for controller actions
    hold_cost = pulp.lpSum(hold_at_A[t] * section_data['hold_penalty_per_min'] for t in TRAINS)
    loop_cost = pulp.lpSum(uses_loop[t] * section_data['loop_penalty'] for t in TRAINS)
    # FIX: Make speed adjustment penalty very high to discourage its use
    speed_cost = pulp.lpSum((abs_speed_adj_AB[t] + abs_speed_adj_BC[t]) * section_data['speed_adj_penalty_per_min'] for t in TRAINS)

    prob += weighted_delay + hold_cost + loop_cost + speed_cost, "Minimize_Total_Weighted_Delay_Plus_Costs"

    # --- 6. DEFINE CONSTRAINTS ---
    M = 10000
    HEADWAY = section_data.get('headway_mins', 2)

    for t in TRAINS:
        train = train_data[t]

        if lock_departures:
            prob += departure_time[t]['A'] == train['dep_A_mins'], f"LockDep_{t}"
            prob += hold_at_A[t] == 0, f"NoHoldIfLocked_{t}"
        else:
            prob += departure_time[t]['A'] == train['dep_A_mins'] + hold_at_A[t], f"MinDepWithHold_{t}"

        prob += arrival_time[t]['B'] >= departure_time[t]['A'] + train['travel_time_A_B'] + speed_adj_AB[t], f"RunTime_AB_{t}"
        prob += arrival_time[t]['C'] >= departure_time[t]['B'] + train['travel_time_B_C'] + speed_adj_BC[t], f"RunTime_BC_{t}"
        prob += departure_time[t]['B'] >= arrival_time[t]['B'] + (uses_loop[t] * train['stop_penalty_mins']), f"Dwell_B_{t}"

    prob += pulp.lpSum(uses_loop[t] for t in TRAINS) <= section_data['loop_lines_at_B'], "LoopCapacity_B"

    for t1, t2 in TRAIN_PAIRS:
        # --- FIX: Enforce sequencing on BOTH track segments ---
        # Segment A -> B
        prob += departure_time[t2]['A'] >= departure_time[t1]['A'] - M * (1 - departs_A_first[t1, t2]), f"DepOrder_A_{t1}_{t2}"
        prob += departure_time[t1]['A'] >= departure_time[t2]['A'] - M * departs_A_first[t1, t2], f"DepOrder_A_{t2}_{t1}"
        prob += arrival_time[t2]['B'] >= arrival_time[t1]['B'] + HEADWAY - M * (1 - departs_A_first[t1, t2]), f"Headway_B_{t1}_{t2}"
        prob += arrival_time[t1]['B'] >= arrival_time[t2]['B'] + HEADWAY - M * departs_A_first[t1, t2], f"Headway_B_{t2}_{t1}"

        # Segment B -> C (NEW)
        prob += departure_time[t2]['B'] >= departure_time[t1]['B'] - M * (1 - departs_B_first[t1, t2]), f"DepOrder_B_{t1}_{t2}"
        prob += departure_time[t1]['B'] >= departure_time[t2]['B'] - M * departs_B_first[t1, t2], f"DepOrder_B_{t2}_{t1}"
        prob += arrival_time[t2]['C'] >= arrival_time[t1]['C'] + HEADWAY - M * (1 - departs_B_first[t1, t2]), f"Headway_C_{t1}_{t2}"
        prob += arrival_time[t1]['C'] >= arrival_time[t2]['C'] + HEADWAY - M * departs_B_first[t1, t2], f"Headway_C_{t2}_{t1}"

        # Sequencing at Station B main line
        prob += departure_time[t2]['B'] >= departure_time[t1]['B'] - M * (1 - departs_A_first[t1, t2]) - M * uses_loop[t1] - M * uses_loop[t2], f"Seq_B_Main_{t1}_{t2}"
        prob += departure_time[t1]['B'] >= departure_time[t2]['B'] - M * departs_A_first[t1, t2] - M * uses_loop[t1] - M * uses_loop[t2], f"Seq_B_Main_{t2}_{t1}"


    # --- 7. SOLVE THE MODEL ---
    print("Solving complex schedule with PuLP...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 8. INTERPRET AND DISPLAY THE RESULTS ---
    print("\n--- RESULTS ---")
    print(f"Solver Status: {pulp.LpStatus[prob.status]}")
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"\nTotal Cost (Weighted Delay + Penalties): {pulp.value(prob.objective):.2f}")
        
        print("\n>>>>> OPTIMAL CONTROLLER ACTIONS <<<<<")
        for t in TRAINS:
            loop = "LOOP" if uses_loop[t].value() > 0.5 else "MAIN"
            print(f"  - {t}: Use {loop} line at B | Hold at A: {hold_at_A[t].value():.1f} min | Speed Adj A->B: {speed_adj_AB[t].value():.1f} min | Speed Adj B->C: {speed_adj_BC[t].value():.1f} min")

        print("\n>>>>> GENERATED OPTIMAL SCHEDULE (in minutes from T=0) <<<<<")
        results = []
        for t in TRAINS:
            results.append({
                'Train': t,
                'Priority': train_data[t]['priority'],
                'Dep_A': departure_time[t]['A'].value(),
                'Arr_B': arrival_time[t]['B'].value(),
                'Dep_B': departure_time[t]['B'].value(),
                'Arr_C': arrival_time[t]['C'].value(),
                'Delay': arrival_time[t]['C'].value() - train_data[t]['scheduled_arrival_C']
            })
        
        results_df = pd.DataFrame(results).set_index('Train')[['Priority', 'Dep_A', 'Arr_B', 'Dep_B', 'Arr_C', 'Delay']].round(2)
        print(results_df)
    else:
        print("\nCould not find an optimal solution. The problem might be over-constrained or infeasible.")

if __name__ == "__main__":
    # Define penalties and operational parameters
    section_data = {
        'dist_A_B': 100, 'dist_B_C': 100, 'loop_lines_at_B': 1,
        'headway_mins': 2, 'max_speed_adj_mins': 2, # Small, realistic speed adjustments
        'hold_penalty_per_min': 0.1, # Holding is cheap
        'loop_penalty': 5.0, # Using a loop has a notable cost
        'speed_adj_penalty_per_min': 15.0 # Speed adjustment is VERY expensive
    }

    # Scenario 1: A solvable conflict where holding is the best option
    morning_rush_data = {
        'Goods':       {'priority': 1,  'speed_kph': 70,  'dep_A_mins': -15, 'stop_penalty_mins': 10},
        'Rajdhani':    {'priority': 10, 'speed_kph': 130, 'dep_A_mins': 0,   'stop_penalty_mins': 5},
        'MailExpress': {'priority': 7,  'speed_kph': 110, 'dep_A_mins': 10,  'stop_penalty_mins': 5},
        'Local':       {'priority': 3,  'speed_kph': 90,  'dep_A_mins': 20,  'stop_penalty_mins': 8},
    }
    solve_complex_schedule("Morning Rush (Realistic Penalties)", morning_rush_data, section_data, lock_departures=False)

    # Scenario 2: A complex mid-section conflict that FORCES a loop decision
    mid_section_conflict_data = {
        'Goods':       {'priority': 1,  'speed_kph': 80,  'dep_A_mins': -15, 'stop_penalty_mins': 10},
        'Rajdhani':    {'priority': 10, 'speed_kph': 130, 'dep_A_mins': 0,   'stop_penalty_mins': 5},
        'MailExpress': {'priority': 7,  'speed_kph': 120, 'dep_A_mins': 5,   'stop_penalty_mins': 5},
        'Local':       {'priority': 3,  'speed_kph': 90,  'dep_A_mins': 20,  'stop_penalty_mins': 8},
    }
    solve_complex_schedule("Mid-Section Conflict (Realistic Penalties)", mid_section_conflict_data, section_data, lock_departures=True)