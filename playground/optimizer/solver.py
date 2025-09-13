import pulp

def solve_conflict_with_ilp(network):
    """Formulates and solves the overtake problem using ILP."""
    # 1. Gather data for the model
    goods = network.trains["Goods-456"]
    express = network.trains["Express-123"]
    station_b = network.stations["B"]
    
    # If a train is disrupted, its time to reach the station is infinite.
    if goods.state == "HALTED":
        time_to_b_goods = float('inf')
    else:
        time_to_b_goods = (station_b.position - goods.position) / goods.max_speed if goods.max_speed > 0 else float('inf')
    
    time_to_b_express = (station_b.position - express.position) / express.max_speed if express.max_speed > 0 else float('inf')
    
    arrival_at_b_goods = network.time + time_to_b_goods
    arrival_at_b_express = network.time + time_to_b_express
    
    time_penalty_loop = 60 
    time_clear_main = 30 
    cost_penalty_loop = 200 # Penalty for wear-and-tear/fuel for using the loop

    print(f"[ILP] Inputs:")
    print(f"[ILP]   Current Time: {network.time:.0f}s")
    print(f"[ILP]   Goods Train: Arrives at B at ~{arrival_at_b_goods:.0f}s. Priority={goods.priority}")
    print(f"[ILP]   Express Train: Arrives at B at ~{arrival_at_b_express:.0f}s. Priority={express.priority}")

    # 2. Define the ILP Problem
    prob = pulp.LpProblem("TrainOvertake", pulp.LpMinimize)

    # 3. Define Decision Variables
    depart_b_goods = pulp.LpVariable("DepartB_Goods", lowBound=arrival_at_b_goods)
    depart_b_express = pulp.LpVariable("DepartB_Express", lowBound=arrival_at_b_express)
    goods_takes_loop = pulp.LpVariable("GoodsTakesLoop", cat='Binary')

    # 4. Define the Objective Function
    # Minimize weighted departure time, plus any fixed costs/penalties
    prob += (depart_b_goods * goods.priority) + \
            (depart_b_express * express.priority) + \
            (goods_takes_loop * cost_penalty_loop), \
            "MinimizeWeightedDepartureTimeWithCosts"

    # 5. Define Constraints
    prob += depart_b_goods >= arrival_at_b_goods + (goods_takes_loop * time_penalty_loop), "GoodsLoopPenalty"
    M = 10000 
    prob += depart_b_express >= depart_b_goods + time_clear_main - (M * goods_takes_loop), "ExpressWaitsForGoodsOnMain"

    # 6. Solve the problem
    print(f"[ILP] Solving...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) 
    print(f"[ILP] Status: {pulp.LpStatus[prob.status]}")

    # 7. Extract the solution
    plan = {}
    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"[ILP] Optimal Solution Found:")
        if pulp.value(goods_takes_loop) == 1:
            print(f"[ILP]   DECISION: Goods train should take the LOOP at Station B.")
            plan["Goods-456"] = {"route": ["A_to_B_Main", "B-Loop", "B_to_C_Main"]}
            plan["Express-123"] = {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]}
            
            wait_until = pulp.value(depart_b_express) + time_clear_main
            plan["Goods-456"]["wait_at_station"] = "B"
            plan["Goods-456"]["wait_until_time"] = wait_until
            print(f"[ILP]   PLAN: Goods will wait at B until t={wait_until:.0f}s.")
        else:
            print(f"[ILP]   DECISION: Goods train should stay on the MAIN line.")
            plan["Goods-456"] = {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]}
            plan["Express-123"] = {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]}

        print(f"[ILP]   Optimal Departure Times from B: Goods={pulp.value(depart_b_goods):.0f}s, Express={pulp.value(depart_b_express):.0f}s")
    else: # Fallback to a default if solver fails
        plan["Goods-456"] = {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]}
        plan["Express-123"] = {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]}

    return plan

def solve_headon_conflict_with_ilp(network, train1_id, train2_id, block_name):
    """Solves a head-on conflict for a single-line block using ILP."""
    prob = pulp.LpProblem("HeadOnConflict", pulp.LpMinimize)
    
    # 1. Gather data
    t1 = network.trains[train1_id]
    t2 = network.trains[train2_id]
    block = network.blocks[block_name]
    
    # Estimate time for each train to reach the start of the block
    # This requires knowing the block's entry points
    block_track = network.tracks[block.track_names[0]]
    block_start_pos = min(block_track.start_pos, block_track.end_pos)
    block_end_pos = max(block_track.start_pos, block_track.end_pos)

    # Note: This is a simplification. A real system would find the precise entry point based on the train's route.
    entry_point_t1 = block_start_pos if abs(t1.position - block_start_pos) < abs(t1.position - block_end_pos) else block_end_pos
    entry_point_t2 = block_start_pos if abs(t2.position - block_start_pos) < abs(t2.position - block_end_pos) else block_end_pos

    time_to_block_t1 = abs(entry_point_t1 - t1.position) / t1.max_speed if t1.max_speed > 0 else float('inf')
    time_to_block_t2 = abs(entry_point_t2 - t2.position) / t2.max_speed if t2.max_speed > 0 else float('inf')
    
    # Time to traverse the block
    block_length = abs(block_end_pos - block_start_pos)
    traverse_time_t1 = block_length / t1.max_speed if t1.max_speed > 0 else float('inf')
    traverse_time_t2 = block_length / t2.max_speed if t2.max_speed > 0 else float('inf')
    
    arrival_at_block_t1 = network.time + time_to_block_t1
    arrival_at_block_t2 = network.time + time_to_block_t2

    print(f"[ILP-HEADON] Solving for {block_name} between {t1.id} & {t2.id}")
    print(f"[ILP-HEADON]   {t1.id} arrives at ~{arrival_at_block_t1:.0f}s")
    print(f"[ILP-HEADON]   {t2.id} arrives at ~{arrival_at_block_t2:.0f}s")

    # 2. Decision Variables
    t1_gets_block_first = pulp.LpVariable(f"{t1.id}_first", cat='Binary')
    entry_time_t1 = pulp.LpVariable(f"Entry_{t1.id}", lowBound=arrival_at_block_t1)
    exit_time_t1 = pulp.LpVariable(f"Exit_{t1.id}")
    entry_time_t2 = pulp.LpVariable(f"Entry_{t2.id}", lowBound=arrival_at_block_t2)
    exit_time_t2 = pulp.LpVariable(f"Exit_{t2.id}")
    M = 10000 # Big M

    # 3. Objective Function
    prob += (exit_time_t1 * t1.priority) + (exit_time_t2 * t2.priority), "MinimizeWeightedExitTime"

    # 4. Constraints
    # Traversal time constraints
    prob += exit_time_t1 >= entry_time_t1 + traverse_time_t1
    prob += exit_time_t2 >= entry_time_t2 + traverse_time_t2

    # Exclusivity Constraints (Big M)
    prob += entry_time_t2 >= exit_time_t1 - M * (1 - t1_gets_block_first), "t2_waits_for_t1"
    prob += entry_time_t1 >= exit_time_t2 - M * t1_gets_block_first, "t1_waits_for_t2"

    # 5. Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 6. Extract Plan
    plan = {}
    if pulp.LpStatus[prob.status] == 'Optimal':
        if pulp.value(t1_gets_block_first) == 1:
            winner, loser = t1, t2
        else:
            winner, loser = t2, t1
            
        print(f"[ILP-HEADON]   DECISION: {winner.id} proceeds, {loser.id} must wait.")
        plan = {
            winner.id: {"proceed": True},
            loser.id: {"wait_for_block": block_name}
        }
    else: # Fallback
        winner = t1 if t1.priority >= t2.priority else t2
        loser = t2 if t1.priority >= t2.priority else t1
        plan = { winner.id: {"proceed": True}, loser.id: {"wait_for_block": block_name} }

    return plan
