import pulp

def solve_train_overtake(scenario_name, goods_priority, express_priority):
    """
    This function uses Integer Linear Programming to solve a simple train overtake problem.
    """
    print(f"\n==========================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"==========================================================")
    
    # --- 1. DEFINE THE WORLD (Our Inputs) ---
    
    # Time in minutes
    travel_time_goods = 60      # Normal time for the goods train to finish
    travel_time_express = 30    # Normal time for the express train to finish
    loop_penalty_goods = 15     # Extra time the goods train needs if it uses the loop
    
    print(f"Inputs:")
    print(f"  - Goods Train: Priority={goods_priority}, Base Travel Time={travel_time_goods} mins")
    print(f"  - Express Train: Priority={express_priority}, Base Travel Time={travel_time_express} mins")
    print(f"  - Loop Penalty for Goods: {loop_penalty_goods} mins")
    print("-" * 30)

    # --- 2. CREATE THE OPTIMIZATION PROBLEM ---
    
    # We want to MINIMIZE the total weighted delay.
    prob = pulp.LpProblem("TrainOvertakeDecision", pulp.LpMinimize)

    # --- 3. DEFINE THE DECISION VARIABLES ---
    # These are the "knobs" the optimizer can turn to find the best solution.

    # The CORE decision: Does the goods train use the loop? 1 for YES, 0 for NO.
    goods_takes_loop = pulp.LpVariable("Goods_Takes_Loop", cat='Binary')

    # The outcomes we want to determine: the final finish time for each train.
    finish_time_goods = pulp.LpVariable("Finish_Time_Goods", lowBound=0)
    finish_time_express = pulp.LpVariable("Finish_Time_Express", lowBound=0)

    # --- 4. DEFINE THE OBJECTIVE FUNCTION ---
    # This is WHAT we are trying to minimize.
    # We want to minimize the total "priority-weighted" finish time.
    # This makes delays on high-priority trains "cost" more than delays on low-priority ones.
    
    objective = (finish_time_goods * goods_priority) + (finish_time_express * express_priority)
    prob += objective, "Minimize_Total_Weighted_Finish_Time"

    # --- 5. DEFINE THE CONSTRAINTS ---
    # These are the RULES of our world that the solution MUST obey.

    # RULE 1: If the goods train takes the loop, its finish time is its normal travel time plus the penalty.
    # If it DOESN'T take the loop, its finish time is just its normal travel time.
    prob += finish_time_goods >= travel_time_goods + (goods_takes_loop * loop_penalty_goods), "Goods_Finish_Time_Rule"

    # RULE 2: The Express train's finish time depends on the Goods train's decision.
    # If the goods train stays on the main line, the express is blocked and must wait.
    # This is the "magic" constraint, using the "Big M" method.
    M = 10000 # A very large number

    # This reads: "The express finish time must be AT LEAST the goods finish time,
    # UNLESS the goods train takes the loop (goods_takes_loop=1), which disables this rule."
    prob += finish_time_express >= finish_time_goods - (M * goods_takes_loop), "Express_Blocked_By_Goods_Rule"
    
    # RULE 3: The Express train can't finish faster than its own travel time.
    prob += finish_time_express >= travel_time_express, "Express_Minimum_Time_Rule"
    
    # --- 6. SOLVE THE PROBLEM ---
    
    print("Solving with PuLP...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 hides the solver's own logs

    # --- 7. INTERPRET THE RESULTS ---
    
    print("\n--- RESULTS ---")
    print(f"Solver Status: {pulp.LpStatus[prob.status]}")

    # Check the value of our binary decision variable
    if pulp.value(goods_takes_loop) == 1:
        print("\n>>>>> DECISION: Goods Train should TAKE THE LOOP.")
    else:
        print("\n>>>>> DECISION: Goods Train should STAY ON THE MAIN LINE.")

    print("\nProjected Outcomes:")
    print(f"  - Goods Train will finish at: {pulp.value(finish_time_goods):.2f} minutes.")
    print(f"  - Express Train will finish at: {pulp.value(finish_time_express):.2f} minutes.")
    
    final_objective_value = (pulp.value(finish_time_goods) * goods_priority) + \
                            (pulp.value(finish_time_express) * express_priority)
    print(f"\nTotal 'Priority-Weighted Cost' of this solution: {final_objective_value:.2f}")


if __name__ == "__main__":
    # SCENARIO 1: A standard, high-priority express train.
    # We expect the goods train to be forced onto the loop.
    solve_train_overtake(
        scenario_name="Standard Operation (Express is High Priority)",
        goods_priority=1,
        express_priority=10
    )

    # SCENARIO 2: What if the "slow" train is actually a VIP train with a higher priority?
    # We expect the express train to be forced to wait.
    solve_train_overtake(
        scenario_name="VIP Movement (Slow Train is High Priority)",
        goods_priority=10,
        express_priority=5
    )