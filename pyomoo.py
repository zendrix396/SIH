from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary, NonNegativeReals, value

def solve_train_overtake_pyomo(scenario_name, goods_priority, express_priority):
    print(f"\n==========================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"==========================================================")
    
    # --- 1. DEFINE THE WORLD (Our Inputs) ---
    travel_time_goods = 60
    travel_time_express = 30
    loop_penalty_goods = 15

    print("Inputs:")
    print(f"  - Goods Train: Priority={goods_priority}, Base Travel Time={travel_time_goods} mins")
    print(f"  - Express Train: Priority={express_priority}, Base Travel Time={travel_time_express} mins")
    print(f"  - Loop Penalty for Goods: {loop_penalty_goods} mins")
    print("-" * 30)
    
    # --- 2. CREATE PYOMO MODEL ---
    model = ConcreteModel()

    # --- 3. DEFINE VARIABLES ---
    model.goods_takes_loop = Var(domain=Binary)
    model.finish_time_goods = Var(domain=NonNegativeReals)
    model.finish_time_express = Var(domain=NonNegativeReals)

    # --- 4. OBJECTIVE FUNCTION ---
    model.objective = Objective(
        expr = (model.finish_time_goods * goods_priority) + (model.finish_time_express * express_priority), 
        sense=1  # min
    )

    # --- 5. CONSTRAINTS ---
    model.goods_finish_time_rule = Constraint(
        expr = model.finish_time_goods >= travel_time_goods + (model.goods_takes_loop * loop_penalty_goods)
    )

    M = 10000  # Big M for deactivating block constraint
    model.express_blocked_by_goods_rule = Constraint(
        expr = model.finish_time_express >= model.finish_time_goods - (M * model.goods_takes_loop)
    )

    model.express_minimum_time_rule = Constraint(
        expr = model.finish_time_express >= travel_time_express
    )

    # --- 6. SOLVE ---
    solver = SolverFactory('cbc')  # Use CBC as default MILP solver
    print("Solving with Pyomo...")
    result = solver.solve(model, tee=False)  # tee=False hides solver logs

    # --- 7. INTERPRET THE RESULTS ---
    print("\n--- RESULTS ---")
    status = result.solver.status
    termination = result.solver.termination_condition
    print(f"Solver Status: {status}, Termination: {termination}")

    # Decision interpretation
    if value(model.goods_takes_loop) == 1:
        print("\n>>>>> DECISION: Goods Train should TAKE THE LOOP.")
    else:
        print("\n>>>>> DECISION: Goods Train should STAY ON THE MAIN LINE.")

    print("\nProjected Outcomes:")
    print(f"  - Goods Train will finish at: {value(model.finish_time_goods):.2f} minutes.")
    print(f"  - Express Train will finish at: {value(model.finish_time_express):.2f} minutes.")

    final_objective_value = (value(model.finish_time_goods) * goods_priority) + \
                            (value(model.finish_time_express) * express_priority)
    print(f"\nTotal 'Priority-Weighted Cost' of this solution: {final_objective_value:.2f}")

if __name__ == "__main__":
    solve_train_overtake_pyomo(
        scenario_name="Standard Operation (Express is High Priority)",
        goods_priority=1,
        express_priority=10
    )

    solve_train_overtake_pyomo(
        scenario_name="VIP Movement (Slow Train is High Priority)",
        goods_priority=10,
        express_priority=5
    )
