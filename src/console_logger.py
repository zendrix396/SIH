import os

def clear_console():
    """Clears the console screen."""
    # os.system('cls' if os.name == 'nt' else 'clear')
    pass

def get_train_type_counts(trains):
    """Counts the number of trains of each type."""
    counts = {"EXPRESS": 0, "PASSENGER": 0, "FREIGHT": 0}
    for train in trains:
        if train.type in counts:
            counts[train.type] += 1
    return counts

def print_debug_info(env):
    """Prints a detailed, formatted debug screen to the console."""
    
    clear_console()
    
    header = " AI CONTROLLER ASSIST: DIAMOND JUNCTION "
    print("=" * 60)
    print(f"{header:#^60}")
    print("=" * 60)
    
    # --- Time ---
    sim_hours = int(env.sim_time // 3600)
    sim_mins = int((env.sim_time % 3600) // 60)
    run_mins = int(env.sim_time / 60)
    print(f"SIMULATION TIME: {sim_hours:02d}:{sim_mins:02d}  |  RUNNING FOR: {run_mins} Minutes")
    print("---")
    
    # --- Performance ---
    print("[PERFORMANCE METRICS]")
    gain = ((env.current_throughput / env.cfg['BASELINE_THROUGHPUT']) - 1) * 100 if env.cfg['BASELINE_THROUGHPUT'] > 0 else 0
    print(f">> SECTION THROUGHPUT: {env.current_throughput:.1f} Trains/Hour")
    print(f"   BASELINE THROUGHPUT (Est.): {env.cfg['BASELINE_THROUGHPUT']} Trains/Hour")
    print(f">> CAPACITY GAIN: {gain:+.0f}%")
    print()

    # --- Live Status ---
    print("[LIVE STATUS]")
    trains_in_section = list(env.trains.values())
    type_counts = get_train_type_counts(trains_in_section)
    counts_str = ", ".join(f"{v} {k}" for k, v in type_counts.items() if v > 0)
    print(f"- TRAINS IN SECTION: {len(trains_in_section)} ({counts_str})")
    print(f"- TRAINS WAITING: {len(env._get_waiting_trains())}")
    print(f"- TOTAL NETWORK DELAY: {int(env.total_delay/60)} Minutes")
    print("=" * 60)
    print()

    # --- Latest Decision ---
    print("\n[LATEST AI DECISION]")
    
    # --- SIMPLIFIED: Print the pre-formatted action string directly ---
    action_str = env.last_decision['action']

    print(f"AI ACTION: {action_str}")
    print(f"RATIONALE: {env.last_decision['rationale']}")
    print("-" * 60)

    # --- Detailed Waiting Trains View ---
    print("\n[WAITING TRAINS DETAIL]")
    waiting_trains = env._get_waiting_trains()
    if not waiting_trains:
        print("No trains are currently waiting at signals.")
    else:
        for signal, train in waiting_trains.items():
            path_str = " -> ".join(train.path_node_names)
            print(f"- Signal {signal}: {train.type} #{train.id} (Prio: {train.priority})")
            print(f"  Path: {path_str}")
            print(f"  Delay: {int(train.delay/60)} mins")
            
            # Look ahead on path
            path_blocked = False
            for i in range(train.path_progress_idx // env.cfg["POINTS_PER_SEGMENT"], len(train.path_node_names) - 1):
                edge = tuple(sorted((train.path_node_names[i], train.path_node_names[i+1])))
                occupying_train_id = env.track_occupancy.get(edge)
                if occupying_train_id and occupying_train_id != train.id:
                    print(f"  [!] PATH BLOCKED at {edge} by Train #{occupying_train_id}")
                    path_blocked = True
                    break
            if not path_blocked:
                print("  [✓] Path ahead is clear.")

    # --- NEW: All Trains on Map View ---
    print("\n[ALL TRAINS ON MAP]")
    if not env.trains:
        print("No trains currently in the section.")
    else:
        for train in sorted(list(env.trains.values()), key=lambda t: t.id):
            current_node_idx = train.path_progress_idx // env.cfg["POINTS_PER_SEGMENT"]
            
            if current_node_idx < len(train.path_node_names) - 1:
                loc_str = f"between {train.path_node_names[current_node_idx]} and {train.path_node_names[current_node_idx+1]}"
            else:
                loc_str = f"at {train.path_node_names[-1]}"

            speed_kph = train.current_speed * 3.6
            print(f"- {train.id} ({train.type}): {train.status} @ {speed_kph:.0f} km/h, {loc_str}")

    print("="*60)
