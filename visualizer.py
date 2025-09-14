import streamlit as st
import pandas as pd
import pulp
import random
import itertools
import matplotlib.pyplot as plt
import time
import redis
import json
import hashlib
import copy

# --- Train Class Definitions ---
TRAIN_CLASSES = {
    'Rajdhani/Shatabdi': {'speed_range': (100, 130), 'priority_range': (8, 10), 'color': 'gold', 'stop_penalty_mins': 2},
    'Express': {'speed_range': (80, 110), 'priority_range': (4, 7), 'color': 'limegreen', 'stop_penalty_mins': 5},
    'Mail': {'speed_range': (70, 90), 'priority_range': (3, 5), 'color': 'deepskyblue', 'stop_penalty_mins': 8},
    'Freight': {'speed_range': (50, 70), 'priority_range': (1, 2), 'color': 'lightcoral', 'stop_penalty_mins': 15}
}

# --- Redis Cache Setup ---
try:
    # Connect to a local Redis instance
    redis_conn = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_conn.ping()
    CACHE_ENABLED = True
except redis.exceptions.ConnectionError:
    CACHE_ENABLED = False

# --- Backend Solver and Data Generation Logic ---

def generate_bidirectional_network_data(num_main_stations=4, num_loop_stations=3, num_trains=20, seed=None):
    """Generates data for a dynamic network with bidirectional tracks and loops."""
    if seed is not None:
        random.seed(seed)
    
    stations = [f"Station_{i}" for i in range(num_main_stations)]
    loop_stations = [f"Loop_Station_{i}" for i in range(num_loop_stations)]
    
    junctions = []
    for i in range(num_loop_stations):
        junctions.append(f"Junction_{i}_A")
        junctions.append(f"Junction_{i}_B")

    network = {'stations': stations + loop_stations + junctions, 'segments': {}}
    
    if num_loop_stations > 0:
        network['segments'][(stations[0], junctions[0])] = {'length': random.randint(50, 80), 'type': 'single'}
    elif len(stations) > 1:
        network['segments'][(stations[0], stations[1])] = {'length': random.randint(80, 120), 'type': 'single'}

    for i in range(num_loop_stations):
        j_in, j_out = f"Junction_{i}_A", f"Junction_{i}_B"
        loop = loop_stations[i]
        
        network['segments'][(j_in, j_out)] = {'length': random.randint(10, 20), 'type': 'single'}
        network['segments'][(j_in, loop)] = {'length': random.randint(40, 60), 'type': 'single'}
        network['segments'][(loop, j_out)] = {'length': random.randint(40, 60), 'type': 'single'}
        
        if i < num_main_stations - 1:
            next_station = stations[i+1]
            network['segments'][(j_out, next_station)] = {'length': random.randint(50, 90), 'type': 'single'}
            if i + 1 < num_loop_stations and i + 1 < num_main_stations -1:
                next_j_in = f"Junction_{i+1}_A"
                network['segments'][(next_station, next_j_in)] = {'length': random.randint(50, 80), 'type': 'single'}
    
    for i in range(num_loop_stations, num_main_stations - 1):
        network['segments'][(stations[i], stations[i+1])] = {'length': random.randint(80, 120), 'type': 'single'}

    train_data = {}
    
    def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end: return [path]
        if start not in graph: return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths: paths.append(newpath)
        return paths

    adj_list = {}
    for u, v in network['segments'].keys():
        adj_list.setdefault(u, []).append(v)
        adj_list.setdefault(v, []).append(u)
    
    possible_routes = []
    if stations:
        start_node, end_node = stations[0], stations[-1]
        if start_node in adj_list and end_node in adj_list:
            possible_routes.extend(find_all_paths(adj_list, start_node, end_node))
            possible_routes.extend(find_all_paths(adj_list, end_node, start_node))
    if not possible_routes:
        possible_routes.append(stations)
        if len(stations) > 1: possible_routes.append(list(reversed(stations)))

    class_names = list(TRAIN_CLASSES.keys())
    for i in range(num_trains):
        train_id = f"Train_{i:02d}"
        train_class = random.choice(class_names)
        class_props = TRAIN_CLASSES[train_class]
        
        speed_min, speed_max = class_props['speed_range']
        base_speed = random.randint(speed_min, speed_max)
        p_min, p_max = class_props['priority_range']
        priority = random.randint(p_min, p_max)
        stop_penalty = class_props['stop_penalty_mins']
        scheduled_departure = i * 25 + random.randint(0, 10)
        
        if not possible_routes: continue
        route = random.choice(possible_routes)
        
        scheduled_arrival = scheduled_departure
        for j in range(len(route) - 1):
            u, v = route[j], route[j+1]
            key = (u, v) if (u, v) in network['segments'] else (v, u)
            segment = network['segments'][key]
            scheduled_arrival += (segment['length'] / base_speed) * 60
            if j < len(route) - 2: scheduled_arrival += stop_penalty
        
        train_data[train_id] = {
            'class': train_class, 'priority': priority, 'base_speed_kph': base_speed,
            'departure_time': scheduled_departure, 'origin': route[0], 'destination': route[-1], 
            'route': route, 'stop_penalty_mins': stop_penalty, 'scheduled_arrival': scheduled_arrival
        }

    op_params = {'headway_mins': 4}
    return train_data, network, op_params

def solve_bidirectional_schedule(train_data, network, op_params):
    """Solves the bidirectional scheduling problem and returns detailed results."""
    prob = pulp.LpProblem("BidirectionalTrainSchedule", pulp.LpMinimize)
    TRAINS, NODES = list(train_data.keys()), network['stations']
    TRAIN_PAIRS = list(itertools.combinations(TRAINS, 2))
    M, HEADWAY = 100000, op_params['headway_mins']

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

    # *** JUNCTION CAPACITY LOGIC REMOVED AS REQUESTED ***
    # The previous logic for JUNCTION_CAPACITY was making the problem infeasible.
    # We are reverting to the model where junctions have unlimited capacity,
    # which represents a more abstract but solvable version of the problem.

    def find_subpath(route, path): return any((route[i], route[i+1]) == path for i in range(len(route) - 1))
    
    for (t1, t2), seg_key in order_indices:
        u, v = seg_key
        order = train_order[((t1, t2), seg_key)]
        
        t1_uv, t1_vu = find_subpath(train_data[t1]['route'],(u,v)), find_subpath(train_data[t1]['route'],(v,u))
        t2_uv, t2_vu = find_subpath(train_data[t2]['route'],(u,v)), find_subpath(train_data[t2]['route'],(v,u))

        if not ((t1_uv or t1_vu) and (t2_uv or t2_vu)): continue

        if (t1_uv and t2_uv) or (t1_vu and t2_vu):
            start_node, end_node = (u, v) if t1_uv else (v, u)
            prob += departure_time[t2][start_node] >= departure_time[t1][start_node] + HEADWAY - M * (1 - order), f"H_Dep_{t1}_{t2}_{start_node}"
            prob += departure_time[t1][start_node] >= departure_time[t2][start_node] + HEADWAY - M * order, f"H_Dep_{t2}_{t1}_{start_node}"
            prob += arrival_time[t2][end_node] >= arrival_time[t1][end_node] + HEADWAY - M * (1 - order), f"H_Arr_{t1}_{t2}_{end_node}"
            prob += arrival_time[t1][end_node] >= arrival_time[t2][end_node] + HEADWAY - M * order, f"H_Arr_{t2}_{t1}_{end_node}"
            prob += arrival_time[t2][end_node] >= arrival_time[t1][end_node] - M * (1 - order), f"NoOvertake_{t1}_{t2}_{seg_key}"
            prob += arrival_time[t1][end_node] >= arrival_time[t2][end_node] - M * order, f"NoOvertake_{t2}_{t1}_{seg_key}"
        elif (t1_uv and t2_vu):
            prob += departure_time[t1][u] >= arrival_time[t2][u] - M * order, f"C_Opp_{t1}_{t2}_{seg_key}"
            prob += departure_time[t2][v] >= arrival_time[t1][v] - M * (1 - order), f"C_Opp_{t2}_{t1}_{seg_key}"
        elif (t1_vu and t2_uv):
            prob += departure_time[t2][u] >= arrival_time[t1][u] - M * order, f"C_Opp_{t2}_{t1}_{seg_key}"
            prob += departure_time[t1][v] >= arrival_time[t2][v] - M * (1 - order), f"C_Opp_{t1}_{t2}_{seg_key}"
    
    solver = pulp.HiGHS(msg=True, timeLimit=120)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        schedule_results = {}
        for t in TRAINS:
            schedule_results[t] = {}
            for n in NODES:
                schedule_results[t][n] = {
                    'arrival': arrival_time[t][n].value(),
                    'departure': departure_time[t][n].value()
                }
        return schedule_results, pulp.LpStatus[prob.status]
    return None, pulp.LpStatus[prob.status]

def format_controller_briefing(train_data, schedule_results):
    briefing = "## Controller's Briefing and Operational Plan\n\n"
    briefing += "### I. Train Performance Summary\n"
    
    results_summary = {}
    for t in train_data.keys():
        train = train_data[t]
        actual_arrival = schedule_results[t][train['destination']]['arrival']
        actual_departure = schedule_results[t][train['origin']]['departure']
        results_summary[t] = {
            'FinalDelay': actual_arrival - train['scheduled_arrival'], 'ScheduledArrival': train['scheduled_arrival'],
            'ActualArrival': actual_arrival, 'ScheduledDeparture': train['departure_time'],
            'ActualDeparture': actual_departure,
        }

    sorted_trains = sorted(results_summary.keys(), key=lambda t: train_data[t]['departure_time'])
    
    for train_id in sorted_trains:
        res = results_summary[train_id]
        delay = res['FinalDelay']
        status = "**ON TIME**"
        if delay > 10: status = f"**SIGNIFICANT DELAY** ({delay:.0f} mins)"
        elif delay > 1: status = f"**MINOR DELAY** ({delay:.0f} mins)"
        
        briefing += f"- **{train_id} (P{train_data[train_id]['priority']})**: Final Status - {status}\n"
        briefing += f"  - **Route**: {' -> '.join(train_data[train_id]['route'])}\n"
        briefing += f"  - **Scheduled**: Depart {res['ScheduledDeparture']:.0f}, Arrive {res['ScheduledArrival']:.0f}\n"
        briefing += f"  - **Actual**: Depart {res['ActualDeparture']:.0f}, Arrive {res['ActualArrival']:.0f}\n"
        if delay > 1:
            initial_delay = res['ActualDeparture'] - res['ScheduledDeparture']
            if initial_delay > 1:
                briefing += f"  - *Delayed by an initial hold of {initial_delay:.0f} mins to resolve conflicts.*\n\n"
            else:
                briefing += f"  - *Incurred delays en-route due to network congestion.*\n\n"

    return briefing

# --- Visualization Logic ---

def get_node_positions(network):
    positions = {}
    x_pos = 0
    stations = sorted([s for s in network['stations'] if 'Station' in s and 'Loop' not in s])
    loop_stations = sorted([l for l in network['stations'] if 'Loop' in l])

    ordered_nodes = []
    if stations: ordered_nodes.append(stations[0])
    num_loops = len(loop_stations)
    for i in range(num_loops):
        ordered_nodes.append(f'Junction_{i}_A')
        ordered_nodes.append(f'Junction_{i}_B')
        if i < len(stations) -1: ordered_nodes.append(stations[i+1])
    for s in stations:
        if s not in ordered_nodes: ordered_nodes.append(s)

    for node in ordered_nodes:
        if node not in positions:
            positions[node] = (x_pos, 0)
            x_pos += 2
            
    for i in range(num_loops):
      j_in_name, j_out_name = f'Junction_{i}_A', f'Junction_{i}_B'
      loop_name = f'Loop_Station_{i}'
      if j_in_name in positions and j_out_name in positions:
          j_in_pos, j_out_pos = positions[j_in_name], positions[j_out_name]
          positions[loop_name] = ((j_in_pos[0] + j_out_pos[0]) / 2, 1)

    return positions

def get_train_positions_at_time_t(t, schedule, train_data, node_pos):
    train_positions = {}
    junction_dwell_count = {j: 0 for j in node_pos if 'Junction' in j}

    for train_id, data in train_data.items():
        route = data['route']
        position_found = False
        for i in range(len(route) - 1):
            u_node, v_node = route[i], route[i+1]
            dep_time = schedule[train_id][u_node]['departure']
            arr_time = schedule[train_id][v_node]['arrival']

            if dep_time is None or arr_time is None: continue
            if i == 0 and t < dep_time:
                position_found = True; break
            if dep_time <= t < arr_time:
                duration = arr_time - dep_time
                progress = (t - dep_time) / duration if duration > 0 else 0
                x1, y1 = node_pos[u_node]
                x2, y2 = node_pos[v_node]
                pos_x = x1 + progress * (x2 - x1)
                pos_y = y1 + progress * (y2 - y1)
                train_positions[train_id] = (pos_x, pos_y)
                position_found = True; break
            
            next_dep_time = schedule[train_id][v_node]['departure']
            if next_dep_time and arr_time <= t < next_dep_time:
                if 'Junction' in v_node:
                    x, y = node_pos[v_node]
                    offset_x = (junction_dwell_count[v_node] % 4) * 0.25 - 0.375
                    offset_y = (junction_dwell_count[v_node] // 4) * 0.25 + 0.2
                    train_positions[train_id] = (x + offset_x, y + offset_y)
                    junction_dwell_count[v_node] += 1
                else:
                    train_positions[train_id] = node_pos[v_node]
                position_found = True; break
        
        if not position_found:
            final_node = route[-1]
            if schedule[train_id][final_node]['arrival'] and t >= schedule[train_id][final_node]['arrival']:
                train_positions[train_id] = node_pos[final_node]
                
    return train_positions

def get_event_log(t, schedule, train_data):
    events = []
    for train_id, data in train_data.items():
        for node in data['route']:
            arr = schedule[train_id][node]['arrival'] 
            dep = schedule[train_id][node]['departure']
            if arr is not None and arr <= t:
                events.append((arr, f"T:{arr:.0f} - {train_id} (P{data['priority']}) ARRIVED at {node}"))
            if dep is not None and dep <= t:
                events.append((dep, f"T:{dep:.0f} - {train_id} (P{data['priority']}) DEPARTED from {node}"))
    events.sort()
    return "\n".join([msg for time, msg in events])

def format_schedule_as_dataframe(train_data, schedule_results):
    records = []
    for t_id, t_data in train_data.items():
        record = {
            "Train ID": t_id, "Class": t_data.get('class', 'N/A'), "Priority": t_data['priority'],
            "Route": ' -> '.join(t_data['route']), "Sch. Dep.": t_data['departure_time'],
            "Act. Dep.": schedule_results[t_id][t_data['origin']]['departure'],
            "Sch. Arr.": t_data['scheduled_arrival'],
            "Act. Arr.": schedule_results[t_id][t_data['destination']]['arrival'],
        }
        delay = record["Act. Arr."] - record["Sch. Arr."]
        record["Delay (mins)"] = f"{delay:.1f}"
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df[["Train ID", "Class", "Priority", "Sch. Dep.", "Act. Dep.", "Sch. Arr.", "Act. Arr.", "Delay (mins)", "Route"]]
    df = df.sort_values(by="Sch. Dep.").reset_index(drop=True)
    return df

# --- Streamlit Application UI ---

st.set_page_config(layout="wide")
st.title("Railway Operations Scheduler & Visualizer")

if 'simulation_data' not in st.session_state: st.session_state['simulation_data'] = None
if 'animating' not in st.session_state: st.session_state['animating'] = False
if 'current_time' not in st.session_state: st.session_state['current_time'] = 0.0
if 'animation_speed' not in st.session_state: st.session_state['animation_speed'] = 1
if 'scenario_seed' not in st.session_state: st.session_state['scenario_seed'] = 42 # Default seed

# --- Display KPIs ---
if st.session_state.get('simulation_data'):
    data = st.session_state.simulation_data
    c1, c2 = st.columns(2)
    c1.metric("Solver Status", data.get('status', 'N/A'))
    c2.metric("Total Delay (mins)", f"{data.get('total_delay', 0):.1f}")

# Display Redis connection status in the sidebar
if CACHE_ENABLED:
    st.sidebar.success("Redis connected. Caching is ON.")
else:
    st.sidebar.warning("Redis not found. Caching is OFF.")

st.sidebar.header("Controls")

# Add a number input for the scenario seed to control randomness
st.session_state.scenario_seed = st.sidebar.number_input(
    "Scenario Seed",
    value=st.session_state.scenario_seed,
    step=1,
    help="Change this number to generate a new random scenario. The same seed will always produce the same scenario."
)

num_trains = st.sidebar.number_input("Number of Trains", min_value=1, max_value=50, value=20, step=1)
num_main_stations = st.sidebar.number_input("Number of Main Stations", min_value=2, max_value=10, value=4, step=1)
num_loop_stations = st.sidebar.number_input("Number of Loop Stations", min_value=0, max_value=5, value=3, step=1)

if st.sidebar.button("Generate & Run New Simulation"):
    st.session_state['animating'] = False
    with st.spinner("Generating scenario and solving schedule... This may take up to 2 minutes."):
        train_data, network, op_params = generate_bidirectional_network_data(
            num_main_stations=num_main_stations,
            num_loop_stations=num_loop_stations,
            num_trains=num_trains,
            seed=st.session_state.scenario_seed
        )
        
        # --- Caching Logic ---
        schedule_results, status = None, "Not Solved"
        cache_key = None # Initialize cache_key
        
        if CACHE_ENABLED:
            # Create a JSON-serializable representation of the network for the cache key
            serializable_network = copy.deepcopy(network)
            serializable_network['segments'] = {str(k): v for k, v in network['segments'].items()}
            
            # Create a stable representation of the input for hashing
            input_data_str = json.dumps((train_data, serializable_network, op_params), sort_keys=True)
            input_hash = hashlib.sha256(input_data_str.encode()).hexdigest()
            cache_key = f"rail_sim_cache:{input_hash}"
            
            cached_result = redis_conn.get(cache_key)
            if cached_result:
                st.success("Loaded result from Redis cache!")
                cached_data = json.loads(cached_result)
                schedule_results, status = cached_data['schedule'], cached_data['status']
            else:
                st.info("Cache miss. Running optimizer...")

        if schedule_results is None:
            schedule_results, status = solve_bidirectional_schedule(train_data, network, op_params)
            # If the solver was successful and caching is on, save the result
            if CACHE_ENABLED and schedule_results and cache_key:
                result_to_cache = json.dumps({'schedule': schedule_results, 'status': status})
                redis_conn.set(cache_key, result_to_cache)
                st.info("Saved new result to Redis cache.")
        
        if schedule_results:
            total_delay = sum(
                max(0, schedule_results[t][train_data[t]['destination']]['arrival'] - train_data[t]['scheduled_arrival'])
                for t in train_data
            )
            st.session_state['simulation_data'] = {
                "train_data": train_data, "network": network,
                "schedule": schedule_results,
                "briefing": format_controller_briefing(train_data, schedule_results),
                "total_delay": total_delay,
                "status": status,
            }
            st.session_state['current_time'] = 0.0
            st.success(f"Schedule Found! Status: {status}")
        else:
            st.error(f"Solver failed to find a feasible solution. Status: {status}")
            st.session_state['simulation_data'] = None

if st.session_state.simulation_data:
    st.sidebar.subheader("Animation")
    start_stop_text = "Stop Animation" if st.session_state.animating else "Start Animation"
    if st.sidebar.button(start_stop_text): st.session_state.animating = not st.session_state.animating
    if st.sidebar.button("Reset Time"):
        st.session_state.current_time = 0.0
        st.session_state.animating = False
    st.session_state.animation_speed = st.sidebar.select_slider(
        "Animation Speed", options=[1, 2, 5, 10, 20, 50], value=st.session_state.animation_speed
    )

    data = st.session_state.simulation_data
    train_data, network, schedule = data['train_data'], data['network'], data['schedule']
    
    max_time = 0.0
    for t_data in schedule.values():
        for n_data in t_data.values():
            if n_data['arrival'] and n_data['arrival'] > max_time: max_time = n_data['arrival']

    st.session_state.current_time = st.slider("Simulation Time (minutes)", 0.0, max_time, st.session_state.current_time)
    current_time = st.session_state.current_time

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Network Visualization")
        node_positions = get_node_positions(network)
        train_positions = get_train_positions_at_time_t(current_time, schedule, train_data, node_positions)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for u, v in network['segments'].keys():
            x_vals = [node_positions[u][0], node_positions[v][0]]
            y_vals = [node_positions[u][1], node_positions[v][1]]
            ax.plot(x_vals, y_vals, 'k-', alpha=0.3, zorder=1)

        for name, pos in node_positions.items():
            ax.plot(pos[0], pos[1], 's', markersize=12, color='skyblue', zorder=2)
            ax.text(pos[0], pos[1] + 0.05, name, ha='center', va='bottom', fontsize=9)
            if 'Junction' in name:
                ax.add_patch(plt.Rectangle((pos[0] - 0.5, pos[1] + 0.15), 1.0, 0.5, 
                                          edgecolor='gray', facecolor='whitesmoke', linestyle='--', zorder=0))

        for train_id, pos in train_positions.items():
            train_class = train_data[train_id].get('class', 'Express')
            color = TRAIN_CLASSES[train_class]['color']
            ax.plot(pos[0], pos[1], 'o', markersize=10, color=color, zorder=3, markeredgecolor='black')
            ax.text(pos[0], pos[1] - 0.1, train_id.split('_')[1], color='black', ha='center', va='top', fontsize=8, weight='bold')

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f"{name}",
                                      markerfacecolor=props['color'], markersize=10)
                           for name, props in TRAIN_CLASSES.items()]
        ax.legend(handles=legend_elements, loc='upper left', title='Train Classes')
        max_x = max(p[0] for p in node_positions.values()) if node_positions else 10
        ax.set_xlim(-1, max_x + 1)
        ax.set_ylim(-0.5, 2.0)
        ax.set_title(f"Train Positions at Time: {current_time:.0f} mins")
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)

    with col2:
        st.subheader("Controller's Log")
        event_log_text = get_event_log(current_time, schedule, train_data)
        st.text_area("Live Events", event_log_text, height=400)

    st.markdown("---")
    st.markdown(data['briefing'])
    st.markdown("---")
    st.subheader("Detailed Schedule")
    schedule_df = format_schedule_as_dataframe(train_data, schedule)
    st.dataframe(schedule_df, hide_index=True, use_container_width=True)

    if st.session_state.animating:
        time_increment = 0.5 * st.session_state.animation_speed
        new_time = st.session_state.current_time + time_increment
        if new_time < max_time: st.session_state.current_time = new_time
        else:
            st.session_state.current_time = max_time
            st.session_state.animating = False
        time.sleep(1/120)
        st.rerun()

else:
    st.info("Click 'Generate & Run New Simulation' in the sidebar to begin.")