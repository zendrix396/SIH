import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import pygame

from src.config import CONFIG

class Train:
    def __init__(self, train_id, train_type, path_key):
        self.id = train_id
        self.type = train_type
        self.priority = CONFIG["PRIORITY"][train_type]
        self.speed_mps = CONFIG["SPEED_KPH"][train_type] * 1000 / 3600 # Convert km/h to m/s
        self.color = CONFIG[f"COLOR_TRAIN_{train_type.upper()}"]
        
        self.path_key = path_key # Store the key to look up the high-res path
        self.path_node_names = CONFIG["PATHS"][path_key]
        
        # --- NEW: Physics and High-Res Path Tracking ---
        self.high_res_path = CONFIG["PATHS_HIGH_RES"][self.path_key]
        self.path_progress_idx = 0
        self.current_pos = self.high_res_path[0]
        self.current_speed = 0.0 # m/s
        self.target_speed = 0.0 # m/s
        
        # --- REVISED: Simplified state machine ---
        self.status = "STOPPED" # STOPPED -> MOVING
        self.ideal_departure_time = 0
        self.actual_departure_time = 0
        self.delay = 0 # in seconds

        self.reroute(path_key) # Initial setup

    def reroute(self, new_path_key):
        """Updates the train's path to a new one."""
        self.path_key = new_path_key
        self.path_node_names = CONFIG["PATHS"][new_path_key]
        self.high_res_path = CONFIG["PATHS_HIGH_RES"][new_path_key]
        # Reset progress to the start of the new path
        self.path_progress_idx = 0 
        self.current_pos = self.high_res_path[0]


class RailwayEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": CONFIG["FPS"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = None
        self.cfg = CONFIG # Make config accessible to logger

        # --- NEW: DYNAMIC RE-ROUTING ACTION SPACE ---
        # For each of the 4 entry signals, the agent can choose:
        # 0: Hold, 1: Take Primary Route, 2: Take Alternative Route
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        
        # --- REVISED: Observation Space with Alt-Route Info ---
        # For each signal: [has_train, prio, delay, primary_path_occ, alt_path_occ]
        num_signal_features = 5
        num_signals = len(CONFIG["SIGNAL_NODES"])
        num_junction_tracks = 4
        obs_size = num_signals * num_signal_features + num_junction_tracks
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim_time = 0 # seconds
        self.current_step = 0
        self.last_disruption_time = 0
        
        self.trains = {}
        self._load_schedule()
        
        # State variables
        self.signals = {node: "RED" for node in CONFIG["SIGNAL_NODES"]}
        self.track_occupancy = {edge: None for edge in CONFIG["EDGES"]}
        
        # KPI Tracking
        self.total_delay = 0
        self.cumulative_trains_finished = 0
        self.newly_finished_trains = 0
        self.total_delay_at_last_step = 0
        self.throughput_deque = deque(maxlen=20) # Store timestamps of last 20 finished trains
        self.last_decision = {"action": "N/A", "rationale": "System Initialized."}

        return self._get_observation(), {}

    def _load_schedule(self):
        self.schedule = []
        # --- NEW: Denser, more complex, and stochastic schedule ---
        schedule_data = [
            # Morning Rush Hour (Head-on conflicts)
            (0, "EXPRESS", "N_S"),
            (60, "EXPRESS", "S_N"), 
            (120, "PASSENGER", "W_E"),
            (180, "PASSENGER", "E_W"),

            # Mid-day Freight Wave (Resource contention)
            (600, "FREIGHT", "N_TERM_E_IND"),
            (630, "FREIGHT", "S_W_SID"),
            (660, "FREIGHT", "W_E"),
            (690, "FREIGHT", "E_HUB_N"),

            # Afternoon Passenger Peak (Complex junction interactions)
            (1200, "PASSENGER", "W_HUB_S"),
            (1215, "EXPRESS", "E_HUB_N"),
            (1230, "PASSENGER", "N_S"),
            (1245, "EXPRESS", "S_N"),
            (1260, "PASSENGER", "E_W"),

            # Evening Rush
            (1800, "EXPRESS", "W_E"),
            (1830, "PASSENGER", "S_N"),
            (1860, "EXPRESS", "N_S"),
            (1890, "FREIGHT", "S_W_SID"),
        ]

        for i, (departure_time, train_type, path_key) in enumerate(schedule_data):
            train = Train(f"T{101+i}", train_type, path_key)
            # --- NEW: Add stochastic delay imperfection ---
            imperfection = random.uniform(0.95, 1.15)
            train.ideal_departure_time = departure_time * imperfection
            self.schedule.append(train)
        
        # Sort schedule by departure time after adding imperfections
        self.schedule.sort(key=lambda t: t.ideal_departure_time)

    def _get_path_occupancy(self, path_node_names):
        """Calculates the normalized occupancy of a given path."""
        # This is a placeholder. In a real simulation, you'd track occupancy
        # along the path and return a normalized value.
        # For now, we'll return a dummy value.
        return 0.0 # No occupancy tracking implemented yet

    def _get_observation(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Signal Features
        waiting_trains = self._get_waiting_trains()
        for i, signal_node in enumerate(self.cfg["SIGNAL_NODES"]):
            if signal_node in waiting_trains:
                train = waiting_trains[signal_node]
                obs[i*5 + 0] = 1.0 # has_train
                obs[i*5 + 1] = train.priority / 3.0 # normalized priority
                obs[i*5 + 2] = min(train.delay / 3600.0, 1.0) # normalized delay
                
                # --- NEW: Observe occupancy of BOTH primary and alt paths ---
                primary_path_key = train.path_key
                obs[i*5 + 3] = self._get_path_occupancy(self.cfg["PATHS"][primary_path_key])

                alt_path_key = self.cfg["ALT_ROUTE_MAP"].get(primary_path_key)
                if alt_path_key:
                    obs[i*5 + 4] = self._get_path_occupancy(self.cfg["PATHS"][alt_path_key])


        # Junction Occupancy
        junction_nodes = {"J1", "J2", "J3", "J4"}
        idx = len(CONFIG["SIGNAL_NODES"]) * 5
        for edge in self.track_occupancy:
            if self.track_occupancy[edge] is not None:
                if edge[0] in junction_nodes or edge[1] in junction_nodes:
                    # Simple hash to spread occupancy across the observation space
                    track_idx = hash(edge) % 4
                    obs[idx + track_idx] = 1.0
                    
        return obs

    def step(self, action):
        self.current_step += 1
        self.sim_time += self.cfg["SIM_STEP_SECONDS"]

        # 1. Apply Action: Set Signals
        self._apply_action(action)
        
        # 2. Update World State
        self._update_trains()
        self._spawn_trains()
        self._inject_disruptions()
        
        # 3. Calculate Reward
        reward = self._calculate_reward()
        
        # 4. Check for Termination
        terminated = self.current_step >= self.cfg["MAX_EPISODE_STEPS"]
        
        # 5. Get next observation
        observation = self._get_observation()
        
        # 6. Update KPIs for rendering
        self._update_kpis()

        # 7. Reset step-specific counters
        self.newly_finished_trains = 0
        
        return observation, reward, terminated, False, {}

    def _apply_action(self, actions):
        # --- REVISED: Generate a clear, consolidated action summary ---
        action_descriptions = []
        
        # First, turn all signals RED as a safe default
        for signal in self.signals:
            self.signals[signal] = "RED"

        waiting_trains = self._get_waiting_trains()

        for i, action_choice in enumerate(actions):
            if action_choice == 0 or i >= len(self.cfg["SIGNAL_NODES"]): # Hold or invalid index
                continue

            signal = self.cfg["SIGNAL_NODES"][i]
            if signal in waiting_trains:
                train = waiting_trains[signal]
                
                chosen_path_key = train.path_key
                action_type = "Proceed"
                if action_choice == 2: # Take Alt Route
                    alt_key = self.cfg["ALT_ROUTE_MAP"].get(train.path_key)
                    if alt_key:
                        chosen_path_key = alt_key
                        action_type = "Reroute"
                
                chosen_path_nodes = self.cfg["PATHS"][chosen_path_key]
                first_edge = tuple(sorted((chosen_path_nodes[0], chosen_path_nodes[1])))
                
                # Check if the path is clear before setting the signal green
                if self.track_occupancy.get(first_edge) is None:
                    train.reroute(chosen_path_key)
                    self.signals[signal] = "GREEN"
                    action_descriptions.append(f"{action_type} {train.id} at {signal}")

        # Consolidate the log message for the console logger
        if not action_descriptions:
            final_action = "Hold all signals at RED"
            rationale = "No waiting trains or all paths are blocked."
        else:
            final_action = "; ".join(action_descriptions)
            rationale = "Executing AI-determined optimal actions."
        
        self.last_decision = {
            "action": final_action,
            "rationale": rationale
        }

    def _update_trains(self):
        for train_id, train in list(self.trains.items()):
            
            # --- NEW: Physics-Based Movement ---
            
            # 1. Determine Target Speed
            # Check for red signals or occupied tracks within braking distance
            braking_dist = (train.current_speed**2) / (2 * CONFIG["DECELERATION"]) if CONFIG["DECELERATION"] > 0 else float('inf')
            
            should_stop = False
            # Check for red signal at the next major node
            current_node_idx = train.path_progress_idx // CONFIG["POINTS_PER_SEGMENT"]
            if current_node_idx + 1 < len(train.path_node_names):
                next_major_node = train.path_node_names[current_node_idx + 1]
                if next_major_node in self.signals and self.signals[next_major_node] == "RED":
                    dist_to_signal = np.linalg.norm(np.array(CONFIG["NODES"][next_major_node]) - train.current_pos)
                    if dist_to_signal <= braking_dist:
                        should_stop = True

            # Check for occupied tracks ahead
            # (Simple check for now, can be improved)
            if not should_stop and current_node_idx + 1 < len(train.path_node_names):
                 next_major_node = train.path_node_names[current_node_idx + 1]
                 current_major_node = train.path_node_names[current_node_idx]
                 edge = tuple(sorted((current_major_node, next_major_node)))
                 if self.track_occupancy.get(edge) not in [None, train.id]:
                     should_stop = True

            if should_stop:
                train.target_speed = 0.0
                train.status = "STOPPED"
            else:
                train.target_speed = train.speed_mps
                train.status = "MOVING"

            # 2. Update Current Speed (Acceleration/Deceleration)
            if train.current_speed < train.target_speed:
                train.current_speed += CONFIG["ACCELERATION"] * CONFIG["SIM_STEP_SECONDS"]
                train.current_speed = min(train.current_speed, train.target_speed)
            elif train.current_speed > train.target_speed:
                train.current_speed -= CONFIG["DECELERATION"] * CONFIG["SIM_STEP_SECONDS"]
                train.current_speed = max(train.current_speed, train.target_speed)

            # 3. Update Position along High-Res Path
            distance_to_move = train.current_speed * CONFIG["SIM_STEP_SECONDS"]
            
            while distance_to_move > 0 and train.path_progress_idx < len(train.high_res_path) - 1:
                vec_to_next_point = train.high_res_path[train.path_progress_idx + 1] - train.current_pos
                dist_to_next_point = np.linalg.norm(vec_to_next_point)

                if distance_to_move >= dist_to_next_point:
                    train.current_pos = train.high_res_path[train.path_progress_idx + 1]
                    train.path_progress_idx += 1
                    distance_to_move -= dist_to_next_point
                else:
                    train.current_pos += (vec_to_next_point / dist_to_next_point) * distance_to_move
                    distance_to_move = 0
            
            # --- Update Track Occupancy (Coarse Grained) ---
            new_node_idx = train.path_progress_idx // CONFIG["POINTS_PER_SEGMENT"]
            if new_node_idx != current_node_idx and new_node_idx + 1 < len(train.path_node_names):
                # Free old track
                old_edge = tuple(sorted((train.path_node_names[current_node_idx], train.path_node_names[current_node_idx+1])))
                if self.track_occupancy.get(old_edge) == train.id:
                    self.track_occupancy[old_edge] = None
                # Occupy new track
                new_edge = tuple(sorted((train.path_node_names[new_node_idx], train.path_node_names[new_node_idx+1])))
                self.track_occupancy[new_edge] = train.id


            # --- Handle Finished Trains ---
            if train.path_progress_idx >= len(train.high_res_path) - 1:
                 # Free final track
                final_edge = tuple(sorted((train.path_node_names[-2], train.path_node_names[-1])))
                if self.track_occupancy.get(final_edge) == train.id:
                    self.track_occupancy[final_edge] = None

                if train.id in self.trains:
                    del self.trains[train_id]
                    self.cumulative_trains_finished += 1
                    self.newly_finished_trains += 1
                    self.throughput_deque.append(self.sim_time)
            
            # Update delay
            if train.id in self.trains:
                train.delay = max(0, self.sim_time - train.ideal_departure_time)

    def _spawn_trains(self):
        # --- NEW: Continuous Spawning ---
        if len(self.trains) + len(self.schedule) < 15: # Maintain ~15 trains
            new_train = self._generate_random_train()
            self.schedule.append(new_train)
            self.schedule.sort(key=lambda t: t.ideal_departure_time)

        # Spawn from schedule
        for train in self.schedule[:]:
            if self.sim_time >= train.ideal_departure_time:
                train.status = "WAITING"
                train.actual_departure_time = self.sim_time
                self.trains[train.id] = train
                self.schedule.remove(train)

    def _generate_random_train(self):
        """Creates a new random train for continuous spawning."""
        train_id = f"T{len(self.trains) + 101}"
        train_type = random.choice(["EXPRESS", "PASSENGER", "FREIGHT"])
        path_key = random.choice(list(CONFIG["PATHS"].keys()))
        return Train(train_id, train_type, path_key)

    def _inject_disruptions(self):
        if self.sim_time - self.last_disruption_time > CONFIG["DISRUPTION_INTERVAL_SECONDS"]:
            if self.schedule:
                # Pick a future train and add a delay
                train_to_disrupt = random.choice(self.schedule)
                delay_amount = random.choice([900, 1800, 2700]) # 15, 30, 45 mins
                train_to_disrupt.ideal_departure_time += delay_amount
                self.last_disruption_time = self.sim_time

    def _calculate_reward(self):
        # --- REVISED: Reward function with proactive "Reward Shaping" ---
        reward = 0

        # 1. The Ultimate Goal: Strong reward for newly finished trains.
        reward += self.newly_finished_trains * 250.0

        # 2. Penalize the *increase* in total delay from the last step.
        # This discourages making things worse overall.
        current_total_delay = sum(train.delay for train in self.trains.values())
        delay_change = current_total_delay - self.total_delay_at_last_step
        if delay_change > 0:
            reward -= delay_change / 100.0 # Only penalize if delay gets worse

        # --- THE CORE FIX: REWARD SHAPING ---
        # Instead of just penalizing waiting, we will reward productive actions.
        actions = self.last_decision['action'] # This is the array [0, 1, 2, ...]
        waiting_trains = self._get_waiting_trains()

        # A) Give a small reward for each *correct* GREEN signal
        if isinstance(actions, list):
            for i, action_choice in enumerate(actions):
                if action_choice > 0: # If the AI chose to do something (not HOLD)
                    if i < len(self.cfg["SIGNAL_NODES"]):
                        signal = self.cfg["SIGNAL_NODES"][i]
                        if signal in waiting_trains:
                            # GOOD: AI opened a signal for a waiting train. Give it a cookie.
                            reward += 25.0
                        else:
                            # BAD: AI opened a signal where no train was waiting.
                            reward -= 50.0 # Heavy penalty for a useless, unsafe action

        # B) Add a small, constant penalty for every train that is kept waiting.
        # This creates a constant pressure to DO SOMETHING.
        reward -= len(waiting_trains) * 2.0

        # Update state for the next step's calculation
        self.total_delay_at_last_step = current_total_delay
        
        return reward

    def _update_kpis(self):
        self.total_delay = sum(train.delay for train in self.trains.values())
        
        # --- REVISED: Calculate average throughput for the episode ---
        if self.sim_time > 0:
            # Throughput is total finished trains over elapsed time in hours
            self.current_throughput = self.cumulative_trains_finished / (self.sim_time / 3600.0)
        else:
            self.current_throughput = 0

    def _get_waiting_trains(self):
        waiting = {}
        for train in self.trains.values():
            if train.status == "WAITING" or train.status == "SCHEDULED":
                 current_node = train.path_node_names[train.path_progress_idx // CONFIG["POINTS_PER_SEGMENT"]]
                 if current_node in CONFIG["SIGNAL_NODES"]:
                     if current_node not in waiting:
                         waiting[current_node] = train
        return waiting

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from src.renderer import Renderer
                self.renderer = Renderer(self)
            self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
