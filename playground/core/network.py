import heapq
from .models import Station, Track, Train, Block
from .events import Event, EventType
from ..optimizer.solver import solve_conflict_with_ilp, solve_headon_conflict_with_ilp
from ..config import ENABLE_ILP_OPTIMIZER

class RailwayNetwork:
    """The main class that holds the world state and runs the simulation."""
    def __init__(self):
        self.time = 0
        self.stations = {}
        self.tracks = {}
        self.trains = {}
        self.blocks = {}
        self.event_queue = []
        self.conflict_solver_triggered = False
        self.headon_solver_triggered_for_block = set()
        self.express_departed = False # CRITICAL FIX #2a: Flag to prevent re-departure

    def setup_scenario(self):
        print("--- SCENARIO SETUP ---")
        # 1. Create Stations
        self.stations["A"] = Station("A", 0)
        self.stations["B"] = Station("B", 5000) # 5 km
        self.stations["C"] = Station("C", 10000) # 10 km
        self.stations["D"] = Station("D", 15000) # 15 km

        # 2. Create Tracks between stations
        self.tracks["A_to_B_Main"] = Track("A_to_B_Main", 0, 5000)
        self.tracks["B_to_C_Main"] = Track("B_to_C_Main", 5000, 10000)
        self.tracks["C_to_B_Main"] = Track("C_to_B_Main", 10000, 5000) # CRITICAL FIX: Add missing reverse track
        self.tracks["C_to_D_Single"] = Track("C_to_D_Single", 10000, 15000)
        self.tracks["D_to_C_Single"] = Track("D_to_C_Single", 15000, 10000) # Opposite direction
        
        # 3. Create Blocks for single-line sections
        self.blocks["CD_Block"] = Block("CD_Block", ["C_to_D_Single", "D_to_C_Single"])

        # 4. CRITICAL FIX: Add station tracks to the main track dictionary
        for station in self.stations.values():
            self.tracks.update(station.tracks)
        
        print(f"Registered Tracks: {list(self.tracks.keys())}")
        
        # 5. Create Trains
        # A slow goods train with low priority
        self.trains["Goods-456"] = Train("Goods-456", priority=1, max_speed=20) # 20 m/s = 72 km/h
        # A fast express train with high priority
        self.trains["Express-123"] = Train("Express-123", priority=10, max_speed=40) # 40 m/s = 144 km/h
        # A train coming from the opposite direction
        self.trains["Local-789"] = Train("Local-789", priority=5, max_speed=25)

        # 6. Assign initial routes, positions, and states
        goods = self.trains["Goods-456"]
        goods.position = 0
        goods.route = ["A_to_B_Main", "B-Main", "B_to_C_Main", "C-Main", "C_to_D_Single"]
        goods.current_track_index = 0
        goods.state = "RUNNING"
        self.tracks["A_to_B_Main"].occupied_by = goods.id
        # Only set block occupancy if the track is part of a block
        block_name = self.get_block_for_track("A_to_B_Main")
        if block_name:
            self.blocks[block_name].occupied_by = goods.id


        express = self.trains["Express-123"]
        express.position = 0
        express.route = ["A_to_B_Main", "B-Main", "B_to_C_Main"]
        express.current_track_index = 0
        express.state = "HALTED" 

        local = self.trains["Local-789"]
        local.position = 15000
        local.route = ["D_to_C_Single", "C-Main", "C_to_B_Main"]
        local.current_track_index = 0
        local.state = "RUNNING"
        self.tracks["D_to_C_Single"].occupied_by = local.id
        self.blocks["CD_Block"].occupied_by = local.id
        
        print("Scenario: Express-123 (fast) will start behind Goods-456 (slow).")
        print("Conflict expected before Station B. Optimizer must decide if Goods should use the loop track at B.")
        print("New Conflict: Local-789 is heading towards C on a single line, will conflict with Goods-456.")
        print("-" * 20 + "\n")

    def introduce_disruption(self, train_id, new_state="HALTED"):
        """Introduces a disruption for a specific train."""
        if train_id in self.trains:
            self.trains[train_id].state = new_state
            print(f"\n/!\\ DISRUPTION: Train {train_id} has been disrupted and is now in {new_state} state.")
            # Reset solver triggers to allow re-optimization
            self.conflict_solver_triggered = False
            self.headon_solver_triggered_for_block.clear()
            print("/!\\ Re-evaluating network plan...")

    def run_simulation(self):
        """Main simulation loop using an event-driven model."""
        self.setup_scenario()
        # You can seed the event queue with initial events here
        # For now, we'll stick to a tick-based simulation that can generate events.

        try:
            while True:
                # This would be the main loop in a fully event-driven model:
                # if not self.event_queue:
                #     break
                # event = heapq.heappop(self.event_queue)
                # self.time = event.time
                # self.handle_event(event)

                # For now, we'll keep the tick-based advancement for physics
                # and use the event system for high-level logic.
                # CRITICAL FIX #3: Extend simulation time to allow for complex resolutions
                if all(t.state == "HALTED" and t.speed == 0 for t in self.trains.values()) and self.time > 100:
                    print("\n--- SIMULATION COMPLETE ---")
                    break
                
                # --- DISRUPTION TRIGGER (EXAMPLE) ---
                if self.time == 50:
                    self.introduce_disruption("Goods-456")

                self.tick(1) # Advance by 1 second for finer physics
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")

    def handle_event(self, event):
        """Handles events from the event queue."""
        if event.type == EventType.TRAIN_ARRIVAL:
            train_id = event.data["train_id"]
            station_id = event.data["station_id"]
            print(f"EVENT: Train {train_id} arrived at Station {station_id} at t={self.time}s.")
            # Here you would trigger logic for the next leg of the journey
        
        elif event.type == EventType.TRAIN_DEPARTURE:
            pass # Handle departure logic

    def get_block_for_track(self, track_name):
        """Find the block that a given track belongs to."""
        for block in self.blocks.values():
            if track_name in block.track_names:
                return block.name
        return None

    def _update_track_signals(self):
        """Update signals based on track and block occupancy."""
        # Reset all signals to GREEN
        for track in self.tracks.values():
            track.signal = "GREEN"

        # Set signals to RED based on block occupancy
        for block in self.blocks.values():
            if block.occupied_by is not None:
                for track_name in block.track_names:
                    # Set the signal red for all tracks in the occupied block
                    self.tracks[track_name].signal = "RED"

        # Set signal to RED for individually occupied tracks not in a block
        for track in self.tracks.values():
            if track.occupied_by is not None and not self.get_block_for_track(track.name):
                track.signal = "RED"

    def tick(self, delta_time):
        """Main simulation loop tick."""
        self.time += delta_time
        print(f"\n--- TICK: Time = {self.time}s ---")

        # --- Update World State ---
        self._update_track_signals()

        # --- Special Scenario Logic ---
        # Start the express train after a delay
        express = self.trains["Express-123"]
        # CRITICAL FIX #2b: Only depart the express train once.
        if not self.express_departed and self.time >= 20 and express.state == "HALTED":
            express.state = "RUNNING"
            self.express_departed = True
            print(f"** Express-123 has departed from Station A! An overtake will be required. **")

        # --- Conflict Detection & ILP Trigger ---
        # 1. Overtake conflict (original scenario)
        goods = self.trains["Goods-456"]
        express = self.trains["Express-123"]
        
        # FIX: Trigger a re-plan if a high-priority train is obstructed by a lower-priority one,
        # regardless of the lower-priority train's state (running, waiting, or halted/disrupted).
        is_obstructed = False
        if express.state == "RUNNING" and goods.get_current_track(self) is not None:
            goods_track = goods.get_current_track(self)
            express_track = express.get_current_track(self)
            if goods_track and express_track and goods_track.name == express_track.name and express.position < goods.position:
                is_obstructed = True

        if is_obstructed and not self.conflict_solver_triggered:
            self.conflict_solver_triggered = True
            print(f"\n[!!] OVERTAKE CONFLICT DETECTED: {express.id} is approaching {goods.id} (State: {goods.state}).")
            goods.state = "WAITING_FOR_PLAN"
            express.state = "WAITING_FOR_PLAN"
            print(f"[>>] Pausing trains and querying ILP Optimizer for a plan at Station B...")
            
            if ENABLE_ILP_OPTIMIZER:
                plan = solve_conflict_with_ilp(self)
            else:
                print("[--] ILP Optimizer is DISABLED. Using dumb 'first-come, first-served' logic.")
                plan = {
                    "Goods-456": {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]},
                    "Express-123": {"route": ["A_to_B_Main", "B-Main", "B_to_C_Main"]}
                }
            self.apply_plan(plan)
        elif not is_obstructed:
            # The conflict is resolved (e.g., trains are on different tracks), so we can trigger again if needed later.
            self.conflict_solver_triggered = False
        
        # 2. Head-on conflict for single-line blocks
        # FIX: Consider all trains that have a future route, not just those that are currently RUNNING.
        relevant_trains = [t for t in self.trains.values() if t.get_current_track(self) is not None]
        for i in range(len(relevant_trains)):
            for j in range(i + 1, len(relevant_trains)):
                train1 = relevant_trains[i]
                train2 = relevant_trains[j]

                t1_next = train1.get_next_block(self)
                t2_next = train2.get_next_block(self)
                t1_curr = train1.get_current_block(self)
                t2_curr = train2.get_current_block(self)

                # Conflict Case 1: Both trains are heading for the same empty block.
                is_conflict = (t1_next and t1_next == t2_next)
                
                # Conflict Case 2: Train 1 is heading for a block currently occupied by Train 2.
                if not is_conflict:
                    is_conflict = (t1_next and t1_next == t2_curr)

                # Conflict Case 3: Train 2 is heading for a block currently occupied by Train 1.
                if not is_conflict:
                    is_conflict = (t2_next and t2_next == t1_curr)
                
                # If a conflict is found on a single-line block, trigger the solver.
                # (We hardcode "CD_Block" for this scenario, a real system would check block properties)
                block_name = t1_next or t2_next
                if is_conflict and block_name == "CD_Block":
                    if block_name not in self.headon_solver_triggered_for_block:
                        self.headon_solver_triggered_for_block.add(block_name)
                        print(f"\n[!!] HEAD-ON CONFLICT DETECTED: {train1.id} and {train2.id} have conflicting plans for block {block_name}.")
                        train1.state = "WAITING_FOR_PLAN"
                        train2.state = "WAITING_FOR_PLAN"
                        plan = solve_headon_conflict_with_ilp(self, train1.id, train2.id, block_name)
                        self.apply_plan(plan)

        # --- Execute Plan & Advance Trains ---
        for train in self.trains.values():
            # Check for planned waits (wait_until_time)
            if "wait_at_station" in train.plan and train.get_current_track(self).name.startswith(train.plan["wait_at_station"]):
                 if self.time < train.plan["wait_until_time"]:
                     if train.state != "WAITING":
                         print(f"      Train {train.id} is now WAITING at {train.plan['wait_at_station']} as per plan.")
                         train.state = "WAITING"
                 else:
                     print(f"      Train {train.id}'s wait time is over. Resuming journey.")
                     train.state = "RUNNING"
                     train.plan = {} # Clear the wait order
            
            # Check for planned waits (wait_for_block)
            if "wait_for_block" in train.plan:
                block_name = train.plan["wait_for_block"]
                if self.blocks[block_name].occupied_by is None:
                    print(f"      Train {train.id}'s wait for block {block_name} is over. Resuming journey.")
                    train.state = "RUNNING"
                    train.plan = {}
                elif train.state != "WAITING":
                    print(f"      Train {train.id} is WAITING for block {block_name} to be free.")
                    train.state = "WAITING"

            # Simple block logic for the "dumb" scenario (if optimizer is off)
            if not ENABLE_ILP_OPTIMIZER:
                if train.id == "Express-123" and express.get_current_track(self) == goods.get_current_track(self):
                    if express.position > goods.position - 200: # Safety distance
                        if express.speed > goods.speed:
                            print(f"      [DUMB LOGIC] Express-123 is slowing down behind Goods-456.")
                            express.speed = goods.speed

            print(f"   Train {train.id}: Pos={train.position:.0f}m, Speed={train.speed:.1f}m/s, State={train.state}")
            train.advance(self, delta_time)

    def apply_plan(self, plan):
        print(f"[<<] Applying new plan from optimizer...")
        for train_id, instructions in plan.items():
            train = self.trains[train_id]
            if "route" in instructions:
                train.route = instructions["route"]
                print(f"     New route for {train_id}: {' -> '.join(train.route)}")
            
            # Load the plan into the train. The tick loop will handle any waits.
            train.plan = instructions
            if "wait_at_station" in instructions:
                print(f"     Wait order for {train_id}: Wait at {instructions['wait_at_station']} until t={instructions['wait_until_time']:.0f}s")
            if "wait_for_block" in instructions:
                print(f"     Wait order for {train_id}: Wait for block {instructions['wait_for_block']} to be free.")
            
            # CRITICAL FIX: A train must be set to RUNNING to execute its new plan.
            # The waiting logic inside the tick loop will correctly override this if it needs to wait immediately.
            train.state = "RUNNING"
