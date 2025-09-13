# --- Core Simulation Classes ---

class Track:
    """Represents a piece of track connecting two points."""
    def __init__(self, name, start_pos, end_pos):
        self.name = name
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.length = abs(end_pos - start_pos)
        self.occupied_by = None  # Stores the ID of the train on this track
        self.signal = "GREEN" # GREEN, RED

    def __repr__(self):
        return f"Track({self.name}, {self.start_pos}-{self.end_pos}, Signal: {self.signal})"

class Block:
    """Represents a section of track that can only be occupied by one train at a time."""
    def __init__(self, name, track_names):
        self.name = name
        self.track_names = track_names
        self.occupied_by = None

    def __repr__(self):
        return f"Block({self.name}, Tracks: {self.track_names}, Occupied: {self.occupied_by})"

class Station:
    """Represents a station with main and loop tracks."""
    def __init__(self, name, position):
        self.name = name
        self.position = position
        # Stations have a "zero-length" track for the platform itself
        self.tracks = {
            f"{name}-Main": Track(f"{name}-Main", position, position),
            f"{name}-Loop": Track(f"{name}-Loop", position, position)
        }
    
    def __repr__(self):
        return f"Station({self.name} at pos {self.position})"

class Train:
    """Represents a train with its physical properties and state."""
    def __init__(self, train_id, priority, max_speed=30, acceleration=0.2, deceleration=1.0): # speed in m/s, accel in m/s^2
        self.id = train_id
        self.priority = priority  # Higher number = higher priority
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.deceleration = deceleration
        
        # Dynamic State
        self.position = 0.0
        self.speed = 0.0
        self.state = "HALTED"  # HALTED, RUNNING, WAITING_FOR_PLAN
        self.route = []         # A list of track names to follow
        self.current_track_index = -1
        
        # The plan received from the optimizer
        self.plan = {}

    def get_current_track(self, network):
        if 0 <= self.current_track_index < len(self.route):
            track_name = self.route[self.current_track_index]
            return network.tracks[track_name]
        return None

    def get_current_block(self, network):
        """Gets the block the train is currently in."""
        current_track = self.get_current_track(network)
        if current_track:
            return network.get_block_for_track(current_track.name)
        return None

    def get_next_block(self, network):
        """Looks ahead in the route to find the next block."""
        if self.current_track_index + 1 < len(self.route):
            next_track_name = self.route[self.current_track_index + 1]
            return network.get_block_for_track(next_track_name)
        return None

    def advance(self, network, delta_time):
        current_track = self.get_current_track(network)

        # CRITICAL FIX #1: If route is finished, the ONLY action is to decelerate to a stop.
        # No other logic (movement, track switching, etc.) should execute.
        if not current_track:
            self.state = "HALTED"
            if self.speed > 0:
                self.speed -= self.deceleration * delta_time
                self.speed = max(0, self.speed)
            
            # This message will only print once the train has fully stopped at its final position.
            if self.speed == 0 and "has_finished_message" not in self.plan:
                 print(f"      Train {self.id} has completed its route and is halted at {self.position:.0f}m.")
                 self.plan["has_finished_message"] = True # Ensure message prints only once
            return # Absolutely no more processing or movement

        # --- If route is NOT finished, proceed with normal logic ---

        # Update speed based on state
        if self.state == "RUNNING":
            if self.speed < self.max_speed:
                self.speed += self.acceleration * delta_time
                self.speed = min(self.speed, self.max_speed) # Clamp to max speed
        else: # HALTED, WAITING, etc.
            if self.speed > 0:
                self.speed -= self.deceleration * delta_time
                self.speed = max(0, self.speed) # Clamp to 0
        
        # --- Signal/Block Logic ---
        # Look ahead to the *next* track in the route
        distance_to_end = current_track.end_pos - self.position
        stopping_distance = (self.speed ** 2) / (2 * self.deceleration)

        if self.current_track_index + 1 < len(self.route):
            next_track_name = self.route[self.current_track_index + 1]
            next_track = network.tracks[next_track_name]
            if next_track.signal == "RED":
                if distance_to_end <= stopping_distance:
                    # Must start braking now!
                    self.state = "WAITING" 
                    print(f"      Train {self.id} sees RED signal for {next_track.name}. Braking!")

        # Move the train
        distance_to_move = self.speed * delta_time
        self.position += distance_to_move
        
        # Check if we've reached the end of the current track
        # This logic needs to account for direction of travel
        direction = 1 if current_track.end_pos > current_track.start_pos else -1
        reached_end = False
        if direction == 1 and self.position >= current_track.end_pos:
            reached_end = True
        elif direction == -1 and self.position <= current_track.end_pos:
            reached_end = True

        if reached_end:
            print(f"      Train {self.id} reached end of {current_track.name} at {current_track.end_pos:.0f}m.")
            
            # Free the old track and its block
            current_track.occupied_by = None
            old_block_name = network.get_block_for_track(current_track.name)
            if old_block_name and network.blocks[old_block_name].occupied_by == self.id:
                network.blocks[old_block_name].occupied_by = None
                print(f"      Train {self.id} has cleared block {old_block_name}.")

            # Move to the next track
            self.current_track_index += 1
            next_track = self.get_current_track(network)
            if next_track:
                # Determine position and direction on the new track
                if next_track.start_pos == current_track.end_pos:
                    self.position = next_track.start_pos
                else: # Assuming train is reversing at a station or entering a reverse track
                    self.position = next_track.end_pos

                # Occupy the new track and its block
                next_track.occupied_by = self.id
                new_block_name = network.get_block_for_track(next_track.name)
                if new_block_name:
                    network.blocks[new_block_name].occupied_by = self.id
                
                print(f"      Train {self.id} entered next track {next_track.name}.")
            else:
                self.state = "HALTED" # End of route
