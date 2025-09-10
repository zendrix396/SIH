import numpy as np
from src.path_finder import interpolate_path

# --- CONFIGURATION CONSTANTS ---
# Use a dictionary for clean organization
CONFIG_DATA = {
    # Simulation Settings
    "SIM_STEP_SECONDS": 10,  # Each simulation step represents 10 seconds
    "MAX_EPISODE_STEPS": 2500, # Max steps before an episode ends
    "DISRUPTION_INTERVAL_SECONDS": 1800, # Introduce a delay every 30 sim minutes

    # Screen & Visualization Settings
    "SCREEN_WIDTH": 1600,
    "SCREEN_HEIGHT": 900,
    "FPS": 60, # Control the visualization speed

    # Colors
    "COLOR_BACKGROUND": (20, 20, 40),
    "COLOR_TRACK": (100, 100, 120),
    "COLOR_STATION": (200, 200, 200),
    "COLOR_SIGNAL_RED": (255, 50, 50),
    "COLOR_SIGNAL_GREEN": (50, 255, 50),
    "COLOR_TEXT": (240, 240, 240),
    "COLOR_TRAIN_EXPRESS": (50, 150, 255),
    "COLOR_TRAIN_PASSENGER": (50, 255, 150),
    "COLOR_TRAIN_FREIGHT": (255, 150, 50),

    # --- NEW: EXPANDED Railway Network Layout ---
    "NODES": {
        # Original Diamond
        "N_STATION": (800, 50), "S_STATION": (800, 850),
        "E_STATION": (1550, 450), "W_STATION": (50, 450),
        "N_ENTRY": (800, 250), "S_ENTRY": (800, 650),
        "E_ENTRY": (1250, 450), "W_ENTRY": (350, 450),
        "J1": (600, 350), "J2": (1000, 350),
        "J3": (600, 550), "J4": (1000, 550),

        # West Expansion
        "W_HUB": (-400, 450),
        "W_SID": (-400, 650), # Siding/Yard
        "W_J5": (-200, 450),

        # East Expansion (Industrial loop)
        "E_HUB": (2000, 450),
        "E_J6": (1750, 450),
        "E_IND1": (1750, 250),
        "E_IND2": (2000, 250),

        # North Spur
        "N_J7": (800, -150),
        "N_TERM": (600, -150) # Terminus
    },
    "EDGES": [
        # Original Diamond
        ("N_STATION", "N_ENTRY"), ("S_STATION", "S_ENTRY"),
        ("E_STATION", "E_ENTRY"), ("W_STATION", "W_ENTRY"),
        ("N_ENTRY", "J2"), ("N_ENTRY", "J1"), ("W_ENTRY", "J1"), 
        ("W_ENTRY", "J3"), ("J1", "J3"), ("J2", "J4"),
        ("J3", "S_ENTRY"), ("J4", "S_ENTRY"), ("J2", "E_ENTRY"),
        ("J4", "E_ENTRY"),

        # West Expansion
        ("W_STATION", "W_J5"),
        ("W_J5", "W_HUB"),
        ("W_J5", "W_SID"),

        # East Expansion
        ("E_STATION", "E_J6"),
        ("E_J6", "E_HUB"),
        ("E_J6", "E_IND1"),
        ("E_IND1", "E_IND2"),

        # North Spur
        ("N_STATION", "N_J7"),
        ("N_J7", "N_TERM")
    ],
    "PATHS": {
        # Original Paths
        "N_S": ["N_STATION", "N_ENTRY", "J2", "J4", "S_ENTRY", "S_STATION"],
        "S_N": ["S_STATION", "S_ENTRY", "J3", "J1", "N_ENTRY", "N_STATION"],
        "E_W": ["E_STATION", "E_ENTRY", "J4", "J3", "W_ENTRY", "W_STATION"],
        "W_E": ["W_STATION", "W_ENTRY", "J1", "J2", "E_ENTRY", "E_STATION"],

        # --- FIX: Re-add all necessary paths for the schedule ---
        "W_HUB_S": ["W_HUB", "W_J5", "W_STATION", "W_ENTRY", "J3", "S_ENTRY", "S_STATION"],
        "E_HUB_N": ["E_HUB", "E_J6", "E_STATION", "E_ENTRY", "J2", "N_ENTRY", "N_STATION"],
        "N_TERM_E_IND": ["N_TERM", "N_J7", "N_STATION", "N_ENTRY", "J2", "E_ENTRY", "E_STATION", "E_J6", "E_IND1", "E_IND2"],
        "S_W_SID": ["S_STATION", "S_ENTRY", "J3", "W_ENTRY", "W_STATION", "W_J5", "W_SID"],

        # --- NEW: Alternative "Scenic" Routes ---
        "N_S_ALT": ["N_STATION", "N_ENTRY", "J1", "J3", "S_ENTRY", "S_STATION"],
        "S_N_ALT": ["S_STATION", "S_ENTRY", "J4", "J2", "N_ENTRY", "N_STATION"],
        "E_W_ALT": ["E_STATION", "E_ENTRY", "J2", "J1", "W_ENTRY", "W_STATION"],
        "W_E_ALT": ["W_STATION", "W_ENTRY", "J3", "J4", "E_ENTRY", "E_STATION"],
    },
    # Mapping from primary paths to their alternatives
    "ALT_ROUTE_MAP": {
        "N_S": "N_S_ALT", "S_N": "S_N_ALT",
        "E_W": "E_W_ALT", "W_E": "W_E_ALT",
    },
    # Signals control entry into the main junction and new hubs
    "SIGNAL_NODES": ["N_ENTRY", "S_ENTRY", "E_ENTRY", "W_ENTRY", "W_J5", "E_J6", "N_J7"],

    # Train Physics
    "SPEED_KPH": {"EXPRESS": 120, "PASSENGER": 90, "FREIGHT": 60},
    "PRIORITY": {"EXPRESS": 3, "PASSENGER": 2, "FREIGHT": 1},
    
    # Baseline for KPI calculation
    "BASELINE_THROUGHPUT": 5.0, # Estimated trains/hour for a simple system

    # --- NEW: Train Physics ---
    "ACCELERATION": 0.5, # m/s^2
    "DECELERATION": 1.0, # m/s^2
    "POINTS_PER_SEGMENT": 30 # Resolution for path interpolation
}

# --- DYNAMICALLY GENERATE HIGH-RESOLUTION PATHS ---
# This converts the simple node-to-node paths into detailed lists of coordinates
# that the trains can follow smoothly.
CONFIG_DATA["PATHS_HIGH_RES"] = {
    name: interpolate_path(CONFIG_DATA["NODES"], path, CONFIG_DATA["POINTS_PER_SEGMENT"])
    for name, path in CONFIG_DATA["PATHS"].items()
}

# Final immutable config
CONFIG = CONFIG_DATA
