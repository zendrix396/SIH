import time
from .core.network import RailwayNetwork
from .config import TICK_DURATION_SECS, SIMULATION_SPEED_MULTIPLIER

# --- Main Execution ---
if __name__ == "__main__":
    network = RailwayNetwork()
    network.run_simulation()
    # network.setup_scenario()

    # try:
    #     while True:
    #         # End condition: when all trains have finished
    #         if all(t.state == "HALTED" for t in network.trains.values()) and network.time > 100:
    #             print("\n--- SIMULATION COMPLETE ---")
    #             break
            
    #         network.tick(TICK_DURATION_SECS)
    #         time.sleep(TICK_DURATION_SECS / SIMULATION_SPEED_MULTIPLIER)
    # except KeyboardInterrupt:
    #     print("\nSimulation stopped by user.")
