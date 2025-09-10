# AI Railway Traffic Controller

This project is a simulation of a railway network where a Reinforcement Learning agent is trained to manage traffic flow, minimize delays, and maximize throughput. The agent is built using the Stable Baselines3 library and the environment is created with Pygame for visualization.

### Setup Instructions
```bash
pip install -r requirements.txt
```

### How to Run
To run the simulation, use the `agent.py` script. You can switch between training the agent and running a demo with the trained model.

- **To train the agent:**
  In `agent.py`, set `TRAIN_MODE = True`. Then run:
  ```bash
  python agent.py
  ```
  The trained model will be saved as `ppo_throughput_agent.zip`.

- **To run the demo:**
  In `agent.py`, set `TRAIN_MODE = False`. Then run:
  ```bash
  python agent.py
  ```
  This will load the pre-trained `ppo_throughput_agent.zip` model and run the simulation with a visual interface.

### Tweakable Commands and Configuration

The primary way to configure the simulation is by modifying the `src/config.py` file. Here are some of the key parameters you can tweak:

- **Simulation Settings:**
  - `SIM_STEP_SECONDS`: How many seconds each simulation step represents.
  - `MAX_EPISODE_STEPS`: The maximum number of steps before an episode ends.
  - `DISRUPTION_INTERVAL_SECONDS`: How often to introduce random delays to trains.

- **Visualization:**
  - `SCREEN_WIDTH`, `SCREEN_HEIGHT`: The dimensions of the simulation window.
  - `FPS`: The frame rate for the visualization.
  - Various `COLOR_*` settings for the UI elements.

- **Railway Network:**
  - `NODES`: The coordinates of stations, junctions, and other points in the network.
  - `EDGES`: The connections between nodes that form the tracks.
  - `PATHS`: The predefined routes that trains can take through the network.

- **Train Physics:**
  - `SPEED_KPH`: The speed for different types of trains (Express, Passenger, Freight).
  - `ACCELERATION`, `DECELERATION`: The acceleration and deceleration rates for trains.

### File Usage

- `agent.py`: The main script for running the simulation. It handles both training the RL agent and running demos with a trained model.
- `ppo_throughput_agent.zip`: The saved, pre-trained PPO agent model.
- `requirements.txt`: A list of all the Python packages required to run the project.
- `copy_to_clipboard.py`: A utility script to copy the entire codebase to the clipboard, useful for sharing or debugging.
- `railway_tensorboard/`: This directory contains the logs generated during the training process, which can be viewed with TensorBoard to monitor the agent's performance.
- `src/`: This directory contains the core source code for the simulation environment.
  - `environment.py`: Defines the `RailwayEnv` class, which is the main simulation environment. It follows the Gymnasium (formerly OpenAI Gym) interface and manages the state of the railway network, trains, signals, and the reward logic.
  - `config.py`: A centralized file for all the simulation parameters. This is where you can tweak the network layout, train characteristics, and other settings.
  - `renderer.py`: Handles all the visual aspects of the simulation using Pygame. It draws the railway network, trains, and UI elements.
  - `path_finder.py`: A utility for interpolating paths between nodes to create smooth curves for trains to follow.
  - `console_logger.py`: Prints detailed debug information to the console during the simulation, showing the agent's decisions and the status of the trains.

### Future Revamps (TODO)

Here are some areas for future improvement and development:

- **UI/Renderer (`renderer.py`)**
  - Implement more interactive controls like pausing, stepping through the simulation, and changing the simulation speed in real-time.
  - Fixing current bugs and making the animation seamless.
  - Add the ability to click on trains and signals to get more detailed information about their status.
  - Improve the overall UI design to be more informative and user-friendly.

- **RL Backend (`agent.py`, `environment.py`)**
  - Make the current PPO-based reinforcement learning implementation better.
  - Enhance the observation space to provide the agent with more detailed information about the environment (e.g., more granular track occupancy).
  - Refine the reward function to encourage more complex and efficient behaviors from the agent.

- **Simulation Logic (`environment.py`, `config.py`)**
  - Introduce more complex and realistic disruption scenarios, such as track maintenance, signal failures, or unexpected train arrivals.
  - Implement a more dynamic event generation system instead of a fixed schedule.
  - Improve the train physics to more accurately model real-world train behavior.
