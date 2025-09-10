import os
import time
from numpy import False_
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.environment import RailwayEnv
from src.console_logger import print_debug_info

def main():
    # <<<<<<<<<<<<<<< CHOOSE MODE HERE >>>>>>>>>>>>>>>>>
    TRAIN_MODE = False # Set to True to train, False to run demo
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    MODEL_PATH = "ppo_throughput_agent.zip"
    
    if TRAIN_MODE:
        print("--- Starting AI Agent Training ---")
        env = RailwayEnv()
        # check_env(env) # Very useful to run once when you change your environment

        # --- REVISED: Check for existing model to resume training ---
        if os.path.exists(MODEL_PATH):
            print(f"--- Resuming training from saved model: {MODEL_PATH} ---")
            model = PPO.load(MODEL_PATH, env=env, device='cpu')
        else:
            print(f"--- No saved model found. Creating a new one. ---")
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./railway_tensorboard/",device='cpu')
        
        print("Training... Press Ctrl+C to stop and save.")
        try:
            # --- REVISED: Added reset_num_timesteps=False for proper resuming ---
            model.learn(total_timesteps=500000, reset_num_timesteps=False)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        
        model.save(MODEL_PATH)
        print(f"--- Training complete. Model saved to {MODEL_PATH} ---")
        env.close()

    else: # DEMO_MODE
        print("--- Starting AI Controller Demo ---")
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at '{MODEL_PATH}'")
            print("Please run the script in TRAIN_MODE first to train and save the agent.")
            return

        env = RailwayEnv(render_mode="human")
            
        model = PPO.load(MODEL_PATH, device='cpu')
        
        print("Trained model loaded successfully.")
        
        obs, _ = env.reset()
        running = True
        while running:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            if env.renderer:
                running = env.renderer.handle_events()

            print_debug_info(env)
            
            time.sleep(0.1)

            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs, _ = env.reset()
        
        env.close()

if __name__ == "__main__":
    main()