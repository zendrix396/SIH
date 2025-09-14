# File: generate_briefing.py

import os
import json
import google.generativeai as genai
from pathlib import Path

def generate_gemini_briefing(scenario_data, result_data):
    """
    Uses the Gemini API to generate a detailed, analytical briefing for a section controller.
    """
    print("\n--- Contacting Gemini API to generate controller briefing ---")
    
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        client = genai.GenerativeClient()
    except Exception as e:
        print(f"\nError configuring Gemini client: {e}")
        return None

    # Construct the new, more detailed prompt
    prompt = f"""
    You are an expert AI assistant for railway operations, specializing in translating complex optimization data into clear, actionable instructions for a human Section Controller. Your analysis must be deep, insightful, and operationally focused.

    **CONTEXT:**
    An optimization model has generated an optimal schedule for a complex traffic scenario. I will provide you with the initial scenario data (the problem) and the final optimized result data (the solution) in JSON format.

    **SCENARIO DATA (THE PROBLEM):**
    ```json
    {json.dumps(scenario_data, indent=2)}
    ```

    **OPTIMIZATION RESULT DATA (THE SOLUTION):**
    ```json
    {json.dumps(result_data, indent=2)}
    ```

    **YOUR TASK:**
    Based on all the data above, generate a professional briefing for the Section Controller. The briefing must be in natural language and easy to understand. Structure your response into the following three detailed sections:

    1.  **OVERALL SITUATIONAL ANALYSIS:**
        Start with a high-level summary. What is the state of the network for this period? What was the primary operational challenge the schedule had to solve (e.g., "a critical overtake conflict between a slow goods train and a high-priority express on the single-track section after Station_02")? What was the overall strategic outcome of the optimized plan (e.g., "The plan protects the schedule of the high-priority express by executing a strategic loop maneuver, while absorbing the delay on the lower-priority goods service.")?

    2.  **KEY STRATEGIC ACTIONS & ORDERS:**
        This is the most critical section. List the specific, non-obvious actions the controller needs to take to execute this plan. This should be a clear list of orders. For each action, specify the train, the precise action (hold, loop), the location, the timing, and the strategic reason. Be precise.
        *Example: "ORDER 1: Route Train_Slow_Goods (Priority 1) onto the loop line at Station_03. The train is predicted to arrive at minute 150.0 and must be held there until at least minute 165.0. REASON: This action is essential to clear the main line for the overtaking of the critical, high-priority Train_Fast_Express, which is trailing closely behind."*
        *Example: "ORDER 2: Hold Train_02 (Priority 3) at origin (Station_00) for an additional 15 minutes, dispatching at minute 35.0 instead of 20.0. REASON: This proactive hold prevents it from interfering with an earlier, higher-priority traffic cluster, reducing overall network delay."*

    3.  **DETAILED TRAIN PERFORMANCE & DELAY CAUSATION:**
        Provide a status update for all trains that experienced a final delay of more than 2 minutes. For each delayed train, perform a brief "delay causation analysis". Explain *why* it was delayed, specifying where the delay occurred and which other train(s) caused it.
        *Example: "Train_07 (Priority 3) - FINAL DELAY: 25 minutes. ANALYSIS: This train incurred its delay primarily on the single-track segment between Station_05 and Station_06, where it was forced to trail the slower Train_04. The schedule correctly decided not to delay the higher-priority Train_04, so Train_07 absorbed the cascading headway delay."*
        *Example: "Train_Slow_Goods (Priority 1) - FINAL DELAY: 40 minutes. ANALYSIS: This train's delay was a result of two planned strategic actions: [1] It was held on the loop at Station_03 for 15 minutes to allow Train_Fast_Express to overtake. [2] It was then held at Station_05 for another 10 minutes to allow Train_08 to pass. This protected the punctuality of two higher-priority services."*
    """

    try:
        response = client.generate_content(
            model="models/gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"\nAn error occurred during the API call: {e}")
        return None

if __name__ == "__main__":
    input_path = Path("input_scenario.json")
    result_path = Path("optimization_result.json")

    if not input_path.exists() or not result_path.exists():
        print(f"Error: Make sure both '{input_path}' and '{result_path}' exist. Please run the previous scripts first.")
    else:
        with open(input_path, 'r') as f:
            scenario = json.load(f)
        with open(result_path, 'r') as f:
            results = json.load(f)
            
        briefing = generate_gemini_briefing(scenario, results)

        if briefing:
            print("\n==========================================================")
            print("          GEMINI-POWERED CONTROLLER BRIEFING")
            print("==========================================================")
            print(briefing)