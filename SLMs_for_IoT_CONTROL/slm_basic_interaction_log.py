# slm_basic_interaction_log.py
import ollama
from monitor import collect_data, led_status, control_leds
import monitor_log as mlog
import time
from datetime import datetime
from threading import Thread

# Available models
MODELS = {
    1: ('deepseek-r1:1.5b', 'DeepSeek R1 1.5B'),
    2: ('llama3.2:1b', 'Llama 3.2 1B'),
    3: ('llama3.2:3b', 'Llama 3.2 3B'),
    4: ('phi3:latest', 'Phi-3'),
    5: ('gemma:2b', 'Gemma 2B'),
}

def parse_llm_response(response_text):
    """Parse the LLM response to extract LED control instructions."""
    response_lower = response_text.lower()
    red_led = 'activate red led' in response_lower
    yellow_led = 'activate yellow led' in response_lower
    green_led = 'activate green led' in response_lower
    return (red_led, yellow_led, green_led)

def get_user_input():
    """Get user input for model selection and temperature threshold"""
    print("\nAvailable Models:")
    for num, (_, name) in MODELS.items():
        print(f"{num}. {name}")
    
    while True:
        try:
            model_num = int(input("\nSelect model (1-5): "))
            if model_num in MODELS:
                break
            print("Please select a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            temp_threshold = float(input("Enter temperature threshold (°C): "))
            break
        except ValueError:
            print("Please enter a valid number for temperature threshold.")
    
    return MODELS[model_num][0], MODELS[model_num][1], temp_threshold

def query_log(query, model):
    """Query the log data using SLM with improved context and structure"""
    try:
        # Get log summary from monitor_log
        log_summary = mlog.get_log_summary()
        
        # Create a more structured prompt with explicit instructions
        prompt = f"""
        You are an IoT system analyst examining sensor data logs. Here is the current analysis:

        {log_summary}

        The user asks: "{query}"

        Follow these guidelines to answer:
        1. For temperature trends:
           - Look at the temperature trend values (both DHT22 and BMP280)
           - Report if temperatures are increasing, decreasing, or stable
           - Include the rate of change in °C per interval
           
        2. For button or LED history:
           - Report the total number of state changes
           - Mention if activity is high, moderate, or low
           
        3. For specific measurements:
           - Use the most recent values
           - Include both sensor readings when relevant
           - Report the average if asked
           
        4. For general analysis:
           - Focus on notable patterns
           - Highlight any unusual values
           - Compare different sensor readings if relevant

        Provide a clear, concise response focusing ONLY on the requested information.
        """

        # Get SLM response
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        
        return response['response'].strip()

    except Exception as e:
        return f"Error querying log: {e}"

def process_command(model, temp_threshold, user_input):
    """Process a single user command with improved log query detection"""
    try:
        # Enhanced log query detection
        log_keywords = [
            'trend', 'history', 'past', 'record', 'log',
            'average', 'changes', 'times', 'pattern',
            'statistics', 'stats', 'summary'
        ]
        
        # Check if this is a log query
        if any(keyword in user_input.lower() for keyword in log_keywords):
            response = query_log(user_input, model)
            
            # Print status with log query results
            print("\n" + "="*50)
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"Log Query: {user_input}")
            print(f"Analysis: {response}")
            print("="*50 + "\n")
            return response

        # Rest of the existing command processing code...
        sensors = collect_data()
        if any(v is None for v in sensors):
            return "Error: Failed to read sensor data"

        temp_dht, hum, temp_bmp, press, button_state = sensors

        # Original prompt for real-time commands
        prompt = f"""
            You are monitoring an IoT system which is showing the following sensor status: 
            - DHT22 Temp: {temp_dht:.1f}°C and Humidity: {hum:.1f}%
            - BMP280 Temp: {temp_bmp:.1f}°C and Pressure: {press:.2f}hPa
            - Button {"pressed" if button_state else "not pressed"}

            The user command is: "{user_input}"

            Based on the command:
            1. For sensor queries, provide only the relevant current readings
            2. For LED control, respond only with:
               "Activate Red LED" or "Activate Yellow LED" or "Activate Green LED"
            3. If temperature is above {temp_threshold}°C, mention it
        """

        # Get SLM response and process as before...
        # [Rest of the existing code remains the same]

    except Exception as e:
        return f"Error occurred: {str(e)}"
        

def main():
    try:
        # Setup logging
        mlog.setup_log_file()
        
        # Start automatic logging thread
        logging_thread = Thread(target=mlog.automatic_logging, daemon=True)
        logging_thread.start()

        # Get initial user input
        model, model_name, temp_threshold = get_user_input()

        print(f"\nStarting IoT control system with {model_name}")
        print(f"Temperature threshold: {temp_threshold}°C")
        print("Type 'quit' to exit")
        print("\nYou can:")
        print("- Control LEDs (e.g., 'turn on red led')")
        print("- Query sensors (e.g., 'what's the temperature?')")
        print("- Query logs (e.g., 'show temperature trend', 'led history')\n")

        while True:
            user_input = input("Command: ").strip().lower()
            
            if user_input == 'quit':
                print("\nShutting down...")
                mlog.stop_logging.set()  # Signal logging thread to stop
                control_leds(False, False, False)
                break
            
            # Process single command
            process_command(model, temp_threshold, user_input)

    except KeyboardInterrupt:
        print("\nShutting down...")
        mlog.stop_logging.set()
        control_leds(False, False, False)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        mlog.stop_logging.set()
        control_leds(False, False, False)

if __name__ == "__main__":
    main()