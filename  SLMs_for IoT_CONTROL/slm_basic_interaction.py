import ollama
from monitor import collect_data, led_status, control_leds
import time

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

def process_command(model, temp_threshold, user_input):
    """Process a single user command"""
    try:
        # Collect sensor data
        temp_dht, hum, temp_bmp, press, button_state = collect_data()
        
        if any(v is None for v in [temp_dht, hum, temp_bmp, press]):
            return "Error: Failed to read sensor data"

        prompt = f"""
            You are monitoring an IoT system which is showing the following sensor status: 
            - DHT22 Temp: {temp_dht:.1f}°C and Humidity: {hum:.1f}%
            - BMP280 Temp: {temp_bmp:.1f}°C and Pressure: {press:.2f}hPa
            - Button {"pressed" if button_state else "not pressed"}

            The user command is: "{user_input}"

            You should:
            1. Understand what the user wants
            2. If it's a question about sensor data, provide ONLY the relevant information. 
               Be concise and stop. 
            3. If it's a command to control LEDs, you should provide a concise answer only with: 
            "Activate Red LED" or "Activate Yellow LED" or "Activate Green LED"
            4. If temperature is above {temp_threshold}°C, mention it in your response.
        """

        # Get SLM response
        response = ollama.generate(
            model=model,
            prompt=prompt
        )

        # Parse response and control LEDs
        red, yellow, green = parse_llm_response(response['response'])
        control_leds(red, yellow, green)
        
        # Print status
        print("\n" + "="*50)
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"DHT22: {temp_dht:.1f}°C, {hum:.1f}%")
        print(f"BMP280: {temp_bmp:.1f}°C, {press:.1f}hPa")
        print(f"Button: {'pressed' if button_state else 'not pressed'}")
        print(f"SLM Response: {response['response'].strip()}")
        print(f"LED Status: R={'ON' if red else 'off'}, " 
              f"Y={'ON' if yellow else 'off'}, "
              f"G={'ON' if green else 'off'}")
        print("="*50 + "\n")

    except Exception as e:
        return f"Error occurred: {str(e)}"

def main():
    try:
        # Get initial user input
        model, model_name, temp_threshold = get_user_input()

        print(f"\nStarting IoT control system with {model_name}")
        print(f"Temperature threshold: {temp_threshold}°C")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("Command: ").strip().lower()
            
            if user_input == 'quit':
                print("\nShutting down...")
                control_leds(False, False, False)  # Turn off all LEDs
                break
            
            # Process single command
            process_command(model, temp_threshold, user_input)

    except KeyboardInterrupt:
        print("\nShutting down...")
        control_leds(False, False, False)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        control_leds(False, False, False)  # Ensure LEDs are off on exit

if __name__ == "__main__":
    main()