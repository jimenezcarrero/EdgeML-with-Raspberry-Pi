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
            model_num = int(input("\nSelect model (1-4): "))
            if model_num in MODELS:
                break
            print("Please select a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            temp_threshold = float(input("Enter temperature threshold (°C): "))
            break
        except ValueError:
            print("Please enter a valid number for temperature threshold.")
    
    return MODELS[model_num][0], MODELS[model_num][1], temp_threshold

def monitor_system(model, model_name, temp_threshold):

    """Monitor system continuously"""
    print(f"\nStarting monitoring with {model_name}")
    print(f"Temperature threshold: {temp_threshold}°C")
    print("Press Ctrl+C to stop monitoring\n")

    while True:
        try:
            # Collect sensor data
            temp_dht, hum, temp_bmp, press, button_state = collect_data()
            
            if any(v is None for v in [temp_dht, hum, temp_bmp, press]):
                print("Error: Failed to read sensor data")
                time.sleep(2)
                continue

            prompt = f"""

                You are monitoring an IoT system which is showing the following sensor status: 
                - DHT22 Temp: {temp_dht:.1f}°C and Humidity: {hum:.1f}%
                - BMP280 Temp: {temp_bmp:.1f}°C and Pressure: {press:.2f}hPa
                - Button {"pressed" if button_state else "not pressed"}

                Based on the Rules: 
                - If system is working in normal conditions → Activate Green LED 
                - If DHT22 Temp or BMP280 Temp are greater than Temperature 
                Threshold ({temp_threshold}°C) → Activate Yellow LED 
                - If Button pressed, it is an emergency → Activate Red LED

                You should provide a brief answer only with: "Activate Red LED" or "Activate Yellow LED" 
                or "Activate Green LED"

            """


            # Format prompt with current data
            current_prompt = prompt.format(
                temp_dht=temp_dht,
                hum=hum,
                temp_bmp=temp_bmp,
                press=press,
                button_state="pressed" if button_state else "not pressed"
            )

            # Get SLM response
            response = ollama.generate(
                model=model,
                prompt=current_prompt
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
            
            time.sleep(2)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            control_leds(False, False, False)  # Turn off all LEDs
            break
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            time.sleep(2)

def main():
    # Get initial user input
    model, model_name, temp_threshold = get_user_input()
    
    # Start continuous monitoring
    monitor_system(model, model_name, temp_threshold)

if __name__ == "__main__":
    main()