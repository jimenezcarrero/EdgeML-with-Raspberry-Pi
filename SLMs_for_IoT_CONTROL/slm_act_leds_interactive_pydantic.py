import ollama
import json
from monitor import collect_data, led_status, control_leds
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---

class LEDControl(BaseModel):
    """The required state for the three LEDs."""
    red_led: bool = Field(description="Turn the red LED on (true) or off (false).")
    yellow_led: bool = Field(description="Turn the yellow LED on (true) or off (false).")
    green_led: bool = Field(description="Turn the green LED on (true) or off (false).")

class SLMResponse(BaseModel):
    """The complete response object from the SLM."""
    message: str = Field(description="A helpful text response to the user.")
    leds: LEDControl

# Generate the JSON schema once
RESPONSE_SCHEMA_JSON = json.dumps(SLMResponse.model_json_schema(), indent=2)


# --- Core Functions ---

def create_interactive_prompt(temp_dht, hum, temp_bmp, press, 
    button_state, ledRedSts, ledYlwSts, ledGrnSts, user_input):
	"""Create a concise and structured prompt for the SLM."""
    
	# 1. Concise System Status (Optimized for Prompt Evaluation Speed)
	status_str = (
        f"STATUS: "
        f"DHT_T={temp_dht:.1f}C, DHT_H={hum:.1f}%, "
        f"BMP_T={temp_bmp:.1f}C, BMP_P={press:.2f}hPa, "
        f"BTN={'1' if button_state else '0'}, "
        f"LED_R={'1' if ledRedSts else '0'}, "
        f"LED_Y={'1' if ledYlwSts else '0'}, "
        f"LED_G={'1' if ledGrnSts else '0'}"
    )

	return f"""
You are an IoT assistant.
{status_str}

USER REQUEST: "{user_input}"

INSTRUCTIONS:
Analyze the request and respond with a JSON object that strictly follows the provided JSON Schema.
- If the user asks for info, keep the current LED states (R={ledRedSts}, Y={ledYlwSts}, G={ledGrnSts}).
- Evaluate conditional commands (e.g., 'if temp > 20C') using the STATUS data.
- Only ONE LED should be on UNLESS the user explicitly asks for multiple.

REQUIRED JSON SCHEMA:
---
{RESPONSE_SCHEMA_JSON}
---

Respond with ONLY the JSON, no other text.
"""


def slm_inference(PROMPT, MODEL):
    response = ollama.generate(
    	model=MODEL, 
    	prompt=PROMPT
    	)
    return response


def parse_interactive_response(response_text):
    """Parse the interactive SLM JSON response using Pydantic for validation."""
    try:
        # --- Robust JSON Extraction ---
        # 1. Strip whitespace and non-essential text markers
        cleaned_text = response_text.strip()
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
        
        # 2. Find the start of the JSON object (first '{')
        start_index = cleaned_text.find('{')
        if start_index == -1:
            raise ValueError("No starting JSON object brace '{' found.")

        # 3. Find the end of the JSON object (last '}')
        # This handles cases where the model might stop mid-generation or add text after the JSON.
        end_index = cleaned_text.rfind('}')
        if end_index == -1 or end_index < start_index:
            raise ValueError("No ending JSON object brace '}' found.")
        
        # 4. Slice the clean JSON string
        json_str = cleaned_text[start_index : end_index + 1]

        # --- Pydantic Validation ---
        
        # Use Pydantic to parse and validate the JSON
        # This is fast and enforces the strict schema.
        response_data = SLMResponse.model_validate_json(json_str)
        
        # Extract data from the validated Pydantic object
        message = response_data.message
        red_led = response_data.leds.red_led
        yellow_led = response_data.leds.yellow_led
        green_led = response_data.leds.green_led
        
        return message, (red_led, yellow_led, green_led)
    
    except Exception as e:
        # Catch all errors (Pydantic, ValueErrors, etc.) and report
        print(f"Error parsing/validating SLM response: {type(e).__name__}: {e}")
        print(f"Response was (Raw): {response_text}")
        try:
             # Try to print the extracted string for debugging
             print(f"Response was (Extracted): {json_str}")
        except:
             pass
        return "Error: Could not parse or validate SLM response format.", (False, False, False)


def display_system_status(temp_dht, hum, temp_bmp, press, button_state, ledRedSts, ledYlwSts, ledGrnSts):
    """Display comprehensive system status."""
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)
    print(f"DHT22 Sensor:  Temp = {temp_dht:.1f}°C, Humidity = {hum:.1f}%")
    print(f"BMP280 Sensor: Temp = {temp_bmp:.1f}°C, Pressure = {press:.2f}hPa")
    print(f"Button:        {'PRESSED' if button_state else 'NOT PRESSED'}")
    print(f"\nLED Status:")
    print(f"  Red LED:    {'●' if ledRedSts else '○'} {'ON' if ledRedSts else 'OFF'}")
    print(f"  Yellow LED: {'●' if ledYlwSts else '○'} {'ON' if ledYlwSts else 'OFF'}")
    print(f"  Green LED:  {'●' if ledGrnSts else '○'} {'ON' if ledGrnSts else 'OFF'}")
    print("="*60)


def interactive_mode(MODEL):
    """Run the system in interactive mode accepting user commands."""
    print("\n" + "="*60)
    print("IoT Environmental Monitoring System - Interactive Mode")
    print(f"Using Model: {MODEL}")
    print("="*60)
    print("\nCommands you can try:")
    print("  - What's the current temperature?")
    print("  - Turn on the yellow LED")
    print("  - If temperature is above 20°C, turn on yellow LED")
    print("  - If button is pressed, turn on red LED")
    print("  - Type 'status' to see system status")
    print("  - Type 'exit' or 'quit' to stop")
    print("="*60 + "\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', 'q']:
            # Ensure LEDs are off on exit
            control_leds(False, False, False)
            print("\nExiting interactive mode. Goodbye!")
            break
        
        # Get current system status
        ledRedSts, ledYlwSts, ledGrnSts = led_status()
        temp_dht, hum, temp_bmp, press, button_state = collect_data()
        
        # Handle status command
        if user_input.lower() == 'status':
            display_system_status(temp_dht, hum, temp_bmp, press, button_state, 
                                ledRedSts, ledYlwSts, ledGrnSts)
            continue
        
        # Check if sensor data is valid
        if any(v is None for v in [temp_dht, hum, temp_bmp, press]):
            print("Assistant: Error - Unable to read sensor data. Please try again.")
            continue
        
        # Create prompt with user input
        PROMPT = create_interactive_prompt(temp_dht, hum, temp_bmp, press, button_state,
                                          ledRedSts, ledYlwSts, ledGrnSts, user_input)
        
        # Get SLM response
        print("Assistant: [Thinking...]")
        response = slm_inference(PROMPT, MODEL)
        
        # Parse response
        message, (red, yellow, green) = parse_interactive_response(response['response'])
        
        # Display assistant's message
        print(f"Assistant: {message}")
        
        # Control LEDs based on response
        # Ensure that only one LED is turned on unless all are explicitly commanded
        if red and yellow and green:
            control_leds(True, True, True)
        elif sum([red, yellow, green]) > 1:
            # If the model tried to turn on multiple without explicit command,
            # this is an error, so turn all off or keep previous state.
            print("Warning: Model requested multiple LEDs ON without explicit command. Ignoring LED changes.")
        else:
            control_leds(red, yellow, green)
        
        # Display updated system status
        ledRedSts, ledYlwSts, ledGrnSts = led_status()
        print(f"\nLED Update: Red={'ON' if ledRedSts else 'OFF'}, "
              f"Yellow={'ON' if ledYlwSts else 'OFF'}, "
              f"Green={'ON' if ledGrnSts else 'OFF'}\n")


if __name__ == "__main__":
    MODEL = 'llama3.2:3b'  # Use 3b model for better instruction following
    
    # Run in interactive mode
    interactive_mode(MODEL)