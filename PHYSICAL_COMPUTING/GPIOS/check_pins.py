import RPi.GPIO as GPIO

def check_all_pins():
    # Using BCM numbering
    GPIO.setmode(GPIO.BCM)
    
    # Standard GPIO pins on Raspberry Pi (adjust this list based on your Pi model)
    gpio_pins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    
    pins_in_use = []
    available_pins = []
    
    for pin in gpio_pins:
        try:
            GPIO.setup(pin, GPIO.IN)
            # If setup succeeds, the pin is available
            GPIO.cleanup(pin)
            available_pins.append(pin)
        except:
            # If we get an error, the pin is in use
            pins_in_use.append(pin)
    
    # Clean up all GPIO settings
    GPIO.cleanup()
    
    return pins_in_use, available_pins

def print_pin_status():
    in_use, available = check_all_pins()
    
    print("\nPins currently in use:")
    if in_use:
        for pin in sorted(in_use):
            print(f"GPIO {pin} (BCM)")
    else:
        print("None")
        
    print("\nAvailable pins:")
    if available:
        for pin in sorted(available):
            print(f"GPIO {pin} (BCM)")
    else:
        print("None")
        
    print("\nNote: Some pins might be reserved for special functions (I2C, SPI, UART)")
    print("Common reserved pins:")
    print("- GPIO 0, 1: Reserved for ID EEPROM")
    print("- GPIO 2, 3: Reserved for I2C1 (SDA, SCL)")
    print("- GPIO 14, 15: Reserved for UART (TXD, RXD)")
    print("- GPIO 8, 9, 10, 11: Reserved for SPI0")

if __name__ == "__main__":
    try:
        print_pin_status()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Note: This script requires root privileges to run")
        print("Try running with: sudo python3 check_gpio.py")
