# monitor_log.py
import csv
import os
from datetime import datetime
import pandas as pd
from threading import Event, Thread
import time
from monitor import collect_data, led_status

# Global variables for logging
LOG_FILE = 'system_log.csv'
stop_logging = Event()

def setup_log_file():
    """Create or verify log file with headers"""
    headers = ['timestamp', 'temp_dht', 'humidity', 'temp_bmp', 'pressure', 
              'button_state', 'led_red', 'led_yellow', 'led_green', 'command']
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def log_data(timestamp, sensors, leds, command=""):
    """Log system data to CSV file"""
    temp_dht, hum, temp_bmp, press, button = sensors
    red, yellow, green = leds
    
    row = [
        timestamp,
        f"{temp_dht:.1f}" if temp_dht is not None else "NA",
        f"{hum:.1f}" if hum is not None else "NA",
        f"{temp_bmp:.1f}" if temp_bmp is not None else "NA",
        f"{press:.1f}" if press is not None else "NA",
        "1" if button else "0",
        "1" if red else "0",
        "1" if yellow else "0",
        "1" if green else "0",
        command
    ]
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def automatic_logging():
    """Background thread for automatic logging every minute"""
    while not stop_logging.is_set():
        try:
            sensors = collect_data()
            leds = led_status()
            if any(v is None for v in sensors):
                continue
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_data(timestamp, sensors, leds)
            
            # Wait for one minute or until stop signal
            stop_logging.wait(60)
            
        except Exception as e:
            print(f"Logging error: {e}")
            time.sleep(60)

def count_state_changes(series):
    """Count actual state changes in a binary series"""
    # Convert to int and get the changes
    series = series.astype(int)
    changes = 0
    last_state = series.iloc[0]
    
    # Count each time the state changes
    for state in series[1:]:
        if state != last_state:
            changes += 1
            last_state = state
            
    return changes

def calculate_trend(series):
    """Calculate trend with improved statistical analysis"""
    # Convert to numeric and handle NaN values
    series = pd.to_numeric(series, errors='coerce')
    series = series.dropna()
    
    if len(series) < 2:
        return 0.0, "insufficient data"
        
    # Calculate simple moving average to smooth noise
    window = min(5, len(series))
    smoothed = series.rolling(window=window, center=True).mean()
    
    # Calculate overall trend
    total_change = smoothed.iloc[-1] - smoothed.iloc[0]
    time_periods = len(smoothed) - 1
    
    if time_periods == 0:
        return 0.0, "stable"
        
    trend_per_period = total_change / time_periods
    
    # Determine trend direction
    if abs(trend_per_period) < 0.1:  # Threshold for "stable"
        direction = "stable"
    else:
        direction = "increasing" if trend_per_period > 0 else "decreasing"
        
    return trend_per_period, direction

def analyze_log_data():
    """Analyze log data and return statistics"""
    try:
        df = pd.read_csv(LOG_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns
        numeric_columns = ['temp_dht', 'humidity', 'temp_bmp', 'pressure']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate trends using rolling mean to smooth out noise
        window = min(5, len(df))
        if len(df) >= 2:  # Need at least 2 points for trend
            temp_dht_trend = df['temp_dht'].diff().mean()
            temp_bmp_trend = df['temp_bmp'].diff().mean()
            humidity_trend = df['humidity'].diff().mean()
            pressure_trend = df['pressure'].diff().mean()
        else:
            temp_dht_trend = temp_bmp_trend = humidity_trend = pressure_trend = 0.0

        # Calculate statistics
        stats = {
            'temp_dht_trend': temp_dht_trend if not pd.isna(temp_dht_trend) else 0.0,
            'temp_bmp_trend': temp_bmp_trend if not pd.isna(temp_bmp_trend) else 0.0,
            'humidity_trend': humidity_trend if not pd.isna(humidity_trend) else 0.0,
            'pressure_trend': pressure_trend if not pd.isna(pressure_trend) else 0.0,
            'avg_temp_dht': df['temp_dht'].mean(),
            'avg_humidity': df['humidity'].mean(),
            'avg_temp_bmp': df['temp_bmp'].mean(),
            'avg_pressure': df['pressure'].mean(),
            'led_red_changes': count_state_changes(df['led_red']),
            'led_yellow_changes': count_state_changes(df['led_yellow']),
            'led_green_changes': count_state_changes(df['led_green']),
            'button_changes': count_state_changes(df['button_state']),
            'recent_data': df.tail(1)
        }
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing log data: {e}")
        return None

def get_log_summary():
    """Get a formatted summary of log data for SLM prompts"""
    stats = analyze_log_data()
    if not stats:
        return "Error: Unable to analyze log data"
    
    # Add trend direction indicators
    def get_trend_indicator(value):
        if abs(value) < 0.01:  # threshold for "stable"
            return "stable"
        return "increasing" if value > 0 else "decreasing"

    temp_dht_direction = get_trend_indicator(stats['temp_dht_trend'])
    temp_bmp_direction = get_trend_indicator(stats['temp_bmp_trend'])
    humidity_direction = get_trend_indicator(stats['humidity_trend'])
    pressure_direction = get_trend_indicator(stats['pressure_trend'])
    
    summary = f"""
    Recent Statistics:
    - Temperature (DHT22): {stats['temp_dht_trend']:.3f}°C per interval ({temp_dht_direction})
    - Temperature (BMP280): {stats['temp_bmp_trend']:.3f}°C per interval ({temp_bmp_direction})
    - Humidity: {stats['humidity_trend']:.3f}% per interval ({humidity_direction})
    - Pressure: {stats['pressure_trend']:.3f}hPa per interval ({pressure_direction})
    
    Averages:
    - Average Temperature (DHT22): {stats['avg_temp_dht']:.1f}°C
    - Average Temperature (BMP280): {stats['avg_temp_bmp']:.1f}°C
    - Average Humidity: {stats['avg_humidity']:.1f}%
    - Average Pressure: {stats['avg_pressure']:.1f}hPa

    LED and Button Activity (transitions):
    - Red LED changes: {stats['led_red_changes']}
    - Yellow LED changes: {stats['led_yellow_changes']}
    - Green LED changes: {stats['led_green_changes']}
    - Button presses: {stats['button_changes']}

    Most recent values:
    {stats['recent_data'][['timestamp', 'temp_dht', 'humidity', 'temp_bmp', 'pressure']].to_string()}
    """
    
    return summary

if __name__ == "__main__":
    # Setup the log file if it doesn't exist
    setup_log_file()
    
    # Start the automatic logging in a separate thread
    logging_thread = Thread(target=automatic_logging, daemon=True)
    logging_thread.start()
    
    print("Starting log summary test (Press Ctrl+C to stop)")
    
    try:
        while True:
            summary = get_log_summary()
            print("\nLog Summary:")
            print(summary)
            print("="*50)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping test...")
        stop_logging.set()
    finally:
        stop_logging.set()