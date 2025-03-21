#!/usr/bin/env python
# coding: utf-8

import requests
import json
import re
import time
import threading
import concurrent.futures

# Configuration
OLLAMA_URL = "http://localhost:11434/api"
MODEL = "llama3.2:3b"  # Main model for answering
CLASSIFICATION_MODEL = "llama3.2:3b"  # Model for classification
VERBOSE = True

# Keep a persistent session to reuse connections
session = requests.Session()

def ask_ollama_for_classification(user_input):
    """
    Ask Ollama to classify whether the query is a calculation request.
    Using a more efficient, simpler prompt and potentially a smaller model.
    """
    # Simplified classification prompt
    classification_prompt = f"""
    Is this a calculation request? "{user_input}"
    
    Respond with JSON only:
    {{
      "type": "calculation" or "general_question",
      "operation": "add|subtract|multiply|divide" if calculation,
      "numbers": [number1, number2] if calculation
    }}
    """
    
    try:
        if VERBOSE:
            print(f"Sending classification request using {CLASSIFICATION_MODEL}")
        
        # Use the session for connection reuse
        response = session.post(
            f"{OLLAMA_URL}/generate",
            json={
                "model": CLASSIFICATION_MODEL,
                "prompt": classification_prompt,
                "stream": False,
                # Add to prevent model reloading
                "keep_alive": "5m"
            }
        )
        
        if response.status_code == 200:
            response_text = response.json().get("response", "").strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON content if there's any surrounding text
                start_index = response_text.find('{')
                end_index = response_text.rfind('}') + 1
                if start_index >= 0 and end_index > start_index:
                    json_str = response_text[start_index:end_index]
                    classification = json.loads(json_str)
                    
                    # If model failed to include numbers or operation
                    if classification.get("type") == "calculation":
                        # Extract numbers from query if needed
                        if not classification.get("numbers"):
                            numbers = extract_numbers_from_query(user_input)
                            if numbers and len(numbers) >= 2:
                                classification["numbers"] = numbers[:2]
                        # Ensure numbers are floats, not strings
                        elif classification.get("numbers"):
                            classification["numbers"] = [float(n) if isinstance(n, str) else n 
                                                        for n in classification["numbers"]]
                        
                        # Determine operation if needed
                        if not classification.get("operation"):
                            classification["operation"] = determine_operation(user_input)
                    
                    return classification
                return {"type": "general_question"}
            except json.JSONDecodeError:
                if VERBOSE:
                    print(f"Failed to parse JSON: {response_text}")
                
                # Fallback to rule-based classification
                numbers = extract_numbers_from_query(user_input)
                if len(numbers) >= 2 and has_calculation_keywords(user_input):
                    return {
                        "type": "calculation",
                        "operation": determine_operation(user_input),
                        "numbers": [float(n) if isinstance(n, str) else n for n in numbers[:2]]
                    }
                return {"type": "general_question"}
        else:
            if VERBOSE:
                print(f"Error: Received status code {response.status_code} from Ollama.")
            return {"type": "general_question"}
    
    except Exception as e:
        if VERBOSE:
            print(f"Error connecting to Ollama: {str(e)}")
        return {"type": "general_question"}

def has_calculation_keywords(query):
    """Check if the query contains calculation keywords"""
    query = query.lower()
    calc_words = ["add", "plus", "+", "subtract", "minus", "-", 
                 "multiply", "times", "*", "×", "divide", "/", "÷"]
    return any(word in query for word in calc_words)

def extract_numbers_from_query(query):
    """
    Extract numbers from the query string.
    """
    # Look for floating point or integer numbers
    numbers = re.findall(r'(\d+\.?\d*)', query)
    return [float(num) for num in numbers]

def determine_operation(query):
    """
    Determine the arithmetic operation based on keywords in the query.
    """
    query = query.lower()
    
    if any(word in query for word in ["add", "addition", "plus", "sum", "+"]):
        return "add"
    elif any(word in query for word in ["subtract", "subtraction", "minus", "difference", "-"]):
        return "subtract"
    elif any(word in query for word in ["multiply", "multiplication", "times", "product", "*", "×"]):
        return "multiply"
    elif any(word in query for word in ["divide", "division", "/", "÷"]):
        return "divide"
    else:
        # Default to addition if unclear
        return "add"

def calculate(operation, a, b):
    """
    Perform the specified calculation and return a formatted response with comma separators.
    """
    def format_number(num):
        """Format a number with comma separators for thousands"""
        if isinstance(num, int):
            # For integers
            return f"{num:,}"
        else:
            # For floats: format with appropriate decimal places
            # Handle different decimal precision based on the value
            if abs(num) < 0.01:
                # Scientific notation for very small numbers
                return f"{num:.10g}"
            elif abs(num) < 1:
                return f"{num:,.6f}".rstrip('0').rstrip('.') if '.' in f"{num:,.6f}" else f"{num:,}"
            elif abs(num) < 1000:
                return f"{num:,.4f}".rstrip('0').rstrip('.') if '.' in f"{num:,.4f}" else f"{num:,}"
            else:
                return f"{num:,.2f}".rstrip('0').rstrip('.') if '.' in f"{num:,.2f}" else f"{num:,}"
    
    if operation == "add":
        result = a + b
        return f"The sum of {format_number(a)} and {format_number(b)} is {format_number(result)}."
    elif operation == "subtract":
        result = a - b
        return f"The difference between {format_number(a)} and {format_number(b)} is {format_number(result)}."
    elif operation == "multiply":
        result = a * b
        return f"The product of {format_number(a)} and {format_number(b)} is {format_number(result)}."
    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero."
        result = a / b
        return f"The result of dividing {format_number(a)} by {format_number(b)} is {format_number(result)}."
    else:
        return "Unsupported operation."

def ask_ollama(query):
    """
    Send a query to Ollama for general question answering.
    """
    try:
        if VERBOSE:
            print(f"Sending query to Ollama using {MODEL}")
        
        # Use the session for connection reuse
        response = session.post(
            f"{OLLAMA_URL}/generate",
            json={
                "model": MODEL,
                "prompt": query,
                "stream": False,
                # Add to prevent model reloading
                "keep_alive": "5m"
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: Received status code {response.status_code} from Ollama."
    
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def process_query(user_input):
    """
    Process the user input by first asking Ollama to classify it,
    then either performing a calculation or sending it back as a general question.
    """
    # Classify the query
    classification = ask_ollama_for_classification(user_input)
    
    if VERBOSE:
        print("Classification:", classification)
    
    if classification.get("type") == "calculation":
        operation = classification.get("operation", "add")
        numbers = classification.get("numbers", [0, 0])
        if len(numbers) >= 2:
            # Convert numbers to float if they're strings
            a = float(numbers[0]) if isinstance(numbers[0], str) else numbers[0]
            b = float(numbers[1]) if isinstance(numbers[1], str) else numbers[1]
            return calculate(operation, a, b)
        else:
            return "I understood you wanted a calculation, but couldn't extract the numbers properly."
    else:
        return ask_ollama(user_input)

def initialize_models():
    """
    Initialize both models to keep them loaded in memory.
    This prevents the cold start problem.
    """
    print(f"Initializing models: {MODEL} and {CLASSIFICATION_MODEL}")
    
    # Start both model initializations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            session.post,
            f"{OLLAMA_URL}/generate",
            json={"model": MODEL, "prompt": "Hello", "stream": False, "keep_alive": "5m"}
        )
        
        future2 = executor.submit(
            session.post,
            f"{OLLAMA_URL}/generate", 
            json={"model": CLASSIFICATION_MODEL, "prompt": "Hello", "stream": False, "keep_alive": "5m"}
        )
        
        # Wait for both to complete
        concurrent.futures.wait([future1, future2])
    
    print("Models initialized and ready")

def main():
    """
    Main function to run the calculator agent interactively.
    """
    global MODEL, CLASSIFICATION_MODEL

    print(f"Optimized Ollama Calculator Agent")
    print(f"Main model: {MODEL}, Classification model: {CLASSIFICATION_MODEL}")
    print("Type 'exit' to quit, 'model <name>' to change main model, or 'classmodel <name>' to change classification model")
    print("-" * 50)
    
    # Initialize models at startup
    initialize_models()
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Check if user wants to change the models
        if user_input.lower().startswith("model "):
            MODEL = user_input[6:].strip()
            print(f"Main model changed to: {MODEL}")
            continue
        
        if user_input.lower().startswith("classmodel "):
            CLASSIFICATION_MODEL = user_input[11:].strip()
            print(f"Classification model changed to: {CLASSIFICATION_MODEL}")
            continue
        
        start_time = time.time()
        response = process_query(user_input)
        elapsed_time = time.time() - start_time
        
        print(f"\nAgent: {response}")
        print(f"\nTime elapsed: {elapsed_time:.2f} seconds")
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Set to True to see detailed logging
    VERBOSE = True
    
    # Run interactive mode
    main()