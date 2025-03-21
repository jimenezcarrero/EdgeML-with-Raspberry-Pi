import ollama
import json

def validate_response(query, response):
    """Validate that the response is appropriate for the query"""
    validation_prompt = f"""
    User query: {query}
    Generated response: {response}
    
    Evaluate if this response:
    1. Directly addresses the user's query
    2. Is factually accurate to the best of your knowledge
    3. Is helpful and complete
    
    Respond in the following JSON format:
    {{
        "valid": true/false,
        "reason": "Explanation if invalid",
        "score": 0-10
    }}
    """
    
    try:
        validation = ollama.generate(
            model="llama3.2:3b",  
            prompt=validation_prompt
        )
        
        result = json.loads(validation['response'])
        return result
    except Exception as e:
        print(f"Error during validation: {e}")
        return {"valid": False, "reason": "Validation error", "score": 0}

# Test
query = "What is the Raspberry Pi 5?"
response = "It is a pie created with raspberry and cooked in an oven"
validation = validate_response(query, response)
print(validation)
