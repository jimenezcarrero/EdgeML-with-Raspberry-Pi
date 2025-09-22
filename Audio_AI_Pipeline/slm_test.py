import ollama

def generate_voice_response(user_input, model="llama3.2:3b"):
    """
    Generate a response optimized for voice interaction
    """
    
	 # Context-setting prompt that guides the model's behavior
    system_context = """
    	You are a helpful AI assistant designed for voice interactions. 
    	Your responses will be converted to speech and spoken aloud to the user.
    
      Guidelines for your responses:
      - Keep responses conversational and concise (ideally under 50 words)
      - Avoid complex formatting, lists, or visual elements
      - Speak naturally, as if having a friendly conversation
      - If the user's input seems unclear, ask for clarification politely
      - Provide direct answers rather than lengthy explanations unless specifically 
        requested
    """
    
    # Combine system context with user input
    full_prompt = f"{system_context}\n\nUser said: {user_input}\n\nResponse:"
    
    response = ollama.generate(
        model=model,
        prompt=full_prompt
    )
    
    return response['response']


# Answering the user question: 
user_input = "What is the capital of Malawi?"
response = generate_voice_response(user_input)
print (response)