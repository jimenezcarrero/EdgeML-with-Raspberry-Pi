# Example: Knowledge limitation demonstration
import ollama

response = ollama.generate(
    model="llama3.2:3b",
    prompt="Multiply 123456 by 123456"
)
print(response['response'])
# Output will likely show a wrong result