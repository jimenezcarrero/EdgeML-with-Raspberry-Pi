import requests
import json
import re
import time
from datetime import datetime
from tavily import TavilyClient

# Configuration
OLLAMA_URL = "http://localhost:11434/api"
MODEL = "llama3.2:3b"  # Main model for answering
CLASSIFICATION_MODEL = "llama3.2:3b"  # Model for classification
TAVILY_API_KEY = "tvly-YOUR_API_KEY"  # Replace with your actual API key
VERBOSE = True

# Keep a persistent session to reuse connections
session = requests.Session()

def ask_ollama_for_classification(user_input):
    """
    Ask Ollama to classify whether the query is:
    1. Something it can answer from its knowledge
    2. Something that requires recent information from the web
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    classification_prompt = f"""
    Today's date is {current_date} and the current year is {current_year}.
    
    Analyze if the following query requires current or recent information that would be OUTSIDE 
    your training data, or if it's about general knowledge that doesn't change over time.
    
    Query: "{user_input}"
    
    Respond with JSON only:
    {{
      "type": "general_knowledge" or "needs_web_search",
      "reason": "brief explanation of why this is general knowledge or needs web search",
      "search_query": "optimized search terms for web search" if needs_web_search
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
                "keep_alive": "5m"
            },
            timeout=30
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
                    return json.loads(json_str)
                
                # Fallback
                return {"type": "needs_web_search", "reason": "Failed to parse model output", "search_query": user_input}
            except json.JSONDecodeError:
                if VERBOSE:
                    print(f"Failed to parse JSON: {response_text}")
                
                # Default to web search when in doubt
                return {
                    "type": "needs_web_search",
                    "reason": "JSON parsing error - defaulting to web search",
                    "search_query": user_input
                }
        else:
            if VERBOSE:
                print(f"Error: Received status code {response.status_code} from Ollama.")
            return {"type": "needs_web_search", "reason": "API error", "search_query": user_input}
    
    except Exception as e:
        if VERBOSE:
            print(f"Error in classification: {str(e)}")
        return {"type": "needs_web_search", "reason": f"Error: {str(e)}", "search_query": user_input}

def search_tavily(query):
    """
    Search the web using Tavily API which is designed for RAG applications.
    Returns formatted search results.
    """
    try:
        if VERBOSE:
            print(f"Searching Tavily for: {query}")
        
        # Initialize the Tavily client
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Search with Tavily
        response = tavily_client.search(
            query=query,
            search_depth="basic",  # Use 'basic' for faster, more focused results
            max_results=3,  # Limit to 3 results to avoid overloading the context
            include_answer=True,  # Include an AI-generated answer
        )
        
        if VERBOSE:
            print(f"Tavily search successful, found {len(response.get('results', []))} results")
        
        # Format the results
        formatted_results = "Search Results:\n\n"
        
        # First add the Tavily-generated answer if available
        if response.get('answer'):
            formatted_results += f"Summary: {response['answer']}\n\n"
        
        # Then add a limited number of search results
        for i, result in enumerate(response.get('results', [])[:3], 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content available')
            if len(content) > 300:  # Limit long content
                content = content[:300] + "..."
            
            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   {content}\n\n"
        
        if not response.get('results'):
            return "No search results found for this query."
        
        return formatted_results
    
    except Exception as e:
        if VERBOSE:
            print(f"Error during Tavily search: {str(e)}")
        
        # Check if it's an authentication error
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            return "Tavily API authentication failed. Please check your API key."
        
        return f"Error during web search: {str(e)}. Using built-in knowledge instead."

def ask_ollama(query, context=""):
    """
    Send a query to Ollama for answering with improved timeout and streaming.
    """
    try:
        if VERBOSE:
            print(f"Sending query to Ollama using {MODEL}")
        
        prompt = query
        if context:
            # Create a more focused prompt
            prompt = f"""
Here is some recent information related to your question:

{context}

Based on this information, please provide a concise answer to: {query}

If the information doesn't provide a clear answer, please state what you know about the topic.
"""
        
        # Use shorter prompts for general knowledge questions
        # This helps reduce processing time
        if not context:
            prompt = f"Please provide a brief, concise answer to: {query}"
        
        # CRITICAL FIX: Use streaming for general knowledge queries
        # This prevents timeouts by starting to process the output immediately
        stream = not bool(context)  # Stream for general knowledge, don't stream for search results
        
        if stream:
            # Streaming response
            if VERBOSE:
                print("Using streaming response")
            
            response = session.post(
                f"{OLLAMA_URL}/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "5m",
                    "max_tokens": 150  # Shorter response for general knowledge
                },
                stream=True,
                timeout=45
            )
            
            if response.status_code == 200:
                # Process the streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                full_response += json_line['response']
                        except json.JSONDecodeError:
                            continue
                return full_response
            else:
                return f"Error: Received status code {response.status_code} from Ollama."
                
        else:
            # Non-streaming for context-based answers
            response = session.post(
                f"{OLLAMA_URL}/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "5m",
                    "max_tokens": 300  # Longer response for search results
                },
                timeout=45  # Increased timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: Received status code {response.status_code} from Ollama."
    
    except requests.exceptions.Timeout:
        # Specific handling for timeout errors
        if context:
            # If we have context but Ollama timed out, return the context summary directly
            summary_match = re.search(r"Summary: (.*?)(?:\n\n|\Z)", context, re.DOTALL)
            if summary_match:
                return f"Based on search results: {summary_match.group(1)}"
            
            return "The search found information, but the language model timed out while processing it."
        else:
            return "The language model timed out while processing your query. Please try a simpler question or try again later."
    
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def process_query(user_input):
    """
    Process the user input by first classifying whether it needs web search,
    then either searching the web or using the model's knowledge directly.
    """
    start_time = time.time()
    
    # Classify the query
    classification = ask_ollama_for_classification(user_input)
    
    if VERBOSE:
        print("Classification:", classification)
    
    if classification.get("type") == "needs_web_search":
        # Use the provided search query or fall back to the original query
        search_query = classification.get("search_query", user_input)
        
        if VERBOSE:
            print(f"Searching web for: {search_query}")
        
        # Get search results using Tavily
        search_results = search_tavily(search_query)
        
        # Let the model answer based on search results
        answer = ask_ollama(user_input, context=search_results)
        
        elapsed_time = time.time() - start_time
        
        # Format the response to show it used web search
        return {
            "answer": answer,
            "source": "web_search",
            "search_query": search_query,
            "time": elapsed_time
        }
    else:
        # Use the model's knowledge directly
        answer = ask_ollama(user_input)
        
        elapsed_time = time.time() - start_time
        
        return {
            "answer": answer,
            "source": "model_knowledge",
            "reason": classification.get("reason", "Query classified as general knowledge"),
            "time": elapsed_time
        }

def initialize_models():
    """
    Initialize both models to keep them loaded in memory.
    This prevents the cold start problem.
    """
    print(f"Initializing models: {MODEL} and {CLASSIFICATION_MODEL}")
    
    try:
        # Use simple prompts to warm up the models
        session.post(
            f"{OLLAMA_URL}/generate",
            json={"model": MODEL, "prompt": "Hello", "stream": False, "keep_alive": "5m"}
        )
        
        session.post(
            f"{OLLAMA_URL}/generate", 
            json={"model": CLASSIFICATION_MODEL, "prompt": "Hello", "stream": False, "keep_alive": "5m"}
        )
        
        print("Models initialized and ready")
    except Exception as e:
        print(f"Warning: Model initialization failed. Error: {str(e)}")

def main():
    """
    Main function to run the knowledge agent interactively.
    """
    global MODEL, CLASSIFICATION_MODEL

    print(f"Knowledge Router Agent with Tavily Search")
    print(f"Main model: {MODEL}, Classification model: {CLASSIFICATION_MODEL}")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
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
        
        result = process_query(user_input)
        
        # Format the response based on source
        if result["source"] == "web_search":
            print(f"\nAgent (via web search): {result['answer']}")
            print(f"\nSearch query: {result['search_query']}")
        else:
            print(f"\nAgent (from knowledge): {result['answer']}")
            
        print(f"\nTime elapsed: {result['time']:.2f} seconds")
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Run interactive mode
    main()