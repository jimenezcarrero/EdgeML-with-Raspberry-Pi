#!/usr/bin/env python
# coding: utf-8

# Advanced Agentic RAG System
# Extends the basic RAG with agent capabilities and optimizations

import time
import warnings
import os
import json
import re
from typing import Dict, List, Any, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Suppress LangSmith warnings
warnings.filterwarnings("ignore", 
                        message="API key must be provided when using hosted LangSmith API",
                        category=UserWarning)

# Configuration
PERSIST_DIRECTORY = "chroma_db"
MAIN_MODEL = "llama3.2:3b"  # Main model for answering
ROUTER_MODEL = "llama3.2:3b"  # Model for routing decisions
VERBOSE = True

def load_retriever():
    """Load the vector store from disk and create a retriever"""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Database directory {PERSIST_DIRECTORY} not found. Please run create-database.py first.")
    
    if VERBOSE:
        print("Loading existing vector store...")
    
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name="rag-edgeai-eng-chroma",
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Create retriever with more relevant results
    retriever = vectorstore.as_retriever(k=4)
    return retriever

def get_documents(retriever, query: str) -> List[str]:
    """Retrieve relevant documents for a query"""
    start_time = time.time()
    docs = retriever.invoke(query)
    docs_content = [doc.page_content for doc in docs]
    
    if VERBOSE:
        print(f"Retrieved {len(docs)} document chunks in {time.time() - start_time:.2f}s")
    
    return docs_content

def route_query(query: str) -> Dict[str, Any]:
    """Determine if the query is a calculation, otherwise use RAG"""
    if VERBOSE:
        print(f"Routing query: {query}")
    
    # Check for calculation keywords
    calc_terms = ["+", "add", "plus", "sum", "-", "subtract", "minus", "difference", 
                "*", "×", "multiply", "times", "product", "/", "÷", "divide", "division", "quotient"]
    
    # Simple rule-based detection for calculations
    is_calc = any(term in query.lower() for term in calc_terms) and re.search(r'\d+', query)
    
    if is_calc:
        # Use smaller, faster model for operation and number extraction
        llm = ChatOllama(model=ROUTER_MODEL, temperature=0)
        
        router_prompt = f"""
        For this calculation query: "{query}"
        
        Extract the calculation details in JSON format:
        {{
            "operation": "add", "subtract", "multiply", or "divide",
            "numbers": [number1, number2]
        }}
        
        JSON response only:
        """
        
        try:
            result = llm.invoke(router_prompt)
            result_text = result.content
            
            # Extract JSON from the result
            start_index = result_text.find('{')
            end_index = result_text.rfind('}') + 1
            if start_index >= 0 and end_index > start_index:
                json_str = result_text[start_index:end_index]
                route_info = json.loads(json_str)
                route_info["type"] = "calculation"
                
                # Validate and add fallback for numbers
                if not route_info.get("numbers") or len(route_info.get("numbers", [])) < 2:
                    # Fallback to regex for number extraction if LLM failed
                    numbers = re.findall(r'(\d+\.?\d*)', query)
                    if len(numbers) >= 2:
                        route_info["numbers"] = [float(num) for num in numbers[:2]]
                
                # Determine operation if missing
                if not route_info.get("operation"):
                    if any(term in query.lower() for term in ["+", "add", "plus", "sum"]):
                        route_info["operation"] = "add"
                    elif any(term in query.lower() for term in ["-", "subtract", "minus", "difference"]):
                        route_info["operation"] = "subtract"
                    elif any(term in query.lower() for term in ["*", "×", "multiply", "times", "product"]):
                        route_info["operation"] = "multiply"
                    elif any(term in query.lower() for term in ["/", "÷", "divide", "division", "quotient"]):
                        route_info["operation"] = "divide"
                
                return route_info
            
            # Fallback to RAG if JSON parsing fails
            return {"type": "rag", "reasoning": "Parsing error, defaulting to RAG"}
        
        except Exception as e:
            if VERBOSE:
                print(f"Error in routing: {str(e)}")
            return {"type": "rag", "reasoning": f"Routing error: {str(e)}"}
    
    # For everything else, use RAG
    return {"type": "rag", "reasoning": "Non-calculation query, using RAG"}

def perform_calculation(operation: str, numbers: List[float]) -> str:
    """Execute a calculation based on the operation and numbers"""
    if len(numbers) < 2:
        return "Insufficient numbers provided for calculation."
    
    a, b = numbers[0], numbers[1]
    
    def format_number(num):
        """Format a number with comma separators for thousands"""
        if isinstance(num, int):
            return f"{num:,}"
        else:
            # For floats: handle different precision based on size
            if abs(num) < 0.01:
                return f"{num:.10g}"
            elif abs(num) < 1:
                return f"{num:,.6f}".rstrip('0').rstrip('.') if '.' in f"{num:,.6f}" else f"{num:,}"
            elif abs(num) < 1000:
                return f"{num:,.4f}".rstrip('0').rstrip('.') if '.' in f"{num:,.4f}" else f"{num:,}"
            else:
                return f"{num:,.2f}".rstrip('0').rstrip('.') if '.' in f"{num:,.2f}" else f"{num:,}"
    
    try:
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
            return f"Unsupported operation: {operation}"
    except Exception as e:
        return f"Error performing calculation: {str(e)}"

def validate_response(llm, query: str, response: str) -> Dict[str, Any]:
    """Validate the response quality using a dedicated validation prompt"""
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
        validation_result = llm.invoke(validation_prompt)
        
        # Extract JSON from the response
        validation_text = validation_result.content
        start_index = validation_text.find('{')
        end_index = validation_text.rfind('}') + 1
        
        if start_index >= 0 and end_index > start_index:
            json_str = validation_text[start_index:end_index]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                if VERBOSE:
                    print(f"JSON decode error: {e}")
                # Fallback score
                return {"valid": True, "reason": "JSON parsing failed", "score": 5}
        
        return {"valid": True, "reason": "Failed to extract JSON", "score": 5}
    
    except Exception as e:
        if VERBOSE:
            print(f"Validation error: {str(e)}")
        return {"valid": True, "reason": f"Validation error: {str(e)}", "score": 5}

def get_answer_with_rag(query: str, context: List[str], llm) -> str:
    """Generate an answer using the RAG system with context documents"""
    # Join the context with newlines for better readability
    context_text = "\n\n".join(context)
    
    # Create the prompt with context and query
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    # Generate the answer
    answer = rag_chain.invoke({"context": context_text, "question": query})
    
    return answer

# This function is no longer used since we're always using RAG for non-calculation queries
def get_answer_general(query: str, llm) -> str:
    """Generate an answer using the model's general knowledge"""
    # Create a direct prompt that encourages concise answers
    direct_prompt = f"""
    Please provide a brief, helpful answer to this question:
    
    {query}
    
    Answer concisely but completely.
    """
    
    # Generate the answer
    result = llm.invoke(direct_prompt)
    
    return result.content

def process_query(query: str, retriever) -> Dict[str, Any]:
    """Process a query using the appropriate method based on routing"""
    start_time = time.time()
    
    # Initialize the LLM
    llm = ChatOllama(model=MAIN_MODEL, temperature=0)
    
    # Route the query
    route_info = route_query(query)
    query_type = route_info.get("type", "rag")
    
    if VERBOSE:
        print(f"Query type: {query_type}")
        print(f"Routing reasoning: {route_info.get('reasoning', 'No reasoning provided')}")
    
    # Handle calculations
    if query_type == "calculation":
        operation = route_info.get("operation", "")
        numbers = route_info.get("numbers", [])
        
        # Check if we have valid calculation components
        if operation and len(numbers) >= 2:
            answer = perform_calculation(operation, numbers)
            processing_type = "calculation"
            validation_score = 10  # Calculations don't need validation
        else:
            # Fall back to RAG if calculation components are missing
            if VERBOSE:
                print("Incomplete calculation components, falling back to RAG")
            query_type = "rag"  # Force RAG processing
    
    # All non-calculation queries use RAG
    if query_type != "calculation":
        # Get relevant documents
        documents = get_documents(retriever, query)
        
        # Check if we have any documents
        if not documents:
            return {
                "answer": "I couldn't find any relevant information to answer your question in my knowledge base. Please try rephrasing your question or ask about a different topic.",
                "processing_type": "rag_no_documents",
                "processing_time": time.time() - start_time,
                "validation_score": 0
            }
        
        # First RAG attempt with standard prompt
        answer = get_answer_with_rag(query, documents, llm)
        processing_type = "rag_standard"
        
        # Validate the response quality
        validation = validate_response(llm, query, answer)
        validation_score = validation.get("score", 5)
        
        # If validation score is low, try again with enhanced prompt
        if validation_score < 7:
            if VERBOSE:
                print(f"First RAG attempt validation score: {validation_score}/10. Trying enhanced prompt.")
            
            # Second RAG attempt with enhanced prompt
            enhanced_context = "\n\n".join(documents)
            enhanced_prompt = f"""
            I need a more detailed and accurate answer to the following question:
            
            {query}
            
            The previous answer wasn't satisfactory. Let me provide you with relevant information:
            
            {enhanced_context}
            
            Based strictly on this information, please provide a comprehensive and accurate answer.
            Focus specifically on addressing the user's question with precise information from the provided context.
            If the information doesn't fully answer the question, clearly state what you can determine
            from the available information and what remains unknown.
            """
            
            result = llm.invoke(enhanced_prompt)
            improved_answer = result.content
            
            # Validate the improved answer
            improved_validation = validate_response(llm, query, improved_answer)
            improved_score = improved_validation.get("score", 0)
            
            # Use the improved answer if it scores better
            if improved_score > validation_score:
                if VERBOSE:
                    print(f"Enhanced RAG improved score from {validation_score} to {improved_score}")
                answer = improved_answer
                validation_score = improved_score
                processing_type = "rag_enhanced"
            
            # If still low quality after enhancement, add a note
            if improved_score < 6:
                processing_type = "rag_insufficient_info"
                information_gap_note = (
                    "\n\nNote: The information in my knowledge base may be incomplete on this topic. "
                    "I've provided the best answer based on available information, but there might be gaps or "
                    "additional details that would provide a more complete answer."
                )
                answer = answer + information_gap_note
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    
    return {
        "answer": answer,
        "processing_type": processing_type,
        "processing_time": processing_time,
        "validation_score": validation_score
    }

def interactive_mode():
    """Run an interactive query session"""
    # Make these variables accessible within this function
    global MAIN_MODEL, ROUTER_MODEL, VERBOSE
    
    try:
        # Load the retriever once
        retriever = load_retriever()
        
        print("\n==== Advanced Agentic RAG System with Feedback Loop ====")
        print(f"Main Model: {MAIN_MODEL}, Router Model: {ROUTER_MODEL}")
        print("Type your questions and press Enter. Type 'quit' to exit.")
        print("Type 'model:name' to change the main model.")
        print("Type 'router:name' to change the router model.")
        print("Type 'verbose:on/off' to toggle verbose mode.")
        print("=" * 60)
        
        while True:
            query = input("\nYour question: ")
            
            # Handle exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Handle model change commands
            if query.lower().startswith('model:'):
                MAIN_MODEL = query[6:].strip()
                print(f"Main model changed to: {MAIN_MODEL}")
                continue
                
            if query.lower().startswith('router:'):
                ROUTER_MODEL = query[7:].strip()
                print(f"Router model changed to: {ROUTER_MODEL}")
                continue
                
            # Toggle verbose mode
            if query.lower().startswith('verbose:'):
                mode = query[8:].strip().lower()
                if mode == 'on':
                    VERBOSE = True
                    print("Verbose mode enabled")
                elif mode == 'off':
                    VERBOSE = False
                    print("Verbose mode disabled")
                continue
            
            print("\nProcessing...")
            
            # Process the query
            result = process_query(query, retriever)
            
            # Display the result with color coding based on validation score
            print("\nANSWER:")
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)
            
            # Display processing details
            processing_type = result['processing_type']
            score = result['validation_score']
            
            # Add emojis to indicate quality
            score_indicator = "⭐" * min(5, max(1, round(score/2)))
            
            if processing_type == "calculation":
                method_desc = "Direct calculation (bypassed RAG)"
            elif processing_type == "rag_standard":
                method_desc = "Standard RAG"
            elif processing_type == "rag_enhanced":
                method_desc = "Enhanced RAG with feedback loop"
            elif processing_type == "rag_insufficient_info":
                method_desc = "Limited information available in knowledge base"
            elif processing_type == "rag_no_documents":
                method_desc = "No relevant documents found in knowledge base"
            else:
                method_desc = processing_type
                
            print(f"Processing method: {method_desc}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print(f"Answer quality: {score_indicator} ({score}/10)")
            
            # Provide feedback on quality
            if score >= 8:
                print("High quality answer ✓")
            elif score >= 6:
                print("Good answer, but could be improved")
            elif processing_type == "rag_insufficient_info" or processing_type == "rag_no_documents":
                print("⚠️ The information in the knowledge base may be incomplete or missing for this query")
                print("Consider adding more documents to the database or asking about a different topic")
            else:
                print("Lower quality answer - try rephrasing your question")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    interactive_mode()