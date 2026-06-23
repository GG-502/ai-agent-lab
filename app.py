from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from pydantic import SecretStr
from datetime import datetime
import os
import re
import time
import openai
import json
from weather_tools import get_weather_by_date

load_dotenv()

def calculator(expr: str) -> str:
    """Evaluate a mathematical expression given as a string and return the result as a string.

    WARNING: This uses Python's `eval()` for demonstration purposes and is NOT safe
    for untrusted input. For demo/scripting only.
    """
    try:
        # Minimal sandbox: remove builtins and provide empty globals/locals.
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_current_time(_: str) -> str:
    """Return the current date and time as a formatted string.

    The function accepts a single string parameter to satisfy the `Tool` interface
    but ignores it. Returns the current datetime in the format
    "%Y-%m-%d %H:%M:%S".
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def reverse_string(s: str) -> str:
    """Return the reverse of the input string `s`.

    Uses Python slice notation to reverse the string: `s[::-1]`.
    """
    return s[::-1]

def main():
    print("🤖 AI Agent Starting...")

    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("❌ Error: GITHUB_TOKEN not found!")
        print("📋 To fix this:")
        print("   1. Open your .env file")
        print("   2. Add this line: GITHUB_TOKEN=your_token_here")
        print("   3. Replace your_token_here with your actual GitHub Personal Access Token")
        print("   4. Save the file and run again")
        return

    print("✅ GitHub token found!")

    llm = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0,
        base_url="https://models.github.ai/inference",
        api_key=SecretStr(github_token)
    )

    print("🔗 Connected to GitHub AI Models!")

    # Helper: call API functions with retries on rate limit errors
    def call_with_retries(func, *args, retries: int = 3, backoff_factor: int = 2, initial_delay: int = 1, **kwargs):
        delay = initial_delay
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, openai.RateLimitError):
                    if attempt < retries - 1:
                        print(f"⚠️ Rate limit encountered, retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                        continue
                    else:
                        raise
                else:
                    raise

    # Tools available to the agent
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Use this tool to evaluate simple arithmetic expressions provided as a string, for example: '25 * 4 + 10'. Returns the result as a string.",
            args_schema={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate, e.g., '25 * 4 + 10'"
                    }
                },
                "required": ["expr"]
            }
        ),
        Tool(
            name="get_current_time",
            func=lambda **kwargs: get_current_time(""),
            description="Returns the current system date and time in format YYYY-MM-DD HH:MM:SS.",
            args_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="reverse_string",
            func=reverse_string,
            description="Reverses a string. Input should be a single string.",
            args_schema={
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "The string to reverse"
                    }
                },
                "required": ["s"]
            }
        ),
        Tool(
            name="get_weather",
            func=get_weather_by_date,
            description="Retrieve weather information for a specific date. This tool accepts a date parameter in YYYY-MM-DD format (e.g., '2026-06-17'). It returns weather conditions and temperature for that date. If the provided date matches today's date, it returns 'Sunny, 72°F'. For all other dates, it returns 'Rainy, 55°F'. The tool includes proper error handling for invalid date formats.",
            args_schema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to get weather for in YYYY-MM-DD format (e.g., '2026-06-17')"
                    }
                },
                "required": ["date"]
            }
        ),
    ]

    # Test queries to run through the agent
    basic_queries = [
        "What time is it right now?",
        "What is 25 * 4 + 10?",
        "Reverse the string 'Hello World'"
    ]
    
    weather_queries = [
        "What's the weather today?",
        "What's the weather on 2026-06-17?",
        "Get the weather forecast for 2026-06-18"
    ]

    # System prompt for the agent - professional and succinct
    system_prompt = (
        "You are a professional assistant with access to various tools. When asked a question:\n"
        "1. Use the provided tools to get accurate answers\n"
        "2. Always call the relevant tool for the task\n"
        "3. For weather queries, use the get_weather tool with dates in YYYY-MM-DD format\n"
        "4. Return the tool's result with a clear summary\n"
        "5. Be concise and professional in your response\n"
        "6. Do not add unnecessary explanation - just provide the answer"
    )

    # Create a tool map for easy lookup
    tool_map = {tool.name: tool for tool in tools}

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    print("\n" + "=" * 50)
    print("🚀 BASIC QUERIES (Calculator, Time, String Tools)")
    print("=" * 50 + "\n")
    
    for idx, query in enumerate(basic_queries, 1):
        print("─" * 50)
        print(f"📝 Query {idx}: {query}")
        print("─" * 50)
        
        try:
            # Build initial message with system prompt
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            # First LLM call
            try:
                response = call_with_retries(llm_with_tools.invoke, messages)
            except openai.RateLimitError:
                print("⚠️ Rate limit encountered; using direct tool execution.")
                response = None
            
            if response is None:
                # Fallback: execute tools directly
                ql = query.lower()
                if "time" in ql:
                    result = get_current_time("")
                elif any(op in ql for op in ["+", "-", "*", "/"]):
                    expr_match = re.search(r"([-+*/0-9().\s]+)", query)
                    result = calculator(expr_match.group(1).strip()) if expr_match else "Error: could not parse"
                elif "reverse" in ql:
                    m = re.search(r"'(.*?)'|\"(.*?)\"", query)
                    result = reverse_string(m.group(1) or m.group(2)) if m else "Error: could not parse"
                else:
                    result = "No handler available"
                print(f"\n✅ Result:\n   {result}")
            else:
                # Check if response has tool calls
                if response.tool_calls:
                    print("\n🔧 Agent selected tools:")
                    # Process each tool call
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_input = tool_call["args"]
                        
                        print(f"   → Calling '{tool_name}' with args: {tool_input}")
                        
                        # Execute the tool
                        if tool_name in tool_map:
                            tool_func = tool_map[tool_name].func
                            if tool_func is not None:
                                tool_result = tool_func(**tool_input)
                                print(f"   ✓ {tool_name} returned: {tool_result}")
                                tool_results.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
                    
                    # Second LLM call with tool results
                    messages.append(response)
                    messages.extend(tool_results)
                    
                    print("\n🤖 Agent reasoning with tool results...")
                    try:
                        final_response = call_with_retries(llm_with_tools.invoke, messages)
                    except openai.RateLimitError:
                        # Use the tool results directly
                        print(f"\n✅ Result (from tool):\n   {tool_results[0].content if tool_results else 'No result'}")
                        continue
                    
                    if final_response is not None:
                        print(f"\n✅ Result:\n   {final_response.content}")
                else:
                    # No tool calls, just return response
                    print(f"\n✅ Result:\n   {response.content}")
        
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
        
        print()  # Blank line for spacing between queries
    
    print("\n" + "=" * 50)
    print("🌦️  WEATHER QUERIES (Date-based Weather Tool)")
    print("=" * 50 + "\n")
    
    for idx, query in enumerate(weather_queries, 1):
        print("─" * 50)
        print(f"📝 Query {idx}: {query}")
        print("─" * 50)
        
        try:
            # Build initial message with system prompt
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            # First LLM call
            try:
                response = call_with_retries(llm_with_tools.invoke, messages)
            except openai.RateLimitError:
                print("⚠️ Rate limit encountered; using direct tool execution.")
                response = None
            
            if response is None:
                print("⚠️ Could not process query automatically")
                result = "No handler available"
                print(f"\n✅ Result:\n   {result}")
            else:
                # Check if response has tool calls
                if response.tool_calls:
                    print("\n🔧 Agent selected tools:")
                    # Process each tool call
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_input = tool_call["args"]
                        
                        print(f"   → Calling '{tool_name}' with args: {tool_input}")
                        
                        # Execute the tool
                        if tool_name in tool_map:
                            tool_func = tool_map[tool_name].func
                            if tool_func is not None:
                                tool_result = tool_func(**tool_input)
                            print(f"   ✓ {tool_name} returned: {tool_result}")
                            tool_results.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
                    
                    # Second LLM call with tool results
                    messages.append(response)
                    messages.extend(tool_results)
                    
                    print("\n🤖 Agent reasoning with tool results...")
                    try:
                        final_response = call_with_retries(llm_with_tools.invoke, messages)
                    except openai.RateLimitError:
                        # Use the tool results directly
                        print(f"\n✅ Result (from tool):\n   {tool_results[0].content if tool_results else 'No result'}")
                        continue
                    
                    if final_response is not None:
                        print(f"\n✅ Result:\n   {final_response.content}")
                else:
                    # No tool calls, just return response
                    print(f"\n✅ Result:\n   {response.content}")
        
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
        
        print()  # Blank line for spacing between queries
    
    print("🎉 Agent demo complete!")

if __name__ == "__main__":
    main()
