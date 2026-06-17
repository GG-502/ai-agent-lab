"""
Weather Tools Module

This module provides a weather tool that demonstrates function chaining capability.
The get_weather_by_date tool takes a date parameter and returns appropriate weather information.
"""

from datetime import datetime


def get_weather_by_date(date: str) -> str:
    """
    Get weather information for a given date.
    
    Returns weather forecast information for the specified date in YYYY-MM-DD format.
    If the date matches today's date, returns sunny weather at 72°F. 
    For any other date, returns rainy weather at 55°F.
    
    This tool demonstrates the AI's ability to chain function calls by using
    the date input and comparing it with the current date.
    
    Args:
        date: Date string in format "YYYY-MM-DD" (e.g., "2026-06-17")
    
    Returns:
        Weather information as a formatted string (e.g., "Sunny, 72°F")
    
    Raises:
        Implicit error handling: Returns error message if date format is invalid
    """
    try:
        # Validate date format
        try:
            input_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return f"Error: Invalid date format. Please use YYYY-MM-DD format (e.g., '2026-06-17'). You provided: '{date}'"
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Return weather based on whether it's today's date
        if date == today:
            return "Sunny, 72°F"
        else:
            return "Rainy, 55°F"
    
    except Exception as e:
        return f"Error: An unexpected error occurred while processing the date: {str(e)}"
