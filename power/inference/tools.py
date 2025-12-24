
import os

import requests
from langchain_core.tools import tool

@tool
def weather_info(location: str):
    """
    Fetches weather information for a specific location.
    Args:
        location (str): The search location, e.g., "New York", "Tokyo"
    Returns:
        str: The weather information in JSON format.
    """
    # Pretend hard work to fetch weather info
    for _ in range(100000):
        pass
    return '{"location": "' + location + '", "temperature": "22C", "condition": "Sunny"}'


