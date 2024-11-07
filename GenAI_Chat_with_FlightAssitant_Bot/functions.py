from datetime import datetime
import json
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import openai

load_dotenv() 
  
# Initialize the environmental varialbles

WEATHER_API_KEY = os.getenv("openweathermap_weather_api_key")
FLIGHT_API_KEY = os.getenv("aviationstack_flight_api_key")


# Function to load JSON data from the file
def load_airports_data(filepath="FlightDetails.json"):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data

# Function to get IATA code by city name, ensuring the IATA code is not null or empty
def get_iata_code(city_name, airports_data):
    for airport_code, details in airports_data.items():
        if details.get("city").lower() == city_name.lower():
            iata_code = details.get("iata")
            if iata_code:  # Check if IATA code is not null or empty
                return iata_code
            else:
                continue
    return None

# Function to gather flight information from External APIs for the provided origin/destination IATA codes and the travel date
def fetch_flight_details(api_key, origin, destination, travel_date):
    """
    Fetches flights between specified origin and destination.

    Args:
    api_key (str): API key for AviationStack API.
    origin (str): The IATA code of the origin airport (e.g., "JFK").
    destination (str): The IATA code of the destination airport (e.g., "LAX").
    
    Returns:
    list: A list of top 5 flight details including flight name, code, datetime, cities, and cost.
    """
    try:
        # AviationStack API endpoint for flights
        api_url = "http://api.aviationstack.com/v1/flights"

        # Parameters for the API request
        params = {
            'access_key': api_key,
            'dep_iata': origin,
            'arr_iata': destination
        }

        # Print the full URL
        full_url = f"{api_url}?access_key={params['access_key']}&dep_iata={params['dep_iata']}&arr_iata={params['arr_iata']}"
        print("Full URL:", full_url)

        # Make the API request
        response = requests.get(api_url, params=params)

        # Check if the response is successful
        if response.status_code == 200:
            flights = response.json().get('data', [])
            # print('flights data: ',flights)
           
            # Process and display flight data
            flight_details = []
            for flight in flights:
                flight_date = flight.get("flight_date", "N/A")
                flight_status = flight.get("flight_status", "N/A")
                # print(flight_date)
                if flight_date == travel_date and flight_status != "landed":
                    flight_info = {
                            "Flight Name": flight.get("airline", {}).get("name", "N/A"),
                            "Flight Code": flight.get("flight", {}).get("iata", "N/A"),
                            "Departure Time": flight.get("departure", {}).get("estimated", "N/A"),
                            "Arrival Time": flight.get("arrival", {}).get("estimated", "N/A"),
                            "Origin City": flight.get("departure", {}).get("airport", "N/A"),
                            "Destination City": flight.get("arrival", {}).get("airport", "N/A"),
                            "Flight Status": flight.get("flight_status", "N/A"),
                            "Cost": flight.get("price", {}).get("total", "N/A"),  # Assuming the API provides cost information
                            "Travel_Date": flight.get("flight_date", "N/A")
                        }
                    flight_details.append(flight_info)

                # Sort flights by cost (assuming cost is available and is a number)
                sorted_flights = sorted(flight_details, key=lambda x: x["Cost"] if isinstance(x["Cost"], (int, float)) else float('inf'))
                
            # print('sorted_flights: ',sorted_flights)

            # Return the top 5 flights
            return sorted_flights[:5]
        else:
            print(f"Error: Received status code {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Main function to call gather all  Flight information for the provided origin/destincation locations with travel date
def get_flight_info(loc_origin,loc_destination, travel_date = datetime.now().strftime("%Y-%m-%d")):
    airports_data = load_airports_data()  # Load data from JSON file
    origin_iata = get_iata_code(loc_origin, airports_data)
    destination_iata = get_iata_code(loc_destination, airports_data)
  

    print(f"Origin IATA Code: {origin_iata}")
    print(f"Destination IATA Code: {destination_iata}")
    print("Travel Date", travel_date)

    flight_data_info = fetch_flight_details(FLIGHT_API_KEY, origin_iata, destination_iata, travel_date)
    # print('flight_data_info: ', flight_data_info)
    
    # Convert JSON data to DataFrame
    df = pd.json_normalize(flight_data_info)

    # Print DataFrame
    print(df)
    return json.dumps(flight_data_info)

# get_flight_info('Chennai','Pune','2024-11-05')

# Function to call the Flight API based on the flight code for flight booking service
def fetch_flight_details_using_flightcode(api_key, flight_code):
    """
    Fetches flight details based on flight code.

    Args:
    api_key (str): API key for AviationStack API.
    flight_code (str): The IATA code of the flight (e.g., "AA123").

    Returns:
    dict: Flight details including flight name, code, datetime, cities, and status.
    """
    try:
        # AviationStack API endpoint for flight details
        api_url = "http://api.aviationstack.com/v1/flights"

        # Parameters for the API request
        params = {
            'access_key': api_key,
            'flight_iata': flight_code
        }

        # Make the API request
        response = requests.get(api_url, params=params)

        # Check if the response is successful
        if response.status_code == 200:
            flight_data = response.json().get('data', [])

            if flight_data:
                flight = flight_data[0]  # Assuming the first result is the relevant one
                flight_info = {
                    "Flight Name": flight.get("airline", {}).get("name", "N/A"),
                    "Flight Code": flight.get("flight", {}).get("iata", "N/A"),
                    "Departure Time": flight.get("departure", {}).get("estimated", "N/A"),
                    "Arrival Time": flight.get("arrival", {}).get("estimated", "N/A"),
                    "Origin City": flight.get("departure", {}).get("airport", "N/A"),
                    "Destination City": flight.get("arrival", {}).get("airport", "N/A"),
                    "Status": flight.get("flight_status", "N/A")
                }
                return flight_info
            else:
                print("No flight details found for the given flight code.")
                return {}
        else:
            print(f"Error: Received status code {response.status_code}")
            return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    

#Main function to gather the flight booking information for the provided user name, passport number and the flight code
def book_flight(name, passport_num, flight_code): 

    book_flight_info = fetch_flight_details_using_flightcode(FLIGHT_API_KEY ,flight_code)
    
    # Append name and passport data
    book_flight_info['Passenger Name'] = name
    book_flight_info['Passport Number'] = passport_num
    print (book_flight_info)

    return json.dumps(book_flight_info)

# book_flight('Sindhu','ASD123','AA123')

# Function for registering a dummy complaint for the provided customer name, email and the description of the issue
def file_complaint(customer_name,customer_email,issue_desc):
    File_complaint_info = { 
        "user_name": customer_name,
        "user_email": customer_email,
        "issue_description": issue_desc,
        "issue_status": "Successfully filed the compliant"
    }
    return json.dumps(File_complaint_info)

# Define the API endpoint and your API key
def get_location_details(location):
    
    api_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': location,
        'limit': 1,
        'appid': WEATHER_API_KEY # Replace with your actual API key
    }

    # Make the GET request to the API
    response = requests.get(api_url, params=params)
    print('location URL', response)
   
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()        
        
    else:
        print(f"Error: {response.status_code}")

    return data

# data = get_location_details('Paris')

# Function to collect the real time weather information from external api for the provided latitude and longitude
def get_weather_details(lat, lon):
  
    api_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY # Replace with your actual API key
    }
    # Make the GET request to the API
    response = requests.get(api_url, params=params)
   
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()        
      
    else:
        print(f"Error: {response.status_code}")

    return data
    
# Main function to gather the weather report for the provided location
def get_weather_report(location):
    location_data = get_location_details(location)    
    # print(location_data)
    for location in location_data:
        name = location.get('name')
        lat = location.get('lat')
        lon = location.get('lon')
    print(name, lat, lon)

    weather_data = get_weather_details(lat,lon)
    print(weather_data)

    weather = {
    "location": name,
    # "Temperature": weather_data["main"]["temp"],
    "Feels Like": weather_data["main"]["feels_like"],
    "Weather": weather_data["weather"][0]["main"],
    # "temp_min": weather_data["main"]["temp_min"],
    # "temp_max": weather_data["main"]["temp_max"],
    "Description": weather_data["weather"][0]["description"]
    }

    print(weather)
    return json.dumps(weather)

# get_flight_info('Chennai','Pune', '2024-11-02' )