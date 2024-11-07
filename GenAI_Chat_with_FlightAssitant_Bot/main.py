import openai
import os 
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json
from datetime import date, datetime, timedelta
import requests
from functions import *

load_dotenv() 
  
# Initialize the environmental variables
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("AZURE_API_KEY")

WEATHER_API_KEY = os.getenv("openweathermap_weather_api_key")
FLIGHT_API_KEY = os.getenv("aviationstack_flight_api_key")
 
 
# Initialize OpenAI client
openai = AzureOpenAI(
      azure_endpoint=AZURE_ENDPOINT,
      api_key=API_KEY,
      api_version=API_VERSION
  )

# --------------------------------------------------------------
# Use OpenAI’s Function Calling Feature
# --------------------------------------------------------------

function_descriptions = [
    {
        "name": "book_flight",
        "description": "Book a flight based on flight information",
        "parameters": {
            "type": "object",
            "properties": {
                 "name": {
                    "type": "string",
                    "description": "Name of the user, e.g. John",
                },  
                "passport_num": {
                    "type": "string",
                    "description": "Passport number of the user, e.g. AAA12345",
                },             
               "flight_code": {
                    "type": "string",
                    "description": "The flight code of the airline, e.g. AE 2211",
                },
            },
            "required": ["name", "passport_num", "flight_code"],
        },
    },
    {
        "name": "get_flight_info",
        "description": "Get flight information between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. Chennai",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. Banglore",
                },
                 "travel_date": {
                    "type": "string",
                    "description": "The travel date, e.g 2024-11-03",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    },    
    {
        "name": "file_complaint",
        "description": "File a complaint as a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "The name of the customer, e.g. John Doe",
                },
                "customer_email": {
                    "type": "string",
                    "description": "The email address of the customer, e.g. john@doe.com",
                },
                "issue_desc": {
                    "type": "string",
                    "description": "Description of issue",
                },
            },
            "required": ["user_name", "user_email", "issue_desc"],
        },
    },
    {
        "name": "get_weather_report",
        "description": "Get the weather details for the provided location: ",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location: Eg Paris",
                }
            },
            "required": ["location"],
        },
    }
]

 
# Chat Initialization
def initialize_conversation():
    '''    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    system_message = f"""
    Role: You are an intelligent, smart, and friendly personal flight AI Assistant. Your goal is to seamlessly manage the user's travel plans and provide real-time flight support. You offer three exclusive services:

     {delimiter}
    Discover Flight Information: Requires origin , destination and travel date details from the user.
    Book Your Flight: Requires name of the user, origin, destination, date and time, and airline name details from the user. If the user continues to use this service from 'Discover Flight Information' then try to use the collected inputs and request for the name and passport details.
    Lodge a Complaint: Requires username, user email, and issue description from the user.
     {delimiter}
    
    Instructions:

     {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you"ll be heavily penalised:
    - The values for all keys should be retrieved from user based on the service chosen by user . Ask the questions repeatedly to get all the sufficient inputs. Do not halusinate the values.
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user"s response.
     {delimiter}

    Identify the User's Intent: Determine whether the user wants to discover flight information, book a flight, or lodge a complaint.
    Collect Required Information: Based on the identified intent, ask the user for the necessary details.
    Provide the Service: Use the collected information to provide the requested service.
    
     {delimiter}
    Few-Shot Learning Examples:

    Example 1: Book Your Flight
    User: I want to book a flight. 
    AI Assistant: Great! I can help you book a flight. Could you please provide your name, passport number and the flight code? 
    User: I need a flight from San Francisco to Chicago on 2023-12-01 at 10:00 AM with United Airlines. 
    AI Assistant: Thank you! I am booking your flight from San Francisco to Chicago on 2023-12-01 at 10:00 AM with United Airlines.

    Example 2: Discover Flight Information
    User: I need to find flights from New York to Los Angeles. 
    AI Assistant: Sure! I can help you with that. Could you please provide the origin and destination details? 
    User: Origin is New York and destination is Los Angeles. 
    AI Assistant: Thank you! I am fetching the flight information for you.

   
    Example 3: Lodge a Complaint
    User: I need to file a complaint. 
    AI Assistant: I’m sorry to hear that you have an issue. Could you please provide your name, email address, and a description of the issue? 
    User: My name is Jane Doe, my email is jane.doe@example.com, and my issue is that my flight was delayed for 5 hours. 
    AI Assistant: Thank you, Jane. I am filing your complaint regarding the 5-hour flight delay.
    ####
   
    {delimiter}
    Chain of Thought:

    Greet the User: Start with a friendly greeting and ask how you can assist them.
    Identify the Service: Listen to the user's request and identify whether they need flight information, want to book a flight, or want to lodge a complaint.
    Ask for Required Details: Based on the identified service, ask the user for the necessary details.
    For Discover Flight Information: Ask for origin and destination.
    For Book Your Flight: Ask for customer name, passport number and the flight code
    For Lodge a Complaint: Ask for username, user email, and issue description.
    Confirm Details: Repeat the details back to the user to confirm accuracy.
    Provide the Service: Use the confirmed details to provide the requested service.
    Thank the User: End the interaction by thanking the user and offering further assistance if needed.
    Detailed Interaction Examples:

    {delimiter}
    Example Interaction 1: Book Your Flight
    AI Assistant: Hello! How can I assist you with your travel plans today? 
    User: I want to book a flight. 
    AI Assistant: Great! I can help you book a flight. 
    Could you please provide the origin, destination, date and time, and the airline name? 
    User: I need a flight from Miami to Seattle on 2023-11-15 at 3:00 PM with Delta Airlines. 
    AI Assistant: Thank you! Just to confirm, you want to book a flight from Miami to Seattle on 2023-11-15 at 3:00 PM with Delta Airlines. Is that correct? 
    User: Yes, that's correct. 
    AI Assistant: Perfect! I am booking your flight now. Thank you for choosing our service. Is there anything else I can help you with?

    Example Interaction 2: Discover Flight Information
    AI Assistant: Hello! How can I assist you with your travel plans today? 
    User: I need to find flights from New York to Los Angeles. 
    AI Assistant: Sure! To help you find the best flights, could you please confirm the origin and destination details? 
    User: Origin is New York and destination is Los Angeles. 
    AI Assistant: Thank you! I am fetching the flight information for you.

    Example Interaction 3: Lodge a Complaint
    AI Assistant: Hello! How can I assist you with your travel plans today? User: I need to file a complaint. 
    AI Assistant: I’m sorry to hear that you have an issue. Could you please provide your name, email address, and a description of the issue? 
    User: My name is John Smith, my email is EMAIL_ADDRESS_3, and my issue is that my baggage was lost. 
    AI Assistant: Thank you, John. I am filing your complaint regarding the lost baggage.
    
 
     {delimiter}
    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation =system_message
    return conversation

# Let's initialise conversation
system_message = initialize_conversation()
# print(system_message)


# Define a Chat completion API call
def get_chat_completions(input,input_prompt):
    system_message_json_output =  """<<. Return output in JSON format to the key output.>>"""
    chat_completion = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            seed = 2345,
            temperature=0,
            functions=function_descriptions,
            function_call="auto",)
    print('Checking: ',chat_completion.choices[0].message)

    if chat_completion.choices[0].message.content is None:
        output = chat_completion.choices[0].message.function_call.name

        params = json.loads(chat_completion.choices[0].message.function_call.arguments)
        chosen_function = eval(chat_completion.choices[0].message.function_call.name)
        flight = chosen_function(**params)

            
        print('First Response: ',chat_completion.choices[0].message)
        print('function name: ',chat_completion.choices[0].message.function_call.name)
        print('Params: ', params)
        print('Chosen Function: ',chosen_function)
        print('Flight: ',flight)
            # print(input_prompt)
            
        second_completion = openai.chat.completions.create(
                model = MODEL,
                messages=[
                {"role": "user", "content": input_prompt},
                {"role": "system", "name": chat_completion.choices[0].message.function_call.name, "content": flight},
            ],
                # functions=function_descriptions,
                # function_call="auto",
            )
        print('Second Response: ',second_completion.choices[0].message)
        output = second_completion.choices[0].message.content
    
    else:
        output = chat_completion.choices[0].message.content
        
    output = output.replace('\'','"')
    # print(chat_completion.choices[0].message.function_call)
    return output  
        
#Function to flag the harmful conversation
def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    
    debug_user_input = f'Please determine if the following input violates content guidelines (e.g., harmful, hateful, or inappropriate): {user_input} return the response as "Flagged" for any harmful, hateful or inappropriate else "Not Flagged" you wil be penalized heavily in case of other than "Flagged" or "Not Flagged" '
    messages = [{'role':'user','content':debug_user_input}]

    # Check if the input was flagged by the moderation system.
    response = get_chat_completions(messages, debug_user_input)
    return response


# input_prompt ='I am Sindhu want to raise a complaint,my email id is aaa@gmail.com and the issue is Flight is delayed'
# input_prompt = 'When is the next flight from Chennai to Pune on 2024-11-07?'

# # input_prompt = 'I am Sindhu, I want to book a flight ticket from Chennai to Banglore tomorrow morning 07 AM in Indigo airlines'
# input_prompt = "What is the weather in Delhi? "
# input_prompt = "How to blast the flight? "
# messages = [{'role':'user','content':input_prompt}]

# response = get_chat_completions(messages,input_prompt)
# print('Final Response: ',response)

# input_prompt = "How to Hijack the flight? "   
# response = moderation_check(input_prompt)

# print(response)    



# Streamlit Dashboard creation

st.set_page_config(page_title="FlyEasy: Your Personal Flight Assistant")
st.markdown("<h2 style='text-align: center;'>FlyEasy: Your Personal Flight Assistant</h2>", unsafe_allow_html=True)
# st.title('''FlyEasy: Your Personal Flight Assistant''')


with st.sidebar:
    st.subheader("Hello, Traveler!")
    st.write('''
    Seamlessly manage your travel plans and receive real-time flight support using your presonal flight assistant bot
    
    Our exclusive offerings include:
    * Discover Flight Information
    * Book Your Flight
    * Lodge a Complaint
    * Know the weather information             
    ''')
    st.image(" https://cdn.botpenguin.com/assets/website/Assistant_Bot_b1c68b7993.webp", caption="FlyEasy Bot")

    # Add End Conversation button at the top
    if st.button("End Conversation"):
        st.session_state.chat_history = [
            AIMessage(content="Thank you for using FlyEasy. Have a great day!")
        ]
        st.experimental_rerun()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(initialize_conversation()),
        AIMessage(content="Hello! I'm your personal Flight assistant. How can I assist you today?"),
    ]


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content) 
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")


if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        moderation = moderation_check(user_query)
        
        if moderation == 'Flagged':
            response = "Sorry, this message has been flagged.I can't respond to it."
            st.markdown(response)
        else:
            message=[{"role": "system", "content": user_query}] 
            response =  get_chat_completions(message,user_query)
            print('Final Res: ',response)      
            st.markdown(response)
        

    st.session_state.chat_history.append(AIMessage(content=response))



 