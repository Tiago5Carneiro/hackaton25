from google import genai
from google.genai import types

gemini_token = "AIzaSyDsSeHHTa1cOxADdObLhm7Q4hsYC4OOFVw"

client = None 
config = None

test = {
    "Lewis Hamilton": 1.2,
    "Max Verstappen": 2.3,
    "Fernando Alonso": 3.4,
    "Charles Leclerc": 4.4,
    "Carlos Sainz": 3.3,
    "Sergio Perez": 6.8,
}
print(test)
def list_to_prompt(string):
    return string.__str__().replace("[", "").replace("]", "")

def create_client():
    global client,config
    race_prediction = {
        "name": "race_prediction",
        "description": "Predicts the outcome of an f1 race, given the average positions of the f1 pilots and their names.",
        "parameters": {
                "type": "object",
                "properties": {
                    'race' : {
                        'type': 'array',
                        'items': {
                            'type' : 'string',
                            'description': 'The name of the driver.'
                        },
                        'description': 'The outcome of the ray as an ordered array of f1 driver names .'
                    },
                },
                "required": ["race"], 
        }
    
    }
    
    client = genai.Client(api_key=gemini_token)
    tools = types.Tool(function_declarations=[race_prediction])
    config = types.GenerateContentConfig(tools=[tools])

def race_predict_function(pilot_info):
    global client, config
    
    prompt = "The average position of a driver in a formula 1 videogame is a measure of how well they perform in the game. The average position is calculated by taking the sum of the positions of the driver in each lap and dividing it by the number of laps. The lower the average position, the better the performance. "
    
    upper_bound = 100
    lower_bound = 50
    
    prompt += f"Upper Bound has the average position of {upper_bound}. "
    prompt += f"Lower Bound has the average position of {lower_bound}. "
    
    for key,value in pilot_info.items():
        prompt += f"{key} has the average position of {value}. "
        
    prompt += f"Predict the outcome of a race between these f1 pilots value given their average position and considering that the f1 pilot with the lowest average speed player has 100 and the lowest 50. Also consider that the actual outcome can fluctuate a bit. The output should be an ordered array of f1 driver names. "
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config,
    )

    # Check for a function call
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Function to call: {function_call.name}")
        print(f"Arguments: {function_call.args}")
        return function_call.args
    else:
        print("No function call found in the response.")
        print(response.text)
    
create_client()
print(race_predict_function(test))