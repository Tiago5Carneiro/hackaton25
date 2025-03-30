from google import genai
from google.genai import types
from time import sleep

gemini_token = "AIzaSyDuCpQYdMwCnbkIlPys9CQTJ8HcM6yhCcA"

client = None 
config = None

def list_to_prompt(string):
    return string.__str__().replace("[", "").replace("]", "")

def create_client():
    global client,config
    pace_prediction = {
        "name": "pace_prediction",
        "description": "Predicts a theoritical pace value of a specific f1 driver (between 65 and 100) that is derived from a list of lap speeds from different f1 drivers.",
        "parameters": {
                "type": "object",
                "properties": {
                    'pace' : {
                        'type': 'number',
                        'description': 'The pace value for a specific driver between 65 and 100.'
                    },
                },
                "required": ["pace"], 
        }
    }
    
    agress_prediction = {
        "name": "agress_prediction",
        "description": "Predicts a theoritical agressiveness value of a specific f1 driver (between 65 and 100) that is derived from a list of variations in player position from different f1 drivers.",
        "parameters": {
                "type": "object",
                "properties": {
                    'agress' : {
                        'type': 'number',
                        'description': 'The agressiveness value for a specific driver between 65 and 100.'
                    },
                },
                "required": ["agress"], 
        }
    }
    
    tireman_prediction = {
        "name": "tireman_prediction",
        "description": "Predicts a theoritical tire managementd value of a specific f1 driver (between 65 and 100) that is derived from a list of amount of tire changes from different f1 drivers.",
        "parameters": {
                "type": "object",
                "properties": {
                    'tireman' : {
                        'type': 'number',
                        'description': 'The tire management value for a specific driver between 65 and 100.'
                    },
                },
                "required": ["tireman"], 
        }
    }
    
    consist_prediction = {
        "name": "consist_prediction",
        "description": "Predicts a theoritical consistency value of a specific f1 driver (between 65 and 100) that is derived from a list of race placements from different f1 drivers.",
        "parameters": {
                "type": "object",
                "properties": {
                    'consist' : {
                        'type': 'number',
                        'description': 'The consistency value for a specific driver between 65 and 100.'
                    },
                },
                "required": ["consist"], 
        }
    }
    
    qualifying_prediction = {
        "name": "qualifying_prediction",
        "description": "Predicts a theoritical qualifying value of a specific f1 driver (between 65 and 100) that is derived from a list of qualification points from different f1 drivers.",
        "parameters": {
                "type": "object",
                "properties": {
                    'qualifying' : {
                        'type': 'number',
                        'description': 'The qualification value for a specific driver between 65 and 100.'
                    },
                },
                "required": ["qualifying"], 
        }
    }
    
    client = genai.Client(api_key=gemini_token)
    tools = types.Tool(function_declarations=[pace_prediction,agress_prediction,tireman_prediction,consist_prediction,qualifying_prediction])
    config = types.GenerateContentConfig(tools=[tools])

def pace_prediction_function(pilot_list, pilot_name):
    global client, config
    upper_bound = []
    lower_bound = []
    
    size = len(pilot_list[pilot_name]['AvgSpeed'])
    for x in range(size):
        upper_bound.append(100)
    
    for x in range(size):
        lower_bound.append(65)
    
    prompt = "The pace value in a formula 1 videogame is how well pilots can mantain top speeds"
    
    for key,value in pilot_list.items():
        prompt += f"{key} has the lap speeds of {list_to_prompt(value['AvgSpeed'])}. "
        
    prompt += f"Upper Bond has the lap speeds of {list_to_prompt(upper_bound)} and Lower Bond has the lap speeds of {list_to_prompt(lower_bound)}. "
    
    prompt += f"Infer the pace value for {pilot_name} given both of their lap speeds and considering that the player with the highest average speed has 100 and the lowest 50."
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
        #  In a real app, you would call your function here:
        #  result = schedule_meeting(**function_call.args)
        return function_call.args['pace']
    else:
        print("No function call found in the response.")
        print(response.text)

def agress_prediction_function(pilot_list, pilot_name):
    global client, config
    upper_bound = []
    lower_bound = []
    
    size = len(pilot_list[pilot_name]['AvgSpeed'])
    for x in range(size):
        upper_bound.append(100)
    
    for x in range(size):
        lower_bound.append(65)
    
    prompt = "The agressiveness value in a formula 1 videogame is how often f1 pilots change positions"
    
    for key,value in pilot_list.items():
        prompt += f"{key} has the variation in position of {list_to_prompt(value['DeltaPos'])}. "
        
    prompt += f"Upper Bond has the variation in position of {list_to_prompt(upper_bound)} and Lower Bond has the variation in position of {list_to_prompt(lower_bound)}. "
    
    prompt += f"Infer the pace value for {pilot_name} given both of their variation in position and considering that the player with the highest variation has 100 and the lowest 50."
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
        return function_call.args['agress']
    else:
        print("No function call found in the response.")
        print(response.text)

def tireman_prediction_function(pilot_list, pilot_name):
    global client, config
    upper_bound = []
    lower_bound = []
    
    size = len(pilot_list[pilot_name]['AvgSpeed'])
    for x in range(size):
        upper_bound.append(100)
    
    for x in range(size):
        lower_bound.append(65)
    
    prompt = "The tire management value in a formula 1 videogame is how little f1 pilots change tires"
    
    for key,value in pilot_list.items():
        prompt += f"{key} has an amount of tire changes of {list_to_prompt(value['DeltaPos'])}. "
        
    prompt += f"Upper Bond has an amount of tire changes of {list_to_prompt(upper_bound)} and Lower Bond has an amount of tire changes of {list_to_prompt(lower_bound)}. "
    
    prompt += f"Infer the pace value for {pilot_name} given both of their amount of tire changes and considering that the player with the highest amount of tire changes has 100 and the lowest 50."
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
        return function_call.args['tireman']
    else:
        print("No function call found in the response.")
        print(response.text)

def consist_prediction_function(pilot_list, pilot_name):
    global client, config
    upper_bound = []
    lower_bound = []
    
    size = len(pilot_list[pilot_name]['AvgSpeed'])
    for x in range(size):
        upper_bound.append(100)
    
    for x in range(size):
        lower_bound.append(65)
    
    prompt = "The consistancy value in a formula 1 videogame is how often and how well f1 pilots place"
    
    for key,value in pilot_list.items():
        prompt += f"{key} has a list of placements of {list_to_prompt(value['DeltaPos'])}. "
        
    prompt += f"Upper Bond has a list of placements of {list_to_prompt(upper_bound)} and Lower Bond has a list of placements of {list_to_prompt(lower_bound)}. "
    
    prompt += f"Infer the pace value for {pilot_name} given both of their a list of placements and considering that the player with the highest placement has 100 and the lowest 50."
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
        return function_call.args['consist']
    else:
        print("No function call found in the response.")
        print(response.text)

def qualifying_prediction_function(pilot_list, pilot_name):
    global client, config
    upper_bound = []
    lower_bound = []
    
    size = len(pilot_list[pilot_name]['AvgSpeed'])
    for x in range(size):
        upper_bound.append(100)
    
    for x in range(size):
        lower_bound.append(65)
    
    prompt = "The consistancy value in a formula 1 videogame is how well f1 pilots place in qualification rounds"
    
    for key,value in pilot_list.items():
        prompt += f"{key} has a list of qualification points {list_to_prompt(value['DeltaPos'])}. "
        
    prompt += f"Upper Bond has a list of qualification points of {list_to_prompt(upper_bound)} and Lower Bond has a list of qualification points of {list_to_prompt(lower_bound)}. "
    
    prompt += f"Infer the pace value for {pilot_name} given both of their a list of qualification points and considering that the player with the highest qualification points has 100 and the lowest 50."
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
        return function_call.args['qualifying']
    else:
        print("No function call found in the response.")
        print(response.text)

def card_creation(pilots_stats):
    create_client()
    pilot_cards = {}
    
    for key in pilots_stats.keys():
       
        pace = None
        while pace is None:
            try:
                pace = pace_prediction_function(pilots_stats, key)
            except:
                print("Slepping")
                sleep(61)

        agress = None
        while agress is None:
            try:
                agress = agress_prediction_function(pilots_stats, key)
            except:
                print("Slepping")
                sleep(61)
        defense = 165 - agress
        tireman = None
        while tireman is None:
            try:
                tireman = tireman_prediction_function(pilots_stats, key)
            except:
                print("Slepping")
                sleep(61)
        consist = None
        while consist is None:
            try:
                consist = consist_prediction_function(pilots_stats, key)
            except:
                print("Slepping")
                sleep(61)
        qualifying = None
        while qualifying is None:
            try:
                qualifying = qualifying_prediction_function(pilots_stats, key)
            except:
                print("Slepping")
                sleep(61)
            
        pilot_cards[key] = {
            "Pace": round(pace),
            "Agressiveness": round(agress),
            "Defense": round(defense),
            "TireMan" : round(tireman),
            "Consistency" : round(consist),
            "Qualifying" : round(qualifying),
            "Overall" : round(max(0,min(pace*0.4 + agress*0.05 + defense*0.05 + tireman*0.3 + consist*0.15 + qualifying*0.15,99))),
        }
        
    return pilot_cards
