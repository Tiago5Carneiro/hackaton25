from langchain_openai import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import testF1
import pandas as pd
import re


# Load IBM Granite Model Locally
model_id = "ibm-granite/granite-3.0-8b-instruct"

url = "https://apps.aws-london-novaprd1.svc.singlestore.com:8000/modelasaservice/720559d2-59c8-4457-b1fd-64735700276b/v1"
token = "eyJhbGciOiJFUzUxMiIsImtpZCI6IjhhNmVjNWFmLThlNWEtNDQxOS04NmM4LWRkMDkxN2U1YWNlMSIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsibm92YXB1YmxpYyJdLCJleHAiOjE3NDU4Nzc5MjUsIm5iZiI6MTc0MzI4NTkyMCwiaWF0IjoxNzQzMjg1OTI1LCJzdWIiOiIyM2VmMGViMy0yZTMwLTQ1MjMtODc3NS1hMWVkZjAzZjAxYWUiLCJlbWFpbCI6ImRyb2RyaWd1ZXNAc2luZ2xlc3RvcmUuY29tIiwiaWRwSUQiOiJiNmQ2YTZiZC04NjYyLTQzYjItYjlkZS1hZjNhMjdlMGZhYzgiLCJlbWFpbFZlcmlmaWVkIjp0cnVlLCJzc29TdWJqZWN0IjoiMjNlZjBlYjMtMmUzMC00NTIzLTg3NzUtYTFlZGYwM2YwMWFlIiwidmFsaWRGb3JQb3J0YWwiOmZhbHNlLCJyZWFkT25seSI6ZmFsc2UsIm5vdmFBcHBJbmZvIjp7InNlcnZpY2VJRCI6IjcyMDU1OWQyLTU5YzgtNDQ1Ny1iMWZkLTY0NzM1NzAwMjc2YiIsImFwcElEIjoiZTMxNDU4MGUtZTgyMC00OGNjLWE5YTktMGZjNzIyZDY0MGQ2IiwiYXBwVHlwZSI6Ik1vZGVsQXNBU2VydmljZSJ9fQ.ADUYSUYxC6uhKlRv0JX0WG8dVXoc77OJJZt59_F1rtPfZJkixYoqJfKfSTWt2GvMMH7E9dfKUGMRbimQn-J85XYGANfQSSmq3C4yoYQBaRJbnUJaklUDGmx5uro5w9Reyq7nnMM4t5Wg1c217B1e-KuVCb4H7fhEbDuL92AembpLX7zp"

gemini_token = "AIzaSyD-W6A3LjmEbOeFlZK43PDTZ8scUWrhPJ4"

#client = OpenAI(
#    model="unsloth/Meta-Llama-3.1-8B-Instruct",
#    http_client=httpx.Client(proxies = url)
#)
# Replace with your actual API base URL
llm = OpenAI(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",  # Adjust based on your model name
    temperature=0.6,
    openai_api_key=token,
    openai_api_base=url  # External Llama LLM endpoint
)

'''# Define IBM Granite LLM (IBM Watsonx)
ibm_llm = WatsonxLLM(
    model_id="ibm-granite/granite-3.0-8b-instruct",
    api_key=token,
    project_id=url
)'''

drivers = testF1.ret()

print(type(drivers))

def get_driver_lap_speed(string):
    """Retrieve speed for a specific F1 driver."""
    try :
        n, lap = string.split(",")
        # Check if the lap is number
        if not bool(re.search(r'^\d+$',lap)):
            return "Lap number must be a digit."
        if not bool(re.search(r'^\d+$',n)):
            return "Driver number must be a digit."
        return drivers[n]['Lap'][int(lap)-1]['Speed']
    except Exception as e:
        return f"An error occurred: {str(e)}"

driver_lap_speed_tool = Tool(
    name="Get Driver Lap Speed",
    func=get_driver_lap_speed,
    description='''
    Retrieves F1 driver's lap speed. 
    Input : driver number,lap number, event year, event name. 
    Output: speed in km/h.
    '''
)

def get_lap_time(string):
    """Retrieve lap time for a specific F1 driver."""
    n, lap = string.split(",")
    if not bool(re.search(r'^\d+$',lap)):
        return "Lap number must be a digit."
    if not bool(re.search(r'^\d+$',n)):
        return "Driver number must be a digit."
    return drivers[n]['Lap'][int(lap)-1]['LapTime']

driver_lap_time_tool = Tool(
    name="Get Driver Lap Time",
    func=get_lap_time,
    description="Retrieves F1 driver's lap time from the 2019 Monza race. Input example: 16,4. Output: lap time in seconds."
)

def get_lap_delta_position(string):
    """Retrieve speed for a specific F1 driver."""
    try :
        n, lap, year, name = string.split(",")
        # Check if the lap is number
        if not bool(re.search(r'^\d+$',lap)):
            return "Lap number must be a digit."
        if not bool(re.search(r'^\d+$',n)):
            return "Driver number must be a digit."
        return drivers[n]['Lap'][int(lap)-1]['DeltaPosition']
    except Exception as e:
        return f"An error occurred: {str(e)}"

driver_lap_delta_position_tool = Tool(
    name="Get Driver Lap Delta Position",
    func=get_lap_delta_position,
    description='''
    Retrieves F1 driver's lap position variation from the last lap. 
    Input : driver number,lap number, event year, event name. 
    Input example : 16,1,2019,Monza. 
    Output: difference between the position form the last lap and the current lap.
    ''')

def get_lap_current_position(string):
    """Retrieve lap current position for a specific F1 driver."""
    n, lap = string.split(",")
    if not bool(re.search(r'^\d+$',lap)):
        return "Lap number must be a digit."
    if not bool(re.search(r'^\d+$',n)):
        return "Driver number must be a digit."
    return drivers[n]['Lap'][int(lap)-1]['CurrentPosition']

driver_lap_current_position_tool = Tool(
    name="Get Driver Lap Current Position",
    func=get_lap_current_position,
    description="Retrieves F1 driver's current position from the 2019 Monza race. Input example: 16,4. Output: current position."
)

def get_lap_information(string):
    """Retrieve lap information for a specific F1 driver."""
    try :
        n, lap, year, name = string.split(",")
        # Check if the lap is number
        if not bool(re.search(r'^\d+$',lap)):
            return "Lap number must be a digit."
        if not bool(re.search(r'^\d+$',n)):
            return "Driver number must be a digit."
        return drivers[n]['Lap'][int(lap)-1]
    except Exception as e:
        return f"An error occurred: {str(e)}"

driver_lap_information_tool = Tool(
    name="Get Driver Lap Delta Position",
    func=get_lap_information,
    description='''
    Retrieves F1 driver's lap dictionary. 
    Input : driver number,lap number, event year, event name. 
    Input example : 16,1,2019,Monza. 
    Output: Dictionary that is provides the Speed, LapTime, CurrentPosition and DeltaPosition of that lap .
    ''')

def get_lap(string):
    """Retrieve lap list for a specific F1 driver."""
    try :
        n, year, name = string.split(",")
        # Check if the lap is number
        if not bool(re.search(r'^\d+$',n)):
            return "Driver number must be a digit."
        test = drivers[n]['Lap'].__str__()
        test = test.replace('{','').replace('}','').replace('\'','').replace('"','').replace(':','').replace(',','').replace('[','').replace(']','').replace('np.float','').replace('(','').replace(')','')
        return test
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
driver_lap_list_tool = Tool(
    name="Get Driver Lap List",
    func=get_lap,
    description='''
    Retrieves F1 driver's lap list. 
    Input : driver number, event year, event name. 
    Input example : 16,2019,Monza.
    Output: List of laps.
    ''')

tools = [driver_lap_list_tool]

memory = ConversationBufferMemory(memory_key="chat_history")
# Create an agent that can interact with the F1 DataFrame
# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True,
    handle_invalid_responses=True,
    handle_tool_errors=True,
    verbose=True,
    max_iterations=100,
    max_time=100,
)
# Create an agent

response = agent.run("""
Based on past F1 race results of the 2019 Monza race for the driver 16, I want you to give me a value from 0 to 100 to value his agressiveness
. Consider that the race has 53 laps, and that you can acess the variation in his position along the race. Consider as 
well that the higher the variance is the more aggressive the driver is. Consider you can get the information of the a full lap with a tool
that you can access. The input will have to be processed by you.
""")

print(response)