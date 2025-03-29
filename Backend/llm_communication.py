from openai import OpenAI as openai
import os 
import getpass
from langchain_openai import OpenAI
import httpx
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
import testF1
import pandas as pd

url = "https://apps.aws-london-novaprd1.svc.singlestore.com:8000/modelasaservice/47cc17c2-433b-4dca-b2cd-611c2d9ec221/v1"
token = "eyJhbGciOiJFUzUxMiIsImtpZCI6IjhhNmVjNWFmLThlNWEtNDQxOS04NmM4LWRkMDkxN2U1YWNlMSIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsibm92YXB1YmxpYyJdLCJleHAiOjE3NDU4MzgyNTEsIm5iZiI6MTc0MzI0NjI0NiwiaWF0IjoxNzQzMjQ2MjUxLCJzdWIiOiIyM2VmMGViMy0yZTMwLTQ1MjMtODc3NS1hMWVkZjAzZjAxYWUiLCJlbWFpbCI6ImRyb2RyaWd1ZXNAc2luZ2xlc3RvcmUuY29tIiwiaWRwSUQiOiJiNmQ2YTZiZC04NjYyLTQzYjItYjlkZS1hZjNhMjdlMGZhYzgiLCJlbWFpbFZlcmlmaWVkIjp0cnVlLCJzc29TdWJqZWN0IjoiMjNlZjBlYjMtMmUzMC00NTIzLTg3NzUtYTFlZGYwM2YwMWFlIiwidmFsaWRGb3JQb3J0YWwiOmZhbHNlLCJyZWFkT25seSI6ZmFsc2UsIm5vdmFBcHBJbmZvIjp7InNlcnZpY2VJRCI6IjQ3Y2MxN2MyLTQzM2ItNGRjYS1iMmNkLTYxMWMyZDllYzIyMSIsImFwcElEIjoiNTQzMmViM2ItMjhmMC00N2E1LWExM2QtZTJiNjdhZTY3NTkzIiwiYXBwVHlwZSI6Ik1vZGVsQXNBU2VydmljZSJ9fQ.ALXUC5cN3hmJfaiC38j9d1P1_lpBisi5m4cVuqTWVIKrtY7ky19KsttkdgrWafh4Yl13oHL3Fku8WZe1EWmXBPh_ARnGtT_iIsni0i6tp-YBZq5_yuxnxRq9SKeqStOYijqMaqwPFW3j5rt_928NAHevsZW4t8-bi-j5BR7_3LeaxUnL"

#client = OpenAI(
#    model="unsloth/Meta-Llama-3.1-8B-Instruct",
#    http_client=httpx.Client(proxies = url)
#)
# Replace with your actual API base URL
llm = OpenAI(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",  # Adjust based on your model name
    temperature=0.7,
    openai_api_key=token,
    openai_api_base=url  # External Llama LLM endpoint
)

def send_to_openai():
    
    response = client.chat.completions.create(
        model="unsloth/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]
    )
    print(response)

def example_tool(input_text: str):
    return f"Received: {input_text}"

tools = [
    Tool(
        name="ExampleTool",
        func=example_tool,
        description="A tool that echoes the input."
    )
]

drivers = testF1.ret()

from langchain_experimental.agents import create_pandas_dataframe_agent
drivers = drivers["16"]["Lap"]
# Create an agent that can interact with the F1 DataFrame
agent = create_pandas_dataframe_agent(llm, pd.DataFrame(drivers), verbose=True,allow_dangerous_code=True,max_iterations=None,max_execution_time=None)


response = agent.run("""
Based on past F1 race results for this driver, predict his speed and times for his laps for the 2020 Monza race. Consider his 
previous performance. Consider also that the 2020 Monza race will have as many laps as the last race, and that the delta position represents the current position and
the position of the previous lap. The position in this context is his standing.

""")

print(response)