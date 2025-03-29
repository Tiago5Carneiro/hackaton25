from openai import OpenAI as openai

url = "https://apps.aws-london-novaprd1.svc.singlestore.com:8000/modelasaservice/6f2f23ce-b94d-4178-8478-df4b4efc09d9/v1"
token = "eyJhbGciOiJFUzUxMiIsImtpZCI6IjhhNmVjNWFmLThlNWEtNDQxOS04NmM4LWRkMDkxN2U1YWNlMSIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsibm92YXB1YmxpYyJdLCJleHAiOjE3NDU3ODg3NTgsIm5iZiI6MTc0MzE5Njc1MywiaWF0IjoxNzQzMTk2NzU4LCJzdWIiOiIyM2VmMGViMy0yZTMwLTQ1MjMtODc3NS1hMWVkZjAzZjAxYWUiLCJlbWFpbCI6ImRyb2RyaWd1ZXNAc2luZ2xlc3RvcmUuY29tIiwiaWRwSUQiOiJiNmQ2YTZiZC04NjYyLTQzYjItYjlkZS1hZjNhMjdlMGZhYzgiLCJlbWFpbFZlcmlmaWVkIjp0cnVlLCJzc29TdWJqZWN0IjoiMjNlZjBlYjMtMmUzMC00NTIzLTg3NzUtYTFlZGYwM2YwMWFlIiwidmFsaWRGb3JQb3J0YWwiOmZhbHNlLCJyZWFkT25seSI6ZmFsc2UsIm5vdmFBcHBJbmZvIjp7InNlcnZpY2VJRCI6IjZmMmYyM2NlLWI5NGQtNDE3OC04NDc4LWRmNGI0ZWZjMDlkOSIsImFwcElEIjoiYTAyODU3M2EtNzgzMS00ZTAwLWJkMzMtZjIyOGUwMjJlYjk3IiwiYXBwVHlwZSI6Ik1vZGVsQXNBU2VydmljZSJ9fQ.AemvHJJM6E1cx7mEru7ltz5AJJ6vfEBNK3YF7djy1m8BeqXLwmKR82fzxaaXFU9QBpzN7GFbyjaJTEqgnksVYBzjAWaDefdsAIe9qqP8bQFpih5OTHEPa-MYcRxut7N6ao4krZXjAn_PMHMjHfE4C2TTLwEi8XT8v9zmF-sQsac8XWOJ"

client = openai(
    base_url=url,
    api_key=token
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