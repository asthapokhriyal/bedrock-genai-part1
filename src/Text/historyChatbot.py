import boto3
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Get credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION")

# Set up the session
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")
history = []
def get_history():
    return "\n".join(history)

def get_configuration():
    return json.dumps({
            "inputText":get_history(),
            "textGenerationConfig" : {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
})

print(
    "Bot: Hello! I am a chatbot. I can help yu with anything you want to talk about."
)

while True:
    user_input = input("User: ")
    history.append("User: "+user_input)
    if user_input.lower() == "exit":
        break
    response = client.invoke_model(
        body=get_configuration(),
        modelId="amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json")
    response_body = json.loads(response.get('body').read())
    output_text =response_body.get('results')[0]['outputText'].strip()
    print(output_text) #titan config
    history.append(output_text)
    
