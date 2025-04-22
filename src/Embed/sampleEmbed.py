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

fact = "The first moon landing was in 1969."
animal = "cat"

response = client.invoke_model(
    body=json.dumps(
        {
            "inputText": fact,
        }
    ),
    modelId= "amazon.titan-embed-text-v1",
    contentType= "application/json",
    accept= "application/json",
    
)
response_body = json.loads(response.get("body").read())
print(response_body.get("embedding"))