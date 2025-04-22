import boto3
import json
import base64
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

def get_configuration(inputImage: str):
    return json.dumps({
        "taskType": "INPAINTING",
        "inPaintingParams":{
            "text": "Make the cat black and blue",
            "negativeText": "bad quality, low res",
            "image": inputImage,
            "maskPrompt": "cat"
        },
        "imageGenerationConfig":{
            "numberOfImages": 1,
            "height": 512,
            "width":512,
            "cfgScale": 8.0
        }
    })

with open("cat.png", "rb") as f:
    base_image = base64.b64encode(f.read()).decode("utf-8")

response = client.invoke_model(
    body=get_configuration(base_image),
    modelId="amazon.titan-image-generator-v1",
    accept="application/json",
    contentType="application/json")

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]

base_64_image = base64.b64decode(base64_image)

file_path = "catedited.png"
with open(file_path, "wb") as f:
    f.write(base_64_image)