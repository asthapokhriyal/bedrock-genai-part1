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

stability_image_config = json.dumps({
    "text_prompts": [
        {
            "text": 'a real life photo of a lizard in park',
        }
    ],
    "height": 512,
    "width": 512,
    "cfg_scale": 10,
    "style_preset": '3d-model',
})

response = client.invoke_model(
    body=stability_image_config,
    modelId="stability.stable-diffusion-xl-v1",
    accept="application/json",
    contentType="application/json")

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("artifacts")[0].get("base64")

base_64_image = base64.b64decode(base64_image)

file_path = "cat.png"
with open(file_path, "wb") as f:
    f.write(base_64_image)