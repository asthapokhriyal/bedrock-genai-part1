import boto3
import json
import pprint

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

titan_model_id = 'amazon.titan-text-express-v1'

titan_config= json.dumps({
            "inputText":"hi",
            "textGenerationConfig" : {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
})

response = client.invoke_model(body=titan_config,
    modelId=titan_model_id,
    accept="application/json",
    contentType="application/json"

)

response_body = json.loads(response.get('body').read())

pp = pprint.PrettyPrinter(depth=4)
pp.pprint(response_body.get('results')) #titan config