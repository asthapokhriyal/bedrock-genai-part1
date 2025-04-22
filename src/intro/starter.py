import boto3

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

bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")
pp = pprint.PrettyPrinter(depth=4)
models = bedrock.list_foundation_models()
for model in models["modelSummaries"]:
    pp.pprint(model)
# print(models)