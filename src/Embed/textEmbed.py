import boto3
import json
from dotenv import load_dotenv
import os
from similarity import cosineSimilarity

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

facts = [
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.'
]

# newFact = 'I like to play computer games'
newFact = 'Who is the president of USA?'

def getEmbedding(input: str):
    response = client.invoke_model(
        body=json.dumps({
            "inputText": input,
        }),
        modelId= "amazon.titan-embed-text-v1",
    contentType= "application/json",
    accept= "application/json"
    )
    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')

factsWithEmbeddings = []

for fact in facts:
    factsWithEmbeddings.append({
        'text': fact,
        'embedding': getEmbedding(fact)
    })

newFactEmbedding = getEmbedding(newFact)

similarities = []

for fact in factsWithEmbeddings:
    similarities.append({
        'text': fact['text'],
        'similarity': cosineSimilarity(fact['embedding'], newFactEmbedding)
    })

print(f"Similarities for fact: '{newFact}' with: ")
similarities.sort(key=lambda x: x['similarity'], reverse=True)
for similarity in similarities:
    print(f" '{similarity['text']}': {similarity['similarity']:.2f}")