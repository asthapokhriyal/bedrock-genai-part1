import boto3
import json
import base64
from similarity import cosineSimilarity
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

images = [
    'imageDump/img1.png',
    'imageDump/img2.png',
    'imageDump/img3.png',
]


def getImagesEmbedding(imagePath: str):
    with open(imagePath, "rb") as f:
        base_image = base64.b64encode(f.read()).decode("utf-8")
    response = client.invoke_model(

        body=json.dumps({
            "inputImage": base_image,
        }),
        modelId='amazon.titan-embed-image-v1',
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')


imagesWithEmbeddings = []

for image in images:
    imagesWithEmbeddings.append({
        'path': image,
        'embedding': getImagesEmbedding(image)
    })

test_image = 'imageDump/result.png'
test_img_embedding = getImagesEmbedding(test_image)

similarities = []

for image in imagesWithEmbeddings:
    similarities.append({
        'path': image['path'],
        'similarity': cosineSimilarity(image['embedding'], test_img_embedding)
    })


similarities.sort(key=lambda x: x['similarity'], reverse=True)
print(f"Similarities for : '{test_image}' with: ")


for similarity in similarities:
    print(f" '{similarity['path']}': {similarity['similarity']:.2f}")
