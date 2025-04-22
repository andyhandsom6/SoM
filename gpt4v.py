import os
import base64
import requests
from io import BytesIO

# Get OpenAI API Key from environment variable
api_key = os.environ["OPENAI_API_KEY"]
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# metaprompt = '''
# - For any marks mentioned in your answer, please highlight them with [].
# '''    

metaprompt = " You should only give me **one** numerical mark, which can best represent the person you recognized. Highlight your answer with []. "

# left_of_right_no_som: 
# metaprompt = " Choose your answer from \"left\" and \"right\". Mark it with []. "

# left_of_right_gt_mark:
# metaprompt = " Choose your answer from \"[1]\" and \"[2]\". Remember to mark your answer with []. "

# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_inputs(message, image):

    # # Path to your image
    # image_path = "temp.jpg"
    # # Getting the base64 string
    # base64_image = encode_image(image_path)
    base64_image = encode_image_from_pil(image)

    payload = {
        # "model": "gpt-4-vision-preview",
        # "model": "gpt-4-all",
        # "model": "gpt-4-turbo-2024-04-09",
        "model": "gpt-4o-2024-11-20",
        "messages": [
        {
            "role": "system",
            "content": [
                # metaprompt
            ]
        }, 
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": message+metaprompt, 
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 800
    }

    return payload

def request_gpt4v(message, image):
    payload = prepare_inputs(message, image)
    response = requests.post("https://vip.DMXapi.com/v1/chat/completions", headers=headers, json=payload)
    # import pdb
    # pdb.set_trace()
    res = response.json()['choices'][0]['message']['content']
    return res
