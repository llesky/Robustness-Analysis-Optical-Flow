import json
from urllib import request, parse
import random
import websocket
import uuid 
import urllib.request
import urllib.parse
import io
import random

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

file_path = 'workflow_api_img2img.json'
with open(file_path, 'r') as file:
    prompt_text = file.read()

# print(prompt_text)

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


prompt = json.loads(prompt_text)

import os
from PIL import Image

# Define the source and destination folder paths
source_folder = '../mask 2/mask'
destination_folder = '../mask_'

seg_folder = "../seg"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Define a function to process the image
def process_image(image_path):
    # Open the image
    img = Image.open(image_path)
    # Perform some processing on the image (e.g., convert to grayscale)

    width, height = img.size
    img.save("input/1.png")

    img2 = Image.open(image_path.replace(source_folder, seg_folder))
    img2.save("input/2.png")

    # img.save("input/1.png")

    prompt["11"]["inputs"]["image"] = "1.png"

    # prompt["5"]["inputs"]["width"] = width
    # prompt["5"]["inputs"]["height"] = height
    L = ["banana", "dog", "mask", "cat", "fish", "Sofa", "Lamp", "Table", "Book", "Backpack", "Toaster", "Wallet", "weight_scale", "winter_glove", "Wok", "t_shirt"]
    e = random.choice(L)
    prompt["6"]["inputs"]["text"] = e # + prompt from LLM

    prompt["20"]["inputs"]["image"] = "2.png"

    x = random.randint(0,999999)
    #set the seed for our KSampler node
    prompt["3"]["inputs"]["seed"] = x

    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)
    image = Image.open(io.BytesIO(images["9"][0]))
    # for node_id in images:
    #     print(node_id)
    #     for image_data in images[node_id]:
    #         image = Image.open(io.BytesIO(image_data))
    #         # print(image.size)
    #         image.show()
    # dd


    # processed_img = Image.open("output/ComfyUI_00001_.png")

    return image

ws = websocket.WebSocket()

name = '_11'
# Loop through the folder structure
for root, dirs, files in os.walk(source_folder):
    
    for file in files:
        if file.endswith(f'{name}.png'):
        
            file_path = os.path.join(root, file)
            for i in range(1, 11):
                processed_img = process_image(file_path)

                destination_path = file_path.replace(source_folder, destination_folder)

                destination_subfolder = os.path.dirname(destination_path)
                if not os.path.exists(destination_subfolder):
                    os.makedirs(destination_subfolder)
                
                destination_path = destination_path.replace(f"{name}.png", f"{name}_{i}.png")
                print(destination_path)
                # dd
                processed_img.save(destination_path)

                # if os.path.exists("output/ComfyUI_00001_.png"):
                #     os.remove("output/ComfyUI_00001_.png")




