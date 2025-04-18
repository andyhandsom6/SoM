from scipy.ndimage import label
import numpy as np

import os
from PIL import Image
import json
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "sk-s7bZHTQvn4JjuTJTRWmPIk72L1HjBzQaUm8mxO0JhW6a4iq4"
from gpt4v import request_gpt4v


def split_res(res):
    try:
        res_sp = res.split("[")[1].split("]")[0].lower()
        if res_sp != "left" and res_sp != "right": 
            raise ValueError
        return res_sp
    except:
        print("Wrong answer from GPT: ", {res})
        return None

def main():
    json_path = "/mnt/navid/SoM/SYNTH-exp-dgx/cap/human_cap_no_som.json"
    img_name, ins_name = "image_name", "caption_"
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data has three attribute: file, instruction, and gt
    folder = "/mnt/navid/SoM/SYNTH-exp-dgx"
    output_folder = f"{folder}-no-som-mask"
    
    result = 0
    episode_res = 0
    gpt_response = []
    som_error = 0
    idx = -1
    for item in tqdm(data):
        idx += 1
        if idx != 0 and idx % 20 == 0:
            print(f"item {idx} finished.")
            print(f"acc. = {result} / {2*idx} = {result/idx/2}")
            print(f"episode_acc. = {episode_res} / {idx} = {episode_res/idx}")
            print(f"current som error: {som_error}")
        gpt_response_item = {}
        image_path = os.path.join(folder, item[img_name])
        image = Image.open(image_path)
        episode_success = 0
        for direction in ["left", "right"]:
            instruction = item[ins_name+direction]
            try: 
                res = request_gpt4v(instruction, image)
            except Exception as e:
                print("Error Occured! GPT-4V failed. ")
                continue
            gpt_response_item[direction] = res
            # find the seperate numbers in sentence res
            gpt_ans = split_res(res)
            result += int(gpt_ans == direction)
            episode_success += int(gpt_ans == direction)
        if episode_success == 2:
            episode_res += 1
        gpt_response.append(gpt_response_item)
    
    print(f"acc. = {result} / {2*len(data)} = {result/len(data)/2}")
    print(f"episode_acc. = {episode_res} / {len(data)} = {episode_res/len(data)}")
    print(f"som_error = {som_error}")
    with open(os.path.join(output_folder, "gpt_response.json"), 'w') as f:
        json.dump(gpt_response, f)


if __name__ == "__main__":
    main()
    