# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import gradio as gr
import torch
import argparse

# # seem
# from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
# from seem.utils.distributed import init_distributed as init_distributed_seem
# from seem.modeling import build_model as build_model_seem
# from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# # sam
# from segment_anything import sam_model_registry
# from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
# from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np

import os
from PIL import Image
import json
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "sk-s7bZHTQvn4JjuTJTRWmPIk72L1HjBzQaUm8mxO0JhW6a4iq4"
from gpt4v import request_gpt4v

'''
build args
'''
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./model_zoo/swinl_only_sam_many2many.pth"
sam_ckpt = "./model_zoo/sam_vit_h_4b8939.pth"
seem_ckpt = "./model_zoo/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
# opt_seem = load_opt_from_config_file(seem_cfg)
# opt_seem = init_distributed_seem(opt_seem)

'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
# model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
# model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()
# with torch.no_grad():
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
        # model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

@torch.no_grad()
def inference(image, slider = 1.5, mode = 'Automatic', alpha = 0.1, label_mode = 'Number', anno_mode = ['Mask', 'Mark'], *args, **kwargs):
    # _image = image['background'].convert('RGB')
    # _mask = image['layers'][0].convert('L') if image['layers'] else None
    _image = image.convert('RGB')
    
    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        # if mode == "Interactive":
        #     labeled_array, num_features = label(np.asarray(_mask))
        #     spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        else:
            raise NotImplementedError
        # elif model_name == 'sam':
        #     model = model_sam
        #     if mode == "Automatic":
        #         output, mask = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
        #     # elif mode == "Interactive":
        #     #     output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        # elif model_name == 'seem':
        #     model = model_seem
        #     if mode == "Automatic":
        #         output, mask = inference_seem_pano(model, _image, text_size, label_mode, alpha, anno_mode)
        #     # elif mode == "Interactive":
        #     #     output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        return output, mask

def split_res(res):
    res = res.split(' ')
    res = [r.replace('.','').replace(',','').replace(')','').replace('"','') for r in res]
    # find all numbers in '[]'
    res = [r for r in res if '[' in r]
    res = [r.split('[')[1] for r in res]
    res = [r.split(']')[0] for r in res]
    res = [r for r in res if r.isdigit()]
    res = list(set(res))
    return res

if __name__ == "__main__":
    json_path = "/mnt/navid/SoM/SYNTH-exp-dgx/human_caption.json"
    img_name, ins_name = "image_name", "caption_"
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data has three attribute: file, instruction, and gt
    folder = "/mnt/navid/SoM/SYNTH-exp-dgx"
    output_folder = f"{folder}_mask"
    
    result = 0
    episode_res = 0
    gpt_response = []
    som_error = 0
    idx = 153
    item = data[idx]
    if idx != 0 and idx % 20 == 0:
        print(f"item {idx} finished.")
        print(f"acc. = {result} / {2*idx} = {result/idx/2}")
        print(f"episode_acc. = {episode_res} / {idx} = {episode_res/idx}")
        print(f"current som error: {som_error}")
    gpt_response_item = {}
    image_path = os.path.join(folder, item[img_name])
    name_without_suffix = os.path.splitext(item[img_name])[0]
    image = Image.open(image_path)
    try:
        output, mask = inference(image)
    except Exception as e:
        print("Error Occured! SoM segmentation failed. ")
        som_error += 2
        output_saved = image
        output_saved.save(
            os.path.join(output_folder, f"{name_without_suffix}_mark.jpg")
        )
        raise NotImplementedError
    output_image = Image.fromarray(output)
    episode_success = 0
    for direction in ["left", "right"]:
        fail_tag = False
        
        instruction = item[ins_name+direction]
        # import pdb
        # pdb.set_trace()
        try: 
            res = request_gpt4v(instruction, output_image)
        except Exception as e:
            print("Error Occured! GPT-4V failed. ")
        gpt_response_item[direction] = res
        # find the seperate numbers in sentence res
        num_list = split_res(res)
        if len(num_list) != 1:
            som_error += 1
            print("Error Occured! Might because of som. response from gpt is: ", res)
            # print("num_list is: ", num_list, ". Trying again...")
            continue
        try:
            coords = np.argwhere(mask[int(num_list[0])-1]['segmentation'])
        except IndexError:
            print("Error Occured! Might because of GPT. response from gpt is: ", res)
            continue
        mean_w_coords = coords[:, 1].mean()
        if mean_w_coords < mask[0]['segmentation'].shape[1] // 2:
            gpt_ans = "left"
        else:
            gpt_ans = "right"
        result += int(gpt_ans == direction)
        episode_success += int(gpt_ans == direction)
        masked_output = output.copy()
        masked_output[~mask[int(num_list[0])-1]['segmentation']] = [0, 0, 0]
        output_saved = Image.fromarray(masked_output)
        output_saved.save(
            os.path.join(output_folder, f"{name_without_suffix}_{direction}.jpg"))
    if episode_success == 2:
        episode_res += 1
    output_saved = Image.fromarray(output)
    output_saved.save(
        os.path.join(output_folder, f"{name_without_suffix}_mark.jpg")
    )
    gpt_response.append(gpt_response_item)

    # print(f"acc. = {result} / {2*len(data)} = {result/len(data)/2}")
    # print(f"episode_acc. = {episode_res} / {len(data)} = {episode_res/len(data)}")
    # print(f"som_error = {som_error}")
    # with open(os.path.join(output_folder, "gpt_response.json"), 'w') as f:
    #     json.dump(gpt_response, f)
