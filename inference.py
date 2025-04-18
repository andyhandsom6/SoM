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

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np

import os
from PIL import Image
from tqdm import tqdm

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

if __name__ == "__main__":
    input_path = "/mnt/navid/SoM/0_33/images"
    output_path = "/mnt/navid/SoM/0_33_mask"
    os.makedirs(output_path, exist_ok=True)
    woman_list = [
        9, 8, 9, 8, 6, 3, 6, 4, 5, 3,
        3, 3, 4, 5, 4, 4, 2, 3, 4, 2,
        2, 1, 2, 3, 3, 6, 5, 4, 2, 6,
        4, 6, 6, 3, 8, 4
    ]
    for _, _, files in os.walk(input_path):
        for i, file in tqdm(enumerate(files)):
            image_path = os.path.join(input_path, file)
            image = Image.open(image_path)
            output, mask = inference(image)
            save_path = os.path.join(output_path, file)
            masked_output = output.copy()
            masked_output[~mask[woman_list[i]-1]['segmentation']] = [0, 0, 0]
            output = Image.fromarray(masked_output)
            # import pdb
            # pdb.set_trace()
            output.save(save_path)