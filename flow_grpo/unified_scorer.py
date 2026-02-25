import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from flow_grpo.clip_score import CLIPScore
# from flow_grpo.qwen_caller import qwenvl_detection 
from lang_sam import LangSAM
from orient_anything.paths import *
from orient_anything.vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor


import torch.nn.functional as F
from orient_anything.utils import *
from orient_anything.inference import *


class UnifiedScorer:
    def __init__(self, device):
        self.device = device
        # self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        # self.clip_model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14")
        self.POSITION_THRESHOLD = 0.1

        self.seg_model = LangSAM()
        ckpt_path = '/path/to/data'
        save_path = './'
        self.dino = DINOv2_MLP(
                            dino_mode   = 'large',
                            in_dim      = 1024,
                            out_dim     = 360+180+360+2,
                            evaluate    = True,
                            mask_dino   = False,
                            frozen_back = False
                        )

        self.dino.eval()
        # print('model create')
        self.dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.dino = self.dino.to(device)
        print('weight loaded')
        self.val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')

    # def object_detection(self, image, prompt):
    #     inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     results = self.processor.post_process_grounded_object_detection(
    #         outputs,
    #         inputs.input_ids,
    #         threshold=0.3,
    #         text_threshold=0.4,
    #         target_sizes=[image.size[::-1]]
    #     )
    #     return [box.tolist() for box in results[0]["boxes"]] if results else []

    def relative_position(self, box_a, box_b, relation):
        boxes = np.array([box_a, box_b])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return 0
        dx, dy = revised_offset / np.linalg.norm(offset)
        if (dx < -0.01 and 'right' in relation) or (dx > 0.01 and 'left' in relation) :
            return np.abs(dx)
        elif (dy < -0.01 and 'down' in relation) or (dy > 0.01 and 'up' in relation):
            return np.abs(dy)
        return 0

    def orient_anything(self, image0_path, image1_path):
        # img1 = "/path/to/data"
        # img2 = "/path/to/data"
        # start_time = time.time()
        z_angles, y_angles, x_angles = [], [], []
        for origin_image in [image0_path, image1_path]:
            # origin_image = Image.open(image_path).convert('RGB')
            angles = get_3angle(origin_image, self.dino, self.val_preprocess, self.device)
            azimuth     = float(angles[0])
            polar       = float(angles[1])
            rotation    = float(angles[2])
            confidence  = float(angles[3])
            z_angles.append(azimuth)
            y_angles.append(polar)
            x_angles.append(rotation)
            # print(azimuth, polar, rotation)
            # print(time.time()-start_time)
            # start_time = time.time()

        angles_z, angles_y, angles_x = {}, {}, {}
        angles_z['clockwise'] = (z_angles[1] - z_angles[0]) % 360
        angles_z['counterclockwise'] = (z_angles[0] - z_angles[1]) % 360
        direction_z = min(angles_z, key=angles_z.get)
        angle_z = angles_z[direction_z]

        angles_y['clockwise'] = np.abs(y_angles[1] - y_angles[0])
        angles_y['counterclockwise'] = np.abs(y_angles[0] - y_angles[1])
        direction_y = min(angles_y, key=angles_y.get)
        angle_y = angles_y[direction_y]

        angles_x['clockwise'] = np.abs(x_angles[1] - x_angles[0])
        angles_x['counterclockwise'] = np.abs(x_angles[0] - x_angles[1])
        direction_x = min(angles_x, key=angles_x.get)
        angle_x = angles_x[direction_x]
        
        angles = {'z':angle_z, 'y':angle_y, 'x':angle_x}
        directions = {'z':direction_z,'y':direction_y,'x': direction_x}
        axis = max(angles, key=angles.get)
        outputs = {'axis':axis, 'direction':directions[axis], 'angle':angles[axis]}
        total_outputs = {'z':[direction_z, angle_z], 'y':[direction_y, angle_y], 'x':[direction_x, angle_x]}
        return total_outputs, outputs



        
    @torch.no_grad()
    def run(self, images, ref_images, prompts, metadatas):
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            # sim_overall = self.image_consistency(ref_image, image)
            # sim_overall = int(sim_overall >= 0.9)
            # l1_distance = self.l1_distance(ref_image, image)
            correct, error_msg = 0, ""
            objectname = metadata['object']
            ref_image = ref_image.resize(image.size)
            image_h, image_w = image.size
            # objectname = [objectname+'. ']
            objectname = objectname+'. '

            masks_input = self.seg_model.predict([ref_image], [objectname])
            masks_output = self.seg_model.predict([image], [objectname])


            if len(masks_input[0]['masks']) == 0: 
                correct = 0
            elif len(masks_output[0]['masks']) == 0:
                # axis_correct, angle_correct, direction_correct = 0, 0, 0
                correct = 0
            else:
                boxes_input = masks_input[0]['boxes'][0]
                boxes_output = masks_output[0]['boxes'][0]

            if (len(masks_input[0]['masks']) == 0) or (len(masks_output[0]['masks']) == 0): 
                error_msg = "Missing boxes."
                correct = 0
                source_image_crop, edit_image_crop = None, None
                scores.append(correct)
                continue
        
            if 'translation' in metadata['tag']:
                correct = self.relative_position(boxes_input, boxes_output, metadata['direction'])
            elif 'rotation' in metadata['tag']:
                image1 = ref_image
                image2 = image
                # print(masks_input[0]['masks'].shape, masks_output[0]['masks'].shape, image_h, image_w)
                mask1 = masks_input[0]['masks'][0].reshape((image_h, image_w))
                mask1 = np.stack([mask1]*3, axis=-1)
                mask2 = masks_output[0]['masks'][0].reshape((image_h, image_w))
                mask2 = np.stack([mask2]*3, axis=-1)
                image1_masked = np.array(image1)
                image1_masked[mask1==0] = 255
                image1_masked = Image.fromarray(image1_masked.astype(np.uint8))
                image2_masked = np.array(image2)
                image2_masked[mask2==0] = 255
                image2_masked = Image.fromarray(image2_masked.astype(np.uint8))
                # print(boxes_input, boxes_output, boxes_input2, boxes_output2)
                source_image_crop = image1_masked.crop(boxes_input)
                edit_image_crop = image2_masked.crop(boxes_output)

                score = 0
                if source_image_crop is not None and edit_image_crop is not None:
                    _, edit_result = self.orient_anything(source_image_crop, edit_image_crop)
                    
                    mapping = {'psi':'z','phi':'x', 'theta':'y','x':'x','y':'y','z':'z'}
                    
                    if metadata["rotation_angle"] == 180:
                        edit_result["direction"] = metadata["rotation_direction"]
                    axis_correct = (mapping[metadata['rotation_axis']] == edit_result["axis"]) 
                    direction_correct = (metadata["rotation_direction"] == edit_result["direction"])
                    # angle_correct = (edit_result["angle"] == metadata["rotation_angle"])

                    score = int(mapping[metadata['rotation_axis']] == edit_result["axis"]) \
                        + int(metadata["rotation_direction"] == edit_result["direction"]) \
                        - int(metadata["rotation_direction"] == edit_result["direction"]) * (np.abs(edit_result["angle"] - metadata["rotation_angle"]) / 180 )

                    if np.abs(edit_result["angle"]) < 10:
                        score = 0
                    # print(edit_result, metadata['rotation_axis'], metadata['rotation_direction'], metadata["rotation_angle"])
                correct = score / 3
            elif 'resize' in metadata['tag']:
                x1_min, y1_min, x1_max, y1_max = boxes_input
                x2_min, y2_min, x2_max, y2_max = boxes_output
                w1, h1 = x1_max - x1_min, y1_max - y1_min
                w2, h2 = x2_max - x2_min, y2_max - y2_min
                x_c1, y_c1 = (x1_max + x1_min) / (w1+w2), (y1_max + y1_min) / (h1+h2)
                x_c2, y_c2 = (x2_max + x2_min) / (w1+w2), (y2_max + y2_min) / (h1+h2)
                
                if 'bigger' in prompt:
                    w_ratio = w2 / w1
                    h_ratio = h2 / h1
                    correct = 1.0
                    correct -= (np.abs(w_ratio - metadata['ratio']) + np.abs(h_ratio - metadata['ratio'])) /  (2 * metadata['ratio'])
                    
                elif 'smaller' in prompt:
                    w_ratio = w1 / w2
                    h_ratio = h1 / h2
                    correct = 1.0
                    correct -= (np.abs(w_ratio - metadata['ratio']) + np.abs(h_ratio - metadata['ratio'])) /  (2 * metadata['ratio'])

                else:
                    w_ratio = 0
                    h_ratio = 0
                    correct = 0
            else:
                print(f'Undefined Type: {metadata}')
                correct = 0

            scores.append(correct)
        scores = torch.tensor(scores, device=self.device)
        return scores
                # if sim_obj < 0.75:
                #     correct, error_msg = 0, "Object mismatch."
                # if sim_old > 0.95 or result['Source Location Cleared'] == 0:
                #     correct, error_msg = 0, "Original object not removed."


       
