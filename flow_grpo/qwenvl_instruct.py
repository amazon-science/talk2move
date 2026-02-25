import torch
import base64
import mimetypes
from io import BytesIO
from PIL import Image, ImageOps
from openai import OpenAI
import json
import time
import re
import numpy as np
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from orient_anything.paths import *
from orient_anything.vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor


import torch.nn.functional as F
from orient_anything.utils import *
from orient_anything.inference import *

from lang_sam import LangSAM



def file_to_data_url_half(image, out_format="JPEG", quality=85) -> str:
    # Read and correct EXIF orientation
    img = ImageOps.exif_transpose(image).convert("RGB")
    w, h = img.size
    nw, nh = max(1, w ), max(1, h )

    # PIL 10+ uses Image.Resampling.LANCZOS
    img_small = img.resize((nw, nh), Image.Resampling.LANCZOS)

    buf = BytesIO()
    if out_format.upper() == "PNG":
        img_small.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img_small.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"



# vlm_prompt = """
# You are an image-editing evaluator.
# You will be given:
# 1. An original image.
# 2. An editing instruction (prompt).
# 3. An edited image.

# Decide ONLY whether the edited image CORRECTLY follows the instruction.

# Decision rule (must all be satisfied to be correct):
# - The correct target object is present and correctly identified.
# - It has been moved to the specified target location and removed from the original location.
# - The placement looks natural (size, perspective, lighting, shadows reasonably consistent; minor imperfections allowed).
# - The background at the original location is plausibly filled without glaring artifacts.
# - No unintended major changes occurred elsewhere in the image.

# Output Format:
# Return ONLY strict JSON with a single boolean field and NOTHING else (no prose, no code fences):

# {"correct": true}

# Use true if and only if all conditions above are met; otherwise use false.
# """



class QwenVLScorer:
    def __init__(self, device):
        self.device = device
        openai_api_key = "EMPTY"
        openai_api_base = "http://your-api-server-ip:8001/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_id = 'Qwen/Qwen3-VL-235B-A22B-Instruct'
        # self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
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

        self.verify_prompt = """
            You are a strict visual fact-checker.

            You will be given:
            1) a source image
            2) an edited image
            3) a short post-edit description (e.g., "the blue basket is on top of the washing machine").

            Task:
            Decide whether the description is visually true in the edited image compared with the source image.


            Decision rules:
            - Verify the described OBJECT(S) exist and are correctly identified (type/color/attributes).
            - Verify the described SPATIAL / RELATIONAL conditions STRICTLY hold:
              * top/on/in/under/inside/next to/near/beside/behind/in front of
              * between A and B; A left/right of B; hierarchical relations (A on B while B left of C)
              * interactive (A occludes B but not C; A touches B without overlapping C)
            - Verify SIZE: object is larger/smaller/approximately equal to what is described, including relative size comparisons.
            - Verify ROTATION: object orientation (e.g., facing left/right, tilted, angled) matches the description.
            - Verify POSE/ARTICULATION: if the object has joints or movable parts (e.g., chair backrest reclined, mannequin arm raised, lamp head tilted), check that the pose matches what is described.
            - If the description is ambiguous, not clearly supported by the image, or contradicted by the image, treat it as NOT MATCHED.
            - Ignore minor, irrelevant details (lighting, small artifacts).

            Confidence guidelines (float between 0.0 and 1.0):
            - 0.9–1.0: Very clear and unambiguous evidence, almost certain.
            - 0.7–0.89: Mostly supported but with slight uncertainty.
            - 0.4–0.69: Unclear or ambiguous evidence, cannot be sure.
            - 0.1–0.39: Evidence mostly contradicts the description.
            - 0.0–0.09: Clearly false, definitely not matched.

            Output ONLY strict JSON with three fields and NOTHING else:
            {
              "match": true|false,
              "confidence": float between 0.0 and 1.0,
              "explanation": "short justification of your decision"
            }
        """

    def object_detection(self, image, prompt):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.4,
            target_sizes=[image.size[::-1]]
        )
        return [box.tolist() for box in results[0]["boxes"]] if results else []

    def qwenvl_detection(self, image0_path, categories):
        format_prompt = """
        Response in STRICT json format:
            {
          'category_1': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
          'category_2': [[x1, y1, x2, y2], ...]
                ...
            }
            For each category, provide a list of bounding boxes of all its instances in the image.
            Each bounding box must correspond to a single, distinct individual object — never a group or collection.
            Strictly follow this instruction without exceptions or interpretation.
            Strictly follow the format in English, without any irrelevant words.
        """
        start_time = time.time()
        data_url0 = file_to_data_url_half(image0_path)
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"You are required to detect all the instances of the following categories {categories} in the image."+format_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url0
                            }
                        },
                    ]
                }
            ],
        )

        output = completion.choices[0].message.content
        result = {}
        ok = False
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                ok = True
            except json.JSONDecodeError as e:
                print(output)
                print("Still invalid JSON in detection:", e)
        else:
            print("No JSON block found in output")

        return ok, result

    
    def call_qwenvl(self, image0_path, image1_path, edit_instruction):
        start_time = time.time()
        # data_url0 = file_to_data_url_half(image0_path)
        data_url1 = file_to_data_url_half(image1_path)
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url0
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url1
                            }
                        },
                        {
                            "type": "text",
                            "text": f"{self.verify_prompt}\n\nDescription: {edit_instruction}"
                        },
                        
                    ]
                }
            ],
        )

        # print(completion.choices[0].message)
        output = completion.choices[0].message.content
        # print(output)
        result = {}
        # Regex to find JSON object
        ok = False
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                # print("Clean parsed result:", result)
                ok = True
            except json.JSONDecodeError as e:
                print("Still invalid JSON:", e)
        else:
            print("No JSON block found in output")

        # print('Time Usage:', time.time()-start_time)
        return ok, result

    def qwenvl_rotation(self, image0_path, image1_path):
        prompt = """
            You are a strict visual pose-change verifier.

            INPUT:
            - One cropped source image of an object.
            - One cropped edited image of the same object.

            TASK:
            Verify how the object in the source image has rotated to become the object in the edited image.

            Rules:
            - Identify the main rotation axis: x-axis, (this is uncommon, often is plane head, monitor head, fan head rolling left and right in the y-z plane), y-axis (tilt up and down), or z-axis (most common)
            - Decide the rotation direction: clockwise or anticlockwise.
            - Estimate the rotation angle in degrees, you can only choose in [45|90|135|180], use the more likely option.
            - Be consistent: assume a right-handed coordinate system.
            - If uncertain, provide the most plausible estimate.

            OUTPUT FORMAT (STRICT JSON ONLY):
            {
            "axis": "x|y|z",
            "direction": "clockwise|anticlockwise",
            "angle": [45|90|135|180],  
            "explanation": "short justification of your decision",
            }
            """
        start_time = time.time()
        data_url0 = file_to_data_url_half(image0_path)
        data_url1 = file_to_data_url_half(image1_path)
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url0
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url1
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },

                    ]
                }
            ],
        )

        output = completion.choices[0].message.content
        result = {}
        ok = False
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # print(output)
            try:
                result = json.loads(json_str)
                ok = True
            except json.JSONDecodeError as e:
                print(output)
                print("Still invalid JSON in rotation:", e)
                
        else:
            print("No JSON block found in output")
            print(output)
        # print('Time Usage:', time.time()-start_time)
        # print(result)
        return ok, result

    def verify_postedit(self, image0_path, image1_path, postedit_desp):
        start_time = time.time()
        data_url0 = file_to_data_url_half(image0_path)
        data_url1 = file_to_data_url_half(image1_path)
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url0
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url1
                            }
                        },
                        {
                            "type": "text",
                            "text": f"{self.verify_prompt}\nDescription: {postedit_desp}"
                        },
                        
                    ]
                }
            ],
        )

        # print(completion.choices[0].message)
        output = completion.choices[0].message.content
        # print(output)
        result = {}
        # Regex to find JSON object
        ok = False
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                # print("Clean parsed result:", result)
                ok = True
            except json.JSONDecodeError as e:
                print("Still invalid JSON:", e)
        else:
            print("No JSON block found in output")

        # print('Time Usage:', time.time()-start_time)
        return ok, result

    def run(self, images, ref_images, prompts, metadatas):
        # print('in qwenvl run')
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            metadata['postedit_desp'] = metadata['postedit_desp']#.replace('directly','')
            ok, result = self.verify_postedit(ref_image, image, metadata['postedit_desp'])
            total_score = result['confidence'] * int(result['match'])
            print(result)
            scores.append(total_score)
        scores = torch.tensor(scores, device=self.device)
        return scores
    def orient_anything(self, image0_path, image1_path):
        # img1 = "/path/to/data"
        # img2 = "/path/to/data"
        start_time = time.time()
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
            print(time.time()-start_time)
            start_time = time.time()

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

    def evaluate_rotation(self, images, ref_images, prompts, metadatas):
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            # metadata['postedit_desp'] = metadata['postedit_desp']#.replace('directly','')
            # ok, result = self.verify_postedit(ref_image, image, metadata['postedit_desp'])
            # total_score = result['confidence'] * int(result['match'])
            # print(result)
            # scores.append(total_score)

            objectname = metadata['object']
            # imagepath = metadata['image']
            ref_image = ref_image.resize(image.size)
            image_h, image_w = image.size
            
            # ok, results_input = self.qwenvl_detection(ref_image, [objectname])

            # if objectname in results_input.keys() and len(results_input[objectname])>0:
            #     boxes_input = results_input[objectname]
            #     try:
            #         source_image_crop = ref_image.crop(boxes_input[0])
            #     except:
            #         print('boxes_input',boxes_input)
            # else:
            #     source_image_crop = None
            objectname = objectname+'. '
            # boxes_input = self.object_detection(ref_image, [objectname]) 
            # if len(boxes_input) > 0:
            #     source_image_crop = ref_image.crop(boxes_input[0])
            # else:
            #     source_image_crop = None

            # boxes_output = self.object_detection(image, [objectname])
            # if len(boxes_output) > 0:
            #     edit_image_crop = image.crop(boxes_output[0])
            # else:
            #     edit_image_crop = None
            
            masks_input = self.seg_model.predict([ref_image], [prompt])
            masks_output = self.seg_model.predict([image], [prompt])

            if (len(masks_input[0]['masks']) == 0) or (len(masks_output[0]['masks']) == 0): 
                source_image_crop, edit_image_crop = None, None
                
            else:
                boxes_input = masks_input[0]['boxes'][0]
                boxes_output = masks_output[0]['boxes'][0]
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

            # _, results_output = self.qwenvl_detection(image, [objectname])
            # if objectname in results_output.keys() and len(results_output[objectname])>0:
            #     boxes_output = results_output[objectname]
            #     try:
            #         edit_image_crop = image.crop(boxes_output[0])
            #     except:
            #         print('boxes_output',boxes_output)
            # else:
            #     edit_image_crop = None

            score = 0
            if source_image_crop is not None and edit_image_crop is not None:
                _, edit_result = self.orient_anything(source_image_crop, edit_image_crop)
                
                mapping = {'psi':'z','phi':'x', 'theta':'y','x':'x','y':'y','z':'z'}
                # print(mapping[metadata['rotation_axis']] )
                

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
                print(edit_result, metadata['rotation_axis'], metadata['rotation_direction'], metadata["rotation_angle"])
                # score = axis_correct * direction_correct * (1 - np.abs(edit_result["angle"] - metadata["rotation_angle"]) / 180) 

                
            

            scores.append(score/3)
        scores = torch.tensor(scores, device=self.device)
        return scores
