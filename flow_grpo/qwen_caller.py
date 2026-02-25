import os
import base64
import mimetypes
from io import BytesIO
from PIL import Image, ImageOps
from openai import OpenAI
import json
import time
import re

def file_to_data_url_half(path, out_format="JPEG", quality=85) -> str:
    # Read and correct EXIF orientation
    img = ImageOps.exif_transpose(path).convert("RGB")
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




openai_api_key = "EMPTY"
openai_api_base = "http://your-api-server-ip:8001/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_id = 'Qwen/Qwen3-VL-235B-A22B-Instruct'

verify_prompt = """
You are a strict visual fact-checker.

You will be given:
1) a single image
2) a short post-edit description (e.g., "the blue basket is on top of the washing machine").

Task:
Decide whether the description is visually true in the image.

Decision rules:
- Verify the described OBJECT(S) exist and are correctly identified (type/color/attributes).
- Verify the described SPATIAL / RELATIONAL conditions hold:
  * top/on/in/under/inside/next to/near/beside/behind/in front of
  * between A and B; A left/right of B; hierarchical relations (A on B while B left of C)
  * attribute–compositional (A larger/smaller than C)
  * interactive (A occludes B but not C; A touches B without overlapping C)
- If the description is ambiguous, not clearly supported by the image, or contradicted by the image, treat it as NOT MATCHED.
- Ignore minor, irrelevant details (lighting, small artifacts).

Output ONLY strict JSON with a single boolean field, an text string explanation and NOTHING else:
{"match": true, "explanation":...}
or
{"match": false, "explanation":...}
"""

def get_bbox_prompt(categories):
    bbox_prompt = """
    "You are required to detect all the instances of the following categories {categories} in the image."
        "Response in json format:"
        "{"
        "'category_1': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],"
        "'category_2': [[x1, y1, x2, y2], ...]"
        "..."
        "}"
        "For each category, provide a list of bounding boxes of all its instances in the image."
        "Each bounding box must correspond to a single, distinct individual object — never a group or collection."
        "Strictly follow this instruction without exceptions or interpretation."
        "Strictly follow the format in English, without any irrelevant words."
    """
    return bbox_prompt


def qwenvl_detection(image0_path, categories):
    start_time = time.time()
    data_url0 = file_to_data_url_half(image0_path)
    task_prompt = get_bbox_prompt(categories)
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are required to detect all the instances of the following categories {categories} in the image."
                            "Response in json format:"
                            "{"
                            "'category_1': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],"
                            "'category_2': [[x1, y1, x2, y2], ...]"
                            "..."
                            "}"
                            "For each category, provide a list of bounding boxes of all its instances in the image."
                            "Each bounding box must correspond to a single, distinct individual object — never a group or collection."
                            "Strictly follow this instruction without exceptions or interpretation."
                            "Strictly follow the format in English, without any irrelevant words."
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
            print("Still invalid JSON:", e)
    else:
        print("No JSON block found in output")

    # print('Time Usage:', time.time()-start_time)
    # print(result)
    return ok, result

def call_qwenvl(image0_path, image1_path, edit_instruction):
    start_time = time.time()
    data_url0 = file_to_data_url_half(image0_path)
    data_url1 = file_to_data_url_half(image1_path)
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": data_url0
                    #     }
                    # },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url1
                        }
                    },
                    {
                        "type": "text",
                        "text": f"{verify_prompt}\nDescription: {edit_instruction}"
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
    # print(result)
    return ok, result

