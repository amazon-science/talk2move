import torch
from PIL import Image
from editscore import EditScore

# Load the EditScore model. It will be downloaded automatically.
# Replace with the specific model version you want to use.
# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# lora_path = "EditScore/EditScore-7B"

# input_image = Image.open("example_images/input.png")
# output_image = Image.open("example_images/output.png")
# instruction = "Adjust the background to a glass wall."


class EditScorer:
    def __init__(self, device):
        self.device = device
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        lora_path = "EditScore/EditScore-7B"
        self.score_range = 25
        self.scorer = EditScore(
            backbone="qwen25vl", # set to "qwen25vl_vllm" for faster inference
            model_name_or_path=model_path,
            enable_lora=True,
            lora_path=lora_path,
            score_range=self.score_range,
            num_pass=1, # Increase for better performance via self-ensembling
        )
    @torch.no_grad()
    def run(self, images, ref_images, prompts, metadatas):
        scores = []
        for image, ref_image, prompt, metadata in zip(images, ref_images, prompts, metadatas):
            result = self.scorer.evaluate([ref_image, image], prompt)
            total_score = result['overall'] / self.score_range
            scores.append(total_score)
        scores = torch.tensor(scores, device=self.device)
        return scores


