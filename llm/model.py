import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


class ExtractionModel:
    def __init__(self) -> None:
        self.model_id = "microsoft/Florence-2-large"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    @property
    def model_config(self):
        model_config = {
            "model_id": self.model_id,
            "model_size": f"{self.model.get_memory_footprint()/1e+6} MB",
        }
        return model_config

    def run(self, image, text_input=None):
        if text_input is None:
            prompt = "<MORE_DETAILED_CAPTION>"
        else:
            prompt = prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(torch.int64).cuda(),
            pixel_values=inputs["pixel_values"].half().cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        return parsed_answer
