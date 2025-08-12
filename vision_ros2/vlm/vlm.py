#!/usr/bin/env python3

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    VipLlavaForConditionalGeneration,
)


class SupportedModels(Enum):
    Llava3PhiMini = "xtuner/llava-phi-3-mini-hf"
    VipLlava = "llava-hf/vip-llava-7b-hf"


class VLMWrapper:
    def __init__(
        self,
        model: Optional[str] = SupportedModels.VipLlava,
        classes="parking lot, sidewalk, road, park, other",
    ) -> None:
        self.model = None

        if model == SupportedModels.Llava3PhiMini.value:
            self.model = LlavaPhi3()
        elif model == SupportedModels.VipLlava.value:
            self.model = VipLlava()
        else:
            raise ValueError(f"{model} not supported.")

        self.classes = classes.split(",")

        self.scene_prompt = "You are a robot. Where are you currently standing? "

        for i, class_id in enumerate(self.classes):
            self.scene_prompt += f"({i+1}) {class_id}, "

        self.scene_prompt = self.scene_prompt[:-2] + ". "

        self.scene_prompt += "Focus on the nearest parts of the image. Provide a single number. Class id: "

        self.open_scene_prompt = (
            "You are a robot. Describe where you are so you can plan. "
            "Provide your answer as a noun with a short description. "
            "For example: empty sidewalk, road, park with trees and benches, empty parking lot, patio. Answer: "
        )

    def open_query(self, prompt: str, image: np.ndarray) -> str:
        print(image)
        img = Image.fromarray(image)
        output = self.model.infer(raw_prompt=prompt, raw_image=img, crop=False)

        return output

    def classify_scene(self, image: np.ndarray) -> Tuple[str, str]:
        img = Image.fromarray(image)
        output = self.model.infer(
            raw_prompt=self.scene_prompt, raw_image=img, crop=True
        )

        try:
            parsed = int(output.split("Class id:")[-1].strip())
            class_id = self.classes[parsed - 1]
        except:
            class_id = "unkown"

        return output, class_id

    def open_classify_scene(self, image: np.ndarray) -> Tuple[str, str]:
        img = Image.fromarray(image)
        output = self.model.infer(
            raw_prompt=self.open_scene_prompt, raw_image=img, crop=False
        )

        try:
            parsed = output.split("Answer:")[-1].strip()
        except:
            parsed = "unkown"

        return output, parsed


class LlavaPhi3:
    def __init__(self) -> None:
        model_id = "xtuner/llava-phi-3-mini-hf"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(
        self, raw_prompt: str, raw_image: Image, crop: Optional[bool] = False
    ) -> str:
        prompt = self.format_prompt(prompt=raw_prompt)
        # raw_image = Image.open(image_file)

        # raw_image = Image.open(requests.get(image_file, stream=True).raw)

        inputs = self.processor(prompt, raw_image, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        formatted_output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        return formatted_output

    def format_prompt(self, prompt: str) -> str:
        return f"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"


class VipLlava:
    def __init__(self,
                 device: str = "gpu") -> None:
        
        self.device = device

        print(f"Initializing VipLlava model on {device}...", flush=True)

        model_id = "llava-hf/vip-llava-7b-hf"
        if device == "gpu":
            self.model = VipLlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
            )
        elif device == "cpu":
            self.model = VipLlavaForConditionalGeneration.from_pretrained(
                model_id,
                low_cpu_mem_usage=False).to("cpu")
        else:
            raise ValueError(f"Unknown device type: {device}. Choose between 'cpu' and 'gpu'.")

        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(
        self, raw_prompt: str, raw_image: Image, crop: Optional[bool] = False
    ) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": raw_prompt,
                    },
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            chat_template="{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'][0]['text'] }}{% elif message['role'] == 'user' %}{{ '###Human: '}}{% else %}{{ '###' + message['role'].title() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '###Assistant:' }}{% endif %}",
        )

        # model seems to respond better when top half of image is cropped, but this should be checked
        size = raw_image.size
        raw_image = raw_image.resize((640, 480), resample=0)

        if crop:
            size = raw_image.size
            top_half = size[1] // 2
            raw_image = raw_image.crop((0, top_half, 640, 480))

        if self.device == "gpu":
            inputs = self.processor(prompt, raw_image, return_tensors="pt").to(
                0, torch.float16
            )
        else:
            inputs = self.processor(prompt, raw_image, return_tensors="pt").to("cpu")

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        formatted_output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        # model provides some extra formatting which we don't want
        formatted_output = (
            formatted_output.replace("###Assistant:", "")
            .replace("(", "")
            .replace(")", "")
            .replace("#", "")
        )
        return formatted_output
