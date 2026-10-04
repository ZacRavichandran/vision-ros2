#!/usr/bin/env python3

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    VipLlavaForConditionalGeneration,
)


class SupportedModels(Enum):
    Llava3PhiMini = "xtuner/llava-phi-3-mini-hf"
    VipLlava = "llava-hf/vip-llava-7b-hf"


def load_quantized(model_class, model_id: str, quantize: bool, device: str):
    """Load a VLM, optionally quantized to 4-bit NF4 with double quantization.

    Both are set explicitly: BitsAndBytesConfig defaults to plain FP4.
    """
    kwargs = {"dtype": torch.float16, "low_cpu_mem_usage": True}

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # A quantized model cannot be moved after loading, so place it here.
        kwargs["device_map"] = {"": device}

    model = model_class.from_pretrained(model_id, **kwargs)
    return model if quantize else model.to(device)


def load_processor(model_id: str, config):
    """Load a processor, filling in fields older checkpoints leave unset.

    transformers counts image tokens from patch_size and
    vision_feature_select_strategy; older checkpoints (e.g.
    xtuner/llava-phi-3-mini-hf) omit them, so copy them from the model config.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    if getattr(processor, "patch_size", None) is not None:
        return processor

    processor.patch_size = config.vision_config.patch_size
    strategy = getattr(config, "vision_feature_select_strategy", "default")
    processor.vision_feature_select_strategy = strategy
    # A CLIP tower contributes one extra token (CLS) under either strategy:
    # "default" drops it again, "full" keeps it. A checkpoint missing patch_size
    # cannot be trusted on this value either, so set it for every strategy --
    # a stale 0 under "full" is one token short and fails inside generate.
    processor.num_additional_image_tokens = 1
    return processor


class VLMWrapper:
    def __init__(
        self,
        model: Optional[str] = SupportedModels.VipLlava,
        classes="parking lot, sidewalk, road, park, other",
        quantize: Optional[bool] = None,
        device: str = "cuda:0",
    ) -> None:
        self.model = None

        if model == SupportedModels.Llava3PhiMini.value:
            self.model = LlavaPhi3(quantize=bool(quantize), device=device)
        elif model == SupportedModels.VipLlava.value:
            # 7B at fp16 is 14 GB of weights; quantized unless told otherwise.
            self.model = VipLlava(
                quantize=True if quantize is None else quantize, device=device
            )
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
    def __init__(self, quantize: bool = False, device: str = "cuda:0") -> None:
        model_id = "xtuner/llava-phi-3-mini-hf"
        self.device = device
        self.model = load_quantized(
            LlavaForConditionalGeneration, model_id, quantize, device
        )
        self.processor = load_processor(model_id, self.model.config)

    def infer(
        self, raw_prompt: str, raw_image: Image, crop: Optional[bool] = False
    ) -> str:
        prompt = self.format_prompt(prompt=raw_prompt)

        inputs = self.processor(
            text=prompt, images=raw_image, return_tensors="pt"
        ).to(self.device, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        formatted_output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        return formatted_output

    def format_prompt(self, prompt: str) -> str:
        return f"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"


class VipLlava:
    def __init__(self, quantize: bool = True, device: str = "cuda:0") -> None:
        model_id = "llava-hf/vip-llava-7b-hf"
        self.device = device
        self.model = load_quantized(
            VipLlavaForConditionalGeneration, model_id, quantize, device
        )

        self.processor = load_processor(model_id, self.model.config)

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

        inputs = self.processor(
            text=prompt, images=raw_image, return_tensors="pt"
        ).to(self.device, torch.float16)

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
