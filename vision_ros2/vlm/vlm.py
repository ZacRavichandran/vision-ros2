#!/usr/bin/env python3

from enum import Enum
import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    VipLlavaForConditionalGeneration,
)


class SupportedModels(Enum):
    Llava3PhiMini = "xtuner/llava-phi-3-mini-hf"
    VipLlava = "llava-hf/vip-llava-7b-hf"
    Qwen25VL = "Qwen/Qwen2.5-VL-3B-Instruct"


class VLMWrapper:
    def __init__(
        self,
        model: Optional[str] = SupportedModels.VipLlava.value,
        classes="parking lot, sidewalk, road, park, other",
        quantization: str = "4bit",
        device: str = "cuda",
    ) -> None:
        self.model = None

        if model == SupportedModels.Llava3PhiMini.value:
            self.model = LlavaPhi3(quantization=quantization, device=device)
        elif model == SupportedModels.VipLlava.value:
            self.model = VipLlava(quantization=quantization, device=device)
        elif model == SupportedModels.Qwen25VL.value:
            self.model = Qwen25VL(quantization=quantization, device=device)
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
        # print(image)
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


def _model_load_kwargs(quantization: str, device: str) -> dict:
    """Build memory-safe Hugging Face loading options.

    CUDA_VISIBLE_DEVICES is intentionally respected: ``cuda:0`` means the first
    GPU visible to this process, not necessarily physical GPU 0.
    """

    if device == "cpu":
        return {"device_map": "cpu", "torch_dtype": torch.float32}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")

    kwargs = {
        "device_map": {"": 0},
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization not in ("none", "fp16"):
        raise ValueError("quantization must be one of: 4bit, 8bit, fp16, none")
    return kwargs


def _move_inputs(inputs, device: str):
    target = "cpu" if device == "cpu" else "cuda:0"
    return inputs.to(target)


class LlavaPhi3:
    def __init__(self, quantization: str = "4bit", device: str = "cuda") -> None:
        model_id = "xtuner/llava-phi-3-mini-hf"
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            **_model_load_kwargs(quantization, device),
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Some LLaVA checkpoints omit processor metadata required by newer
        # transformers releases. Copy it from the loaded vision config.
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = self.model.config.vision_config.patch_size
        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            self.processor.vision_feature_select_strategy = getattr(
                self.model.config, "vision_feature_select_strategy", "default"
            )
        # This checkpoint uses a CLIP vision tower with a CLS token, but its
        # processor metadata predates this Transformers field.  Without the
        # additional token, LlavaProcessor expands <image> to 575 placeholders
        # while the model produces 576 patch features.
        if not getattr(self.processor, "num_additional_image_tokens", 0):
            self.processor.num_additional_image_tokens = 1

    def infer(
        self, raw_prompt: str, raw_image: Image, crop: Optional[bool] = False
    ) -> str:
        prompt = self.format_prompt(prompt=raw_prompt)
        # raw_image = Image.open(image_file)

        # raw_image = Image.open(requests.get(image_file, stream=True).raw)

        inputs = _move_inputs(self.processor(images=raw_image, text=prompt, return_tensors="pt"), self.device)
        prompt_length = inputs["input_ids"].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        formatted_output = self.processor.decode(output[0][prompt_length:], skip_special_tokens=True)
        return formatted_output

    def format_prompt(self, prompt: str) -> str:
        return self.processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': '<image>\n' + prompt}],
            tokenize=False, add_generation_prompt=True)


class VipLlava:
    def __init__(self, quantization: str = "4bit", device: str = "cuda") -> None:
        self.device = device

        print(f"Initializing VipLlava model on {device}...", flush=True)

        model_id = "llava-hf/vip-llava-7b-hf"
        self.model = VipLlavaForConditionalGeneration.from_pretrained(
            model_id, **_model_load_kwargs(quantization, device)
        )

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

        inputs = _move_inputs(
            self.processor(raw_image, prompt, return_tensors="pt"), self.device
        )

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        formatted_output = self.processor.decode(output[0][prompt_length:], skip_special_tokens=True)
        # model provides some extra formatting which we don't want
        formatted_output = (
            formatted_output.replace("###Assistant:", "")
            .replace("(", "")
            .replace(")", "")
            .replace("#", "")
        )
        return formatted_output


class Qwen25VL:
    def __init__(self, quantization: str = "4bit", device: str = "cuda") -> None:
        self.device = device
        self.profile = False
        model_id = SupportedModels.Qwen25VL.value
        print(f"Initializing Qwen2.5-VL model on {device}...", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **_model_load_kwargs(quantization, device)
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(
        self, raw_prompt: str, raw_image: Image, crop: Optional[bool] = False
    ) -> str:
        from .profiling import FirstTokenTimer, synchronize, report_profile
        if self.profile:
            synchronize(self.device)
        started = time.monotonic()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": raw_prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[raw_image], padding=True, return_tensors="pt"
        )
        prepared = time.monotonic()
        inputs = _move_inputs(inputs, self.device)
        options = {}
        if self.profile:
            synchronize(self.device)
            timer = FirstTokenTimer(self.device)
            options['stopping_criteria'] = StoppingCriteriaList([timer])
        transferred = time.monotonic()
        if self.profile:
            print(f'VLM_PROFILE generation_start threads={torch.get_num_threads()} preprocessing_s={prepared-started:.3f} grid={inputs["image_grid_thw"].tolist()}', flush=True)
        token_limit = 8 if raw_prompt.startswith('CLUE_DIRECTION_V1\n') else 200
        output = self.model.generate(**inputs, max_new_tokens=token_limit, do_sample=False, num_beams=1, use_cache=True, **options)
        if self.profile:
            synchronize(self.device)
            report_profile(self.model, inputs, output, timer, started, prepared,
                           transferred, time.monotonic(), raw_image)
        prompt_length = inputs["input_ids"].shape[1]
        generated = output[0, prompt_length:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if len(generated) >= token_limit and int(generated[-1]) not in eos:
            raise ValueError("VLM output reached token limit without EOS; observation is unknown")
        return self.processor.batch_decode(
            output[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
