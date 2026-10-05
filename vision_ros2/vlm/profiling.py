"""Opt-in single-request timings; synchronize only at stage boundaries."""
import json
import time
import torch
from transformers import StoppingCriteria


class FirstTokenTimer(StoppingCriteria):
    def __init__(self, device):
        self.device = device
        self.finished_at = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.finished_at is None:
            if self.device != 'cpu':
                torch.cuda.synchronize()
            self.finished_at = time.monotonic()
            print(f'VLM_PROFILE first_token_ready threads={torch.get_num_threads()}', flush=True)
        return False


def synchronize(device):
    if device != 'cpu':
        torch.cuda.synchronize()


def report_profile(model, inputs, output, timer, started, prepared, transferred, finished, image):
    ids = output[0, inputs['input_ids'].shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    grid = inputs['image_grid_thw'].tolist()
    print('VLM_PROFILE ' + json.dumps(dict(
        input_size=list(image.size), image_count=len(grid), image_grid_thw=grid,
        processed_hw=[[g[1]*14, g[2]*14] for g in grid],
        visual_tokens=int((inputs['input_ids'] == model.config.image_token_id).sum()),
        prompt_tokens=inputs['input_ids'].shape[1], output_tokens=len(ids),
        ended_with_eos=bool(ids and ids[-1] in eos),
        preprocess_s=prepared-started, transfer_s=transferred-prepared,
        first_token_s=timer.finished_at-transferred,
        decode_s=finished-timer.finished_at, total_s=finished-started,
        torch_threads=torch.get_num_threads(), model_training=model.training,
        inference_mode=torch.is_inference_mode_enabled(), device_map=model.hf_device_map,
        do_sample=False, num_beams=model.generation_config.num_beams,
        use_cache=model.generation_config.use_cache,
    )), flush=True)
