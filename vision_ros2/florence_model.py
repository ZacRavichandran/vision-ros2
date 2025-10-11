from transformers import AutoProcessor, AutoModelForCausalLM
import supervision as sv
from PIL import Image

class FlorenceModel:
    def __init__(self, model_id='microsoft/Florence-2-large-ft'):

        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        self.bounding_box_annotator = sv.RoundBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.polygon_annotator = sv.PolygonAnnotator(color_lookup=sv.ColorLookup.INDEX)

    def preprocess_img_rgb(self, image, resize_dims=None):
        # Convert the input image (numpy array) to a PIL Image in RGB format
        img_PIL = Image.fromarray(image).convert("RGB")

        if resize_dims is not None:
            img_PIL = img_PIL.resize(resize_dims)

        return img_PIL

    def preprocess_img_depth(self, image, resize_dims=None):
        # Convert the input image (numpy array) to a PIL Image in grayscale format
        img_PIL = Image.fromarray(image).convert("I;16")

        if resize_dims is not None:
            img_PIL = img_PIL.resize(resize_dims)

        return img_PIL

    def generate_response(self, task_prompt, task_input=None, image_input=None, resize_dims=None):

        if task_input is None:
            final_prompt = task_prompt
        else:
            final_prompt = task_prompt + task_input

        image_input = self.preprocess_img_rgb(image_input, resize_dims=resize_dims)

        inputs = self.processor(text=final_prompt, images=image_input, return_tensors="pt")

        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=1
        )

        # transition_scores = self.model.compute_transition_scores(
        #     sequences=generated_ids["sequences"],
        #     scores=generated_ids["scores"],
        #     beam_indices=generated_ids["beam_indices"],
        #     normalize_logits=False
        # )

        # print(transition_scores, flush=True)

        # print(generated_ids["sequences"], flush=True)
        # print("--------------------------------------------------")
        # print(generated_ids["sequences_scores"], flush=True)
        # print("----------------------------------------------------")
        # print(generated_ids["scores"], flush=True)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # print(generated_text)

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_input.width, image_input.height)
        )

        # parsed_answer = self.processor.post_process_generation(
        #     text=generated_ids["sequences"][0],
        #     task=task_prompt,
        #     image_size=(image_input.width, image_input.height)
        # )

        annotated_img = self.plot_bbox(parsed_answer, image_input)

        bboxes = parsed_answer[task_prompt]['bboxes']
        labels = parsed_answer[task_prompt]['labels']

        return (annotated_img, labels, bboxes)

    def plot_bbox(self, output_vlm_generation, image, plot_polygon=False):

        image_detections = sv.Detections.from_vlm(sv.VLM.FLORENCE_2, output_vlm_generation, resolution_wh=image.size)

        if not plot_polygon:
            image = self.bounding_box_annotator.annotate(image, image_detections)
        else:
            image = self.polygon_annotator.annotate(image, image_detections)
        image = self.label_annotator.annotate(image, image_detections)

        return image