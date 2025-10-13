from transformers import AutoProcessor, AutoModelForCausalLM
import supervision as sv
from PIL import Image
import numpy as np

class FlorenceModel:
    def __init__(self, model_id='microsoft/Florence-2-large', detection_conf=0.3):

        self.model_id = model_id
        self.detection_conf = detection_conf
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
        num_beams=3,
        output_scores=True,
        return_dict_in_generate=True
        )

        prediction, scores, beam_indices = generated_ids.sequences, generated_ids.scores, generated_ids.beam_indices

        transition_scores = self.model.compute_transition_scores(
            sequences=prediction,
            scores=scores,
            beam_indices=beam_indices,
        )

        parsed_answer = self.processor.post_process_generation(
            sequence=prediction[0],
            transition_beam_score=transition_scores[0],
            task=task_prompt,
            image_size=(image_input.width, image_input.height)
        )

        # parsed_answer = self.processor.post_process_generation(
        #     text=generated_ids["sequences"][0],
        #     task=task_prompt,
        #     image_size=(image_input.width, image_input.height)
        # )

        if "scores" in parsed_answer[task_prompt].keys():

            valid_detections = np.array(parsed_answer[task_prompt]["scores"]) > self.detection_conf
            parsed_answer[task_prompt]["bboxes"] = np.array(parsed_answer[task_prompt]["bboxes"])[valid_detections]
            parsed_answer[task_prompt]["labels"] = np.array(parsed_answer[task_prompt]["labels"])[valid_detections]
            parsed_answer[task_prompt]["scores"] = np.array(parsed_answer[task_prompt]["scores"])[valid_detections]

            scores_strings = np.array([f" ({score:.2f})" for score in parsed_answer[task_prompt]["scores"]])

            if len(scores_strings) > 0:
                non_merged_labels = parsed_answer[task_prompt]["labels"]
                parsed_answer[task_prompt]["labels"] = np.char.add(np.array(parsed_answer[task_prompt]["labels"]), scores_strings)
            else:
                return (image_input, None, None, None)
        else:
            return (image_input, None, None, None)

        annotated_img = self.plot_bbox(parsed_answer, image_input)

        bboxes = parsed_answer[task_prompt]['bboxes']
        scores = parsed_answer[task_prompt]['scores']

        return (annotated_img, non_merged_labels, bboxes, scores)

    def plot_bbox(self, output_vlm_generation, image, plot_polygon=False):

        image_detections = sv.Detections.from_vlm(sv.VLM.FLORENCE_2, output_vlm_generation, resolution_wh=image.size)

        if not plot_polygon:
            image = self.bounding_box_annotator.annotate(image, image_detections)
        else:
            image = self.polygon_annotator.annotate(image, image_detections)
        image = self.label_annotator.annotate(image, image_detections)

        return image