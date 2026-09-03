from __future__ import annotations

import json
from collections.abc import Callable

from .schema import TASKS


def parse_structured_labels(payload: str | dict) -> dict[str, dict]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if set(data) != set(TASKS):
        missing, extra = set(TASKS) - set(data), set(data) - set(TASKS)
        raise ValueError(f"invalid tasks; missing={sorted(missing)}, extra={sorted(extra)}")
    result = {}
    for task, item in data.items():
        label = int(item["label"])
        confidence = float(item["confidence"])
        if label not in (0, 1) or not 0 <= confidence <= 1:
            raise ValueError(f"invalid output for {task}")
        result[task] = {"label": label, "confidence": confidence}
    return result


def llm_prompt(caption: str) -> str:
    tasks = ", ".join(TASKS)
    return (
        "Infer only what the caption explicitly supports. Return strict JSON with exactly these keys: "
        f"{tasks}. Each value is {{\"label\": 0 or 1, \"confidence\": 0..1}}. "
        "Low confidence is required when absence is merely unstated. Caption: " + json.dumps(caption)
    )


def label_with_llm(caption: str, completion: Callable[[str], str]) -> dict[str, dict]:
    """Provider-neutral injection point; callers persist model/version metadata."""
    return parse_structured_labels(completion(llm_prompt(caption)))


CLIP_PROMPTS = {
    "multiple_subjects": ("an image with multiple subjects", "an image with one or no subject"),
    "outdoor": ("an outdoor scene", "an indoor scene"),
    "human_present": ("an image containing a human", "an image without a human"),
    "animal_present": ("an image containing an animal", "an image without an animal"),
    "dynamic_scene": ("a dynamic action scene", "a static scene"),
    "night": ("a nighttime scene", "a daytime or brightly lit indoor scene"),
}


class ClipLabeler:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device="cpu"):
        import open_clip
        self.torch = __import__("torch")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.model_id = f"open_clip:{model_name}:{pretrained}"

    def label(self, image) -> dict[str, dict]:
        torch = self.torch
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        output = {}
        with torch.no_grad():
            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            for task, (positive, negative) in CLIP_PROMPTS.items():
                tokens = self.tokenizer([negative, positive]).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                probability = (100 * image_features @ text_features.T).softmax(dim=-1)[0, 1].item()
                output[task] = {
                    "label": int(probability >= 0.5),
                    "confidence": max(probability, 1 - probability),
                    "positive_probability": probability,
                }
        return output
