import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Tuple
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class SARCLIPModel:
    def __init__(self, model_path: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_path).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

        self.terrain_prompts = [
            "drone search and rescue scene with open accessible terrain",
            "drone search and rescue scene with dense vegetation",
            "drone search and rescue scene with rocky or rubble terrain",
            "drone search and rescue scene with visible road or trail access",
            "drone search and rescue scene with water or wet terrain",
            "drone search and rescue scene with steep slope or uneven ground",
            "drone search and rescue scene with obstructed access for responders",
        ]
        
        self.terrain_short = {
            "drone search and rescue scene with open accessible terrain": "open_accessible",
            "drone search and rescue scene with dense vegetation": "dense_vegetation",
            "drone search and rescue scene with rocky or rubble terrain": "rocky_rubble",
            "drone search and rescue scene with visible road or trail access": "road_or_trail",
            "drone search and rescue scene with water or wet terrain": "water_or_wet",
            "drone search and rescue scene with steep slope or uneven ground": "steep_or_uneven",
            "drone search and rescue scene with obstructed access for responders": "obstructed_access",
        }

        self.activity_prompts = [
            "drone search and rescue image showing a running person or animal",
            "drone search and rescue image showing a walking person or animal",
            "drone search and rescue image showing a laying down person or animal",
            "drone search and rescue image showing a seated person or animal",
            "drone search and rescue image showing active movement",
            "drone search and rescue image showing possible distressed or injured posture",
        ]

    def _run_clip(self, image: Image.Image, prompts: List[str]) -> Dict[str, float]:
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

        return {prompt: float(score) for prompt, score in zip(prompts, probs)}

    def analyze_scene(self, image: Image.Image) -> Dict:
        """Analyze terrain and overall activity in the image."""
        terrain_scores = self._run_clip(image, self.terrain_prompts)
        top_terrain = sorted(terrain_scores.items(), key=lambda x: x[1], reverse=True)[0]

        activity_scores = self._run_clip(image, self.activity_prompts)
        top_activity = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)[0]

        return {
            "terrain": self.terrain_short[top_terrain[0]],
            "terrain_confidence": round(float(top_terrain[1]), 4),
            "activity": top_activity[0].replace("drone search and rescue image showing ", ""),
            "activity_confidence": round(float(top_activity[1]), 4),
        }

    def compute_priority(self, num_objects: int, scene_analysis: Dict) -> Tuple[int, str, str]:
        """Heuristic based priority score."""
        score = 0
        reasons = []

        if num_objects == 0:
            return 0, "P4", "No objects detected."

        if "distressed or injured" in scene_analysis["activity"]:
            score += 40
            reasons.append("distress signal detected")
        elif "laying down" in scene_analysis["activity"]:
            score += 30
            reasons.append("lying down posture detected")
        
        # Terrain hazard additions
        hazard_terrains = ["rocky_rubble", "steep_or_uneven", "water_or_wet"]
        if scene_analysis["terrain"] in hazard_terrains:
            score += 20
            reasons.append(f"hazardous terrain ({scene_analysis['terrain']})")

        score += min(30, num_objects * 10)
        reasons.append(f"multiple subjects ({num_objects})")
        
        score = max(0, min(score, 100))

        if score >= 70:
            tier = "P1"
        elif score >= 40:
            tier = "P2"
        elif score >= 20:
            tier = "P3"
        else:
            tier = "P4"

        return score, tier, ", ".join(reasons).capitalize() + "."
