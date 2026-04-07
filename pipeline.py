import os
from PIL import Image
from typing import Dict
from models import YOLOModel, SARCLIPModel
from utils.drawing import draw_detections, add_panel

class SARPipeline:
    def __init__(self, yolo_path: str = "yolov8n.pt", clip_path: str = "openai/clip-vit-base-patch32"):
        self.yolo = YOLOModel(model_path=yolo_path)
        self.clip = SARCLIPModel(model_path=clip_path)

    def process_image(self, image_path: str) -> Dict:
        """Process a single image through the complete pipeline."""
        image = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)

        # 1. Object Detection (YOLO)
        detections = self.yolo.predict(image)
        
        # 2. Scene Understanding (CLIP)
        scene_analysis = self.clip.analyze_scene(image)
        
        # 3. Priority Scoring
        score, tier, reason = self.clip.compute_priority(len(detections), scene_analysis)

        # 4. Generate Annotated Image
        vis_img = image.copy()
        vis_img = draw_detections(vis_img, detections)
        vis_img = add_panel(
            vis_img, 
            detections=detections,
            priority_score=score,
            tier=tier,
            reason=reason,
            terrain=scene_analysis["terrain"],
            activity=scene_analysis["activity"]
        )

        return {
            "image_name": filename,
            "annotated_image_pil": vis_img,
            "num_objects": len(detections),
            "detections": detections,
            "scene": scene_analysis,
            "priority": {
                "score": score,
                "tier": tier,
                "reason": reason
            }
        }
