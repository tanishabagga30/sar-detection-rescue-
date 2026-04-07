from typing import List, Dict, Union
import torch
from ultralytics import YOLO
from PIL import Image

class YOLOModel:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load {model_path}. Trying fallback yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        self.conf_threshold = conf_threshold

    def predict(self, image: Union[str, Image.Image]) -> List[Dict]:
        """
        Run YOLO detection on the provided image
        Returns a list of dicts: {"bbox_xyxy": [x1, y1, x2, y2], "score": conf, "label": class_name}
        """
        results = self.model(image, conf=self.conf_threshold)[0]
        
        detections = []
        target_classes = ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]

            if label not in target_classes:
                continue

            detections.append({
                "bbox_xyxy": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                "score": conf,
                "label": label
            })
        
        return detections
