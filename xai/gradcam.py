import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class YOLOGradCAM:
    """
    Simplified pseudo-GradCAM for YOLO.
    Extracts feature maps from a deep layer to visualize spatial activations.
    For true GradCAM, gradients w.r.t specific box outputs would be needed. 
    Here we visualize the aggregated activation magnitude across channels for a pseudo-heatmap.
    """
    def __init__(self, model):
        self.model = model.model  # ultralytics model

    def generate_heatmap(self, image_path: str) -> np.ndarray:
        # Since standard ultralytics YOLO gradients are hard to extract directly without breaking the graph,
        # we visualize the raw feature activations of the last bottleneck as a proxy for attention.
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # We need a standardized size (e.g. 640x640)
        input_tensor = torch.from_numpy(cv2.resize(img_rgb, (640, 640))).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        features = []
        def hook(module, input, output):
            features.append(output)
            
        # Hook into a deep layer (e.g., specific Conv)
        # YOLOv8 architectures varied, we try to hook the 15th layer as a proxy
        layer_to_hook = None
        for i, (name, module) in enumerate(self.model.model.named_modules()):
            if i == 15 and hasattr(module, 'forward'): # somewhat arbitrary deep conv layer
                layer_to_hook = module
                break
                
        handle = None
        if layer_to_hook:
            handle = layer_to_hook.register_forward_hook(hook)

        # Forward pass
        with torch.no_grad():
            try:
                _ = self.model(input_tensor)
            except Exception:
                pass

        if handle:
            handle.remove()

        if len(features) == 0:
            # Fallback if hook failed: return empty heatmap
            return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        feature_map = features[0].squeeze(0).cpu() # [C, H, W]
        # Aggregate across channels
        heatmap = torch.mean(feature_map, dim=0).numpy()
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # overlay
        overlay = cv2.addWeighted(img_rgb, 0.5, color_map, 0.5, 0)
        return overlay
