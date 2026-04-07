import cv2
import numpy as np
import torch
from PIL import Image

class CLIPAttention:
    """
    Extracts self-attention from CLIP's Vision Transformer to see which areas 
    contribute most to the embedding.
    """
    def __init__(self, clip_model):
        self.model = clip_model.model
        self.processor = clip_model.processor
        self.device = next(self.model.parameters()).device

    def generate_attention_map(self, image_path: str) -> np.ndarray:
        img_pil = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # We need to access the attention weights
        # We hook into the last layer of the vision transformer
        attention_weights = []
        def hook(module, input, output):
            # output is a tuple for MultiheadAttention
            attention_weights.append(output[1])

        # Hook the last attention layer
        handle = None
        try:
            target_layer = self.model.vision_model.encoder.layers[-1].self_attn
            handle = target_layer.register_forward_hook(hook)
        except AttributeError:
            pass

        with torch.no_grad():
            _ = self.model.get_image_features(**inputs)

        if handle:
            handle.remove()

        img_np = np.array(img_pil)

        if not attention_weights or attention_weights[0] is None:
            # Fallback
            return img_np

        # attention_weights[0] shape might vary, typically [batch_size, num_heads, seq_len, seq_len]
        # We want the attention from the CLS token to all patch tokens
        # For CLIP ViT, seq_len = 1 + num_patches
        attn = attention_weights[0][0].mean(dim=0) # average across heads
        cls_attn = attn[0, 1:] # attention from CLS to patches

        # Reshape to grid
        num_patches = cls_attn.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            return img_np
            
        cls_attn = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize
        cls_attn = cls_attn - np.min(cls_attn)
        cls_attn = cls_attn / np.max(cls_attn)
        
        # Resize to original image size
        heatmap = cv2.resize(cls_attn, (img_np.shape[1], img_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_np, 0.5, color_map, 0.5, 0)
        return overlay
