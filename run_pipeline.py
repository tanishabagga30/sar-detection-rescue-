import os
import glob
import json
from pipeline import SARPipeline

TEST_DIR = "test_images"
OUT_ANN = "outputs/annotated"
OUT_METRICS = "outputs/metrics"

def main():
    os.makedirs(OUT_ANN, exist_ok=True)
    os.makedirs(OUT_METRICS, exist_ok=True)
    
    # Initialize pipeline
    print("Loading Models...")
    pipeline = SARPipeline(yolo_path="yolo_v11.pt", clip_path="openai/clip-vit-base-patch32")
    
    from xai.gradcam import YOLOGradCAM
    from xai.attention import CLIPAttention
    import cv2
    
    yolo_xai = YOLOGradCAM(pipeline.yolo)
    clip_xai = CLIPAttention(pipeline.clip)
    
    OUT_XAI_YOLO = "outputs/xai_yolo"
    OUT_XAI_CLIP = "outputs/xai_clip"
    os.makedirs(OUT_XAI_YOLO, exist_ok=True)
    os.makedirs(OUT_XAI_CLIP, exist_ok=True)
    
    images = glob.glob(os.path.join(TEST_DIR, "*.[jp][pn]*[g]")) # match jpg, png, jpeg
    
    if not images:
        print(f"No images found in {TEST_DIR}. Please add test images.")
        return

    results = {}
    
    for rank, img_path in enumerate(images):
        print(f"Processing {img_path}...")
        try:
            res = pipeline.process_image(img_path)
            
            # Save annotated image
            ann_path = os.path.join(OUT_ANN, res["image_name"])
            res["annotated_image_pil"].save(ann_path)
            
            # --- XAI GENERATION ---
            yolo_hm = yolo_xai.generate_heatmap(img_path)
            clip_hm = clip_xai.generate_attention_map(img_path)
            
            # Save XAI heatmaps
            cv2.imwrite(os.path.join(OUT_XAI_YOLO, "yolo_cam_" + res["image_name"]), yolo_hm)
            cv2.imwrite(os.path.join(OUT_XAI_CLIP, "clip_attn_" + res["image_name"]), clip_hm)
            
            # Remove PIL object from dict for json serialization
            del res["annotated_image_pil"]
            
            # Save info
            results[res["image_name"]] = res
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    # Generate JSON
    metrics_path = os.path.join(OUT_METRICS, "batch_results.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Batch inference complete. Processed {len(images)} images.")
    print(f"Check {OUT_ANN} for images and {metrics_path} for raw results.")

if __name__ == "__main__":
    main()
