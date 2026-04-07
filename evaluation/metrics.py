import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Computes precision, recall, f1-score.
    predictions: dict of {image_name: [{"bbox_xyxy": [x1,y1,x2,y2], "label": str, "score": float}, ...]}
    ground_truths: dict of {image_name: [{"bbox_xyxy": [x1,y1,x2,y2], "label": str}, ...]}
    """
    tp, fp, fn = 0, 0, 0

    for img_name in ground_truths.keys():
        preds = predictions.get(img_name, [])
        gts = ground_truths[img_name]
        
        matched_gt = []

        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, g in enumerate(gts):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.append(best_gt_idx)
            else:
                fp += 1
        
        fn += len(gts) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4)
    }
