import albumentations as A
import cv2
import os
from typing import List

class AugmentationPipeline:
    def __init__(self):
        # SAR Specific Augmentations
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.7
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.augmentations.geometric.rotate.Rotate(limit=45, p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def augment_dataset(self, image_dir: str, labels_dict: dict, output_dir: str, num_variations: int = 2):
        """
        labels_dict: { "image_name.jpg": [ [x1, y1, x2, y2, "person"], ... ] }
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        for img_name, boxes in labels_dict.items():
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = []
            class_labels = []
            for box in boxes:
                bboxes.append(box[:4])
                class_labels.append(box[4])
            
            for i in range(num_variations):
                try:
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    new_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    
                    new_name = f"aug_{i}_{img_name}"
                    new_path = os.path.join(output_dir, new_name)
                    cv2.imwrite(new_path, new_img)
                    
                    # Store new bounding boxes
                    new_boxes = []
                    for t_box, t_cls in zip(transformed['bboxes'], transformed['class_labels']):
                        new_boxes.append([t_box[0], t_box[1], t_box[2], t_box[3], t_cls])
                        
                    results[new_name] = new_boxes
                except Exception as e:
                    print(f"Failed augmentation on {img_name}: {e}")

        return results
