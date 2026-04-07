import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix(tp, fp, fn, save_path="outputs/metrics/confusion_matrix.png"):
    """
    Simplified confusion matrix for Object Detection (ignoring True Negatives).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matrix = np.array([
        [tp, fn],
        [fp, 0] # TN is not defined in standard object detection
    ])
    
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Positive', 'Pred Negative'], 
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title('Detection Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    
def plot_precision_recall(precisions, recalls, save_path="outputs/metrics/pr_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,5))
    
    # Sort by recall for pretty plotting
    sorted_idx = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_idx]
    precisions = np.array(precisions)[sorted_idx]
    
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b', label='PR Curve')
    plt.fill_between(recalls, precisions, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.savefig(save_path)
    plt.close()
