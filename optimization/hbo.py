import numpy as np
from typing import Callable, Tuple

class HoneyBadgerOptimization:
    """
    Honey Badger Algorithm (HBA) to tune hyperparameters.
    Here we focus on 1D or 2D continuous optimization for confidence thresholds.
    """
    def __init__(self, objective_func: Callable, num_badgers: int = 20, max_iter: int = 50, bounds: Tuple[float, float] = (0.01, 0.99)):
        self.objective_func = objective_func
        self.num_badgers = num_badgers
        self.max_iter = max_iter
        self.bounds = bounds
        
        # Dimensions (e.g. 1 for just conf_threshold)
        self.dim = 1
        self.beta = 6
        self.C = 2

    def optimize(self) -> float:
        # Initialize population
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_badgers, self.dim))
        fitness = np.zeros(self.num_badgers)

        # evaluate initial population
        for i in range(self.num_badgers):
            fitness[i] = self.objective_func(positions[i, 0])
            
        best_idx = np.argmin(fitness) # assuming minimization (e.g. 1 - F1_score)
        x_prey = np.copy(positions[best_idx])
        
        for t in range(1, self.max_iter + 1):
            alpha = self.C * np.exp(-t / self.max_iter) # density factor
            
            for i in range(self.num_badgers):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                r4 = np.random.rand()
                r5 = np.random.rand()
                r6 = np.random.rand()
                r7 = np.random.rand()

                I = np.random.normal(0, 1) # intensity
                di = x_prey - positions[i]
                S = np.sign(np.random.normal(0, 1))

                # Digging phase vs Honey phase
                if r6 < 0.5:
                    # Digging phase
                    F = r4 * (2 * r5 - 1)
                    new_pos = x_prey + F * self.beta * I * x_prey + F * r3 * alpha * di * np.abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
                else:
                    # Honey phase
                    new_pos = x_prey + F * r7 * alpha * di

                # Clip to bounds
                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                
                # Evaluate
                new_fit = self.objective_func(new_pos[0])
                
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    positions[i] = new_pos
                    
                if new_fit < fitness[best_idx]:
                    best_idx = i
                    x_prey = np.copy(positions[best_idx])

        # Return best threshold found
        return float(x_prey[0])

# Helper objective function generator
def create_f1_objective(yolo_model, validation_images, ground_truths):
    """
    Creates an objective function for HBO that minimizes (1 - average_F1).
    Since HBO minimizes, we return 1.0 - F1_score.
    """
    def objective_func(thresh: float) -> float:
        yolo_model.conf_threshold = thresh
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for img_path, gt_boxes in zip(validation_images, ground_truths):
            preds = yolo_model.predict(img_path)
            
            # Simplified matching dummy
            # In a real scenario, use IoU matching
            pred_count = len(preds)
            gt_count = len(gt_boxes)
            
            tp = min(pred_count, gt_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return 1.0 - f1

    return objective_func
