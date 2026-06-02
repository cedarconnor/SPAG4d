"""Surface-normals metrics: mean/MSE angular error (deg) and fraction below 5° / 22.5°."""

import torch


class MetricTracker:
    TRACKED = ("mean", "mse", "delta_5", "delta_22.5")

    def __init__(self, tracked_metrics):
        unknown = set(tracked_metrics) - set(self.TRACKED)
        if unknown:
            raise ValueError(f"Unsupported normals metrics: {sorted(unknown)}; "
                             f"supported = {self.TRACKED}")
        self.tracked_metrics = list(tracked_metrics)
        self.metrics_sum = {metric: 0.0 for metric in tracked_metrics}

    def update(self, pred, gt, valid_mask):
        dot_products = torch.sum(pred * gt, dim=0, keepdims=True)
        angle_radians = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        angle_degrees = torch.rad2deg(angle_radians)
        masked = angle_degrees[valid_mask]

        computed = {
            "mean":       torch.mean(masked).item(),
            "mse":        torch.mean(masked ** 2).item(),
            "delta_5":    torch.mean((masked < 5).float()).item(),
            "delta_22.5": torch.mean((masked < 22.5).float()).item(),
        }
        for k in self.tracked_metrics:
            self.metrics_sum[k] += computed[k]

    def calculate_final(self, num_samples):
        final_metrics = {}
        for metric in self.tracked_metrics:
            final_metrics[metric] = self.metrics_sum[metric] / num_samples
        return final_metrics
