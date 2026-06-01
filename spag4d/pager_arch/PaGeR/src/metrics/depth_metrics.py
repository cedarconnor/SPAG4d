import torch
from src.utils.geometry_utils import compute_scale_and_shift, erp_to_pointcloud


def align_pred_gt(depth_pred, depth_gt, valid_mask, alignment_type):
    """LS scale (+ optional shift) alignment, or pass-through for 'metric'."""
    if alignment_type == "scale_and_shift":
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, valid_mask, fit_shift=True)
    elif alignment_type == "scale":
        scale, _ = compute_scale_and_shift(depth_pred, depth_gt, valid_mask, fit_shift=False)
        shift = 0.0
    elif alignment_type == "metric":
        scale, shift = 1.0, 0.0
    else:
        raise ValueError(f"Unknown alignment type: {alignment_type}")

    if isinstance(scale, torch.Tensor) and scale.ndim < depth_pred.ndim:
        scale = scale.view(-1, *([1] * (depth_pred.ndim - 1)))
    if isinstance(shift, torch.Tensor) and shift.ndim < depth_pred.ndim:
        shift = shift.view(-1, *([1] * (depth_pred.ndim - 1)))

    return scale * depth_pred + shift


class MetricTracker:
    def __init__(self, tracked_metrics, save_error_list=False):
        self.tracked_metrics = tracked_metrics
        self.metrics_sum = {metric: 0.0 for metric in tracked_metrics}
        self.save_error_list = save_error_list
        self.error_list = {metric: [] for metric in tracked_metrics}

    @staticmethod
    def _masked_mean(value_map, valid_mask=None):
        if valid_mask is None:
            return value_map.mean(dim=(-1, -2))
        valid_mask = valid_mask.float()
        return (value_map * valid_mask).sum(dim=(-1, -2)) / valid_mask.sum(dim=(-1, -2)).clamp_min(1e-8)

    def abs_relative_difference(self, output, target, valid_mask=None):
        diff = torch.abs(output - target) / target
        return self._masked_mean(diff, valid_mask).mean()

    def squared_relative_difference(self, output, target, valid_mask=None):
        diff = torch.pow(torch.abs(output - target), 2) / target
        return self._masked_mean(diff, valid_mask).mean()

    def rmse_linear(self, output, target, valid_mask=None):
        diff2 = torch.pow(output - target, 2)
        mse = self._masked_mean(diff2, valid_mask)
        return torch.sqrt(mse).mean()

    def threshold_percentage(self, output, target, threshold_val, valid_mask=None):
        max_d1_d2 = torch.max(output / target, target / output)
        bit_mat = (max_d1_d2 < threshold_val).float()
        return self._masked_mean(bit_mat, valid_mask).mean()

    def delta1_acc(self, pred, gt, valid_mask):
        return self.threshold_percentage(pred, gt, 1.25, valid_mask)

    def update(self, depth_pred, depth_gt, valid_mask, id):
        for metric in self.tracked_metrics:
            metric_fn = getattr(self, metric)
            value = metric_fn(depth_pred, depth_gt, valid_mask)
            self.metrics_sum[metric] += value.item()
            if self.save_error_list:
                self.error_list[metric].append((id, value.item()))

    def calculate_final(self, num_samples):
        final_metrics = {}
        for metric in self.tracked_metrics:
            final_metrics[metric] = self.metrics_sum[metric] / num_samples
        return final_metrics, self.error_list