try:
        from .points_in_boxes import (points_in_boxes_batch, points_in_boxes_cpu,
                                  points_in_boxes_gpu)
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    points_in_boxes_batch = points_in_boxes_cpu = points_in_boxes_gpu = None
from .roiaware_pool3d import RoIAwarePool3d

__all__ = [
    'RoIAwarePool3d', 'points_in_boxes_gpu', 'points_in_boxes_cpu',
    'points_in_boxes_batch'
]
