try:
        from .iou3d_utils import boxes_iou_bev, nms_gpu, nms_normal_gpu
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    boxes_iou_bev = nms_gpu = nms_normal_gpu = None

__all__ = ['boxes_iou_bev', 'nms_gpu', 'nms_normal_gpu']
