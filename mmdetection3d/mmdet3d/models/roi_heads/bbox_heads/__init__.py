# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads.bbox_heads import (BBoxHead, ConvFCBBoxHead,
                                               DoubleConvFCBBoxHead,
                                               Shared2FCBBoxHead,
                                               Shared4Conv1FCBBoxHead)
from .h3d_bbox_head import H3DBboxHead
try:
        from .parta2_bbox_head import PartA2BboxHead
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    PartA2BboxHead = None

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'PartA2BboxHead',
    'H3DBboxHead'
]
