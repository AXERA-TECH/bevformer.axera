# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_scatter import PointPillarsScatter
try:
    from .sparse_encoder import SparseEncoder
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    SparseEncoder = None
try:
    from .sparse_unet import SparseUNet
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    SparseUNet = None

__all__ = ['PointPillarsScatter', 'SparseEncoder', 'SparseUNet']
