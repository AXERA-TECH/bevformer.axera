# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
try:
    from .pointnet2_sa_msg import PointNet2SAMSG
    from .pointnet2_sa_ssg import PointNet2SASSG
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    PointNet2SAMSG = None
    PointNet2SASSG = None
try:
    from .second import SECOND
except (ImportError, ModuleNotFoundError):
    SECOND = None

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'PointNet2SASSG', 'PointNet2SAMSG', 'MultiBackbone'
]
