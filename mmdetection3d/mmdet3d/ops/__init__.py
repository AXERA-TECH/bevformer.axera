# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import (RoIAlign, SigmoidFocalLoss, get_compiler_version,
                      get_compiling_cuda_version, nms, roi_align,
                      sigmoid_focal_loss)

# Try to import ops that require C++ extensions, skip if not available (not needed for BEVFormer)
try:
        from .ball_query import ball_query
except (ImportError, ModuleNotFoundError):
    ball_query = None
try:
        from .furthest_point_sample import (Points_Sampler, furthest_point_sample,
                                        furthest_point_sample_with_dist)
except (ImportError, ModuleNotFoundError):
    Points_Sampler = furthest_point_sample = furthest_point_sample_with_dist = None
try:
        from .gather_points import gather_points
except (ImportError, ModuleNotFoundError):
    gather_points = None
try:
        from .group_points import GroupAll, QueryAndGroup, grouping_operation
except (ImportError, ModuleNotFoundError):
    GroupAll = QueryAndGroup = grouping_operation = None
try:
        from .knn import knn
except (ImportError, ModuleNotFoundError):
    knn = None
try:
        from .paconv import PAConv, PAConvCUDA
except (ImportError, ModuleNotFoundError):
    PAConv = PAConvCUDA = None
try:
        from .pointnet_modules import (PointFPModule, PointSAModule,
                                   PointSAModuleMSG, build_sa_module)
except (ImportError, ModuleNotFoundError):
    PointFPModule = PointSAModule = PointSAModuleMSG = build_sa_module = None
try:
        from .roiaware_pool3d import (RoIAwarePool3d, points_in_boxes_batch,
                                   points_in_boxes_cpu, points_in_boxes_gpu)
except (ImportError, ModuleNotFoundError):
    RoIAwarePool3d = points_in_boxes_batch = points_in_boxes_cpu = points_in_boxes_gpu = None
try:
        from .sparse_block import (SparseBasicBlock, SparseBottleneck,
                                make_sparse_convmodule)
except (ImportError, ModuleNotFoundError):
    SparseBasicBlock = SparseBottleneck = make_sparse_convmodule = None
try:
        from .voxel import (DynamicScatter, dynamic_scatter, DynamicVoxelization,
                        HardSimpleVoxelization, Voxelization, dynamic_voxelize,
                        hard_simple_voxelize, voxelization)
except (ImportError, ModuleNotFoundError):
    DynamicScatter = dynamic_scatter = DynamicVoxelization = None
    HardSimpleVoxelization = Voxelization = dynamic_voxelize = None
    hard_simple_voxelize = voxelization = None

__all__ = [
    'ball_query', 'furthest_point_sample', 'furthest_point_sample_with_dist',
    'gather_points', 'GroupAll', 'grouping_operation', 'QueryAndGroup', 'knn',
    'PAConv', 'PAConvCUDA', 'PointFPModule', 'PointSAModule',
    'PointSAModuleMSG', 'build_sa_module', 'RoIAwarePool3d',
    'points_in_boxes_batch', 'points_in_boxes_cpu', 'points_in_boxes_gpu',
    'SparseBasicBlock', 'SparseBottleneck', 'make_sparse_convmodule',
    'DynamicScatter', 'dynamic_scatter', 'DynamicVoxelization',
    'HardSimpleVoxelization', 'Voxelization', 'dynamic_voxelize',
    'hard_simple_voxelize', 'voxelization', 'Points_Sampler'
]
