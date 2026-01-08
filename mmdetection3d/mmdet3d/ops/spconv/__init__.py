# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
        from .conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                       SparseConvTranspose3d, SparseInverseConv2d,
                       SparseInverseConv3d, SubMConv2d, SubMConv3d)
except (ImportError, ModuleNotFoundError):
    # Skip if C++ extensions are not available (not needed for BEVFormer)
    SparseConv2d = SparseConv3d = SparseConvTranspose2d = None
    SparseConvTranspose3d = SparseInverseConv2d = SparseInverseConv3d = None
    SubMConv2d = SubMConv3d = None
try:
        from .modules import SparseModule, SparseSequential
except (ImportError, ModuleNotFoundError):
    SparseModule = SparseSequential = None
try:
        from .pool import SparseMaxPool2d, SparseMaxPool3d
except (ImportError, ModuleNotFoundError):
    SparseMaxPool2d = SparseMaxPool3d = None
try:
        from .structure import SparseConvTensor, scatter_nd
except (ImportError, ModuleNotFoundError):
    SparseConvTensor = scatter_nd = None

__all__ = [
    'SparseConv2d',
    'SparseConv3d',
    'SubMConv2d',
    'SubMConv3d',
    'SparseConvTranspose2d',
    'SparseConvTranspose3d',
    'SparseInverseConv2d',
    'SparseInverseConv3d',
    'SparseModule',
    'SparseSequential',
    'SparseMaxPool2d',
    'SparseMaxPool3d',
    'SparseConvTensor',
    'scatter_nd',
]
