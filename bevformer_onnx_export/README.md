# BEVFormer ONNX Export and Inference Guide

This guide provides step-by-step instructions for exporting BEVFormer models to ONNX format, preparing calibration datasets, and running inference with ONNX Runtime or AXEngine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
  - [Export Environment](#export-environment)
  - [Inference Environment](#inference-environment)
- [Export ONNX Model](#export-onnx-model)
- [Extract Data for Quantization](#extract-data-for-quantization)
- [Prepare Calibration Dataset](#prepare-calibration-dataset)
- [Run Inference](#run-inference)
  - [ONNX Runtime Inference](#onnx-runtime-inference)
  - [AXEngine Inference](#axengine-inference)
- [Troubleshooting](#troubleshooting)

**Related Documentation:**
- [Dataset Preparation Guide](./docs/prepare_dataset.md) - Instructions for preparing NuScenes dataset for model export

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for model export and ONNX Runtime GPU inference)
- Access to mmdetection3d repository (for model export)
- Pre-trained BEVFormer checkpoint file

**Resources:**
- **Sample Data**: Download pre-extracted inference data from [HuggingFace](https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data)
- **Models & Configs**: Available at [HuggingFace](https://huggingface.co/AXERA-TECH/bevformer)

## Environment Setup

**Dependency Reference**: For complete dependency requirements and version compatibility notes, please refer to [`requirements2.txt`](./requirements2.txt).

### Export Environment

The export environment is used to convert PyTorch models to ONNX format. This requires PyTorch, mmcv, and mmdetection3d.

**For detailed dependency requirements, please refer to [`requirements2.txt`](./requirements2.txt).**

#### Quick Setup (Recommended)

For a quick setup, follow these steps. For detailed version requirements and compatibility notes, see [`requirements2.txt`](./requirements2.txt).

#### Step 1: Create Conda Environment

```bash
conda create -n bevformer_onnx_export python=3.8
conda activate bevformer_onnx_export
```

#### Step 2: Install PyTorch

**Important**: Use PyTorch >= 2.2.0. PyTorch 1.10.2 has known issues with ONNX export that cause `onnxsim` to fail. See [`requirements2.txt`](./requirements2.txt) for recommended versions.

```bash
# For CUDA 11.8
pip install torch==2.8.0+cu118 torchvision==0.23.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 12.1
pip install torch==2.8.0+cu121 torchvision==0.23.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 12.9 (recommended in requirements2.txt)
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Step 3: Install mmcv

**Important**: For models using DCN (Deformable Convolution Network) like `bevformer_small` and `bevformer_base`, you **must** install `mmcv-full` with C++ extensions. Models without DCN (like `bevformer_tiny`) can work with basic mmcv, but it's recommended to use `mmcv-full` for consistency.

```bash
# For CUDA 11.8 and PyTorch 2.2+
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html

# For CUDA 12.1 and PyTorch 2.2+
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html

# Or install from source if needed (ensures C++ extensions are built)
# git clone https://github.com/open-mmlab/mmcv.git
# cd mmcv
# MMCV_WITH_OPS=1 pip install -e .
```

**Verify mmcv C++ extensions are available:**

```bash
python -c "from mmcv.ops import ModulatedDeformConv2d; print('mmcv C++ extensions are available')"
```

If this command fails, the C++ extensions are not properly installed and models with DCN will fail to export.

#### Step 4: Install mmdetection3d

**Note**: mmdetection3d must be installed from source. See [`requirements2.txt`](./requirements2.txt) for details.

```bash
# Clone mmdetection3d repository
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d

# Install in development mode
pip install -e .
```

#### Step 5: Install ONNX Tools

See [`requirements2.txt`](./requirements2.txt) for version requirements.

```bash
pip install onnx==1.19.0
pip install onnxsim==0.4.36
pip install onnxruntime==1.23.2  # Optional, for model validation
```

#### Step 6: Install Other Dependencies

See [`requirements2.txt`](./requirements2.txt) for version requirements.

```bash
pip install numpy>=1.19.0 opencv-python>=4.5.0 tqdm>=4.60.0 pyyaml>=5.4.0
```

#### Step 7: Set PYTHONPATH

**Important**: You must set the PYTHONPATH to point to the mmdetection3d directory before running export scripts.

```bash
export PYTHONPATH=/path/to/mmdetection3d:$PYTHONPATH
```

**Why is this needed?**

- `mmdet3d` modules need to be imported from the source directory
- Custom plugin modules (`projects.mmdet3d_plugin`) need to be accessible
- This ensures consistent module imports and avoids import errors

You can add this to your `~/.bashrc` or `~/.zshrc` to make it permanent:

```bash
echo 'export PYTHONPATH=/path/to/mmdetection3d:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### Inference Environment

The inference environment is lightweight and only requires ONNX Runtime or AXEngine. It does not need PyTorch or mmdetection3d.

#### Option 1: ONNX Runtime (CPU/GPU)

```bash
conda create -n bevformer_inference python=3.8
conda activate bevformer_inference

# For CPU inference
pip install onnxruntime>=1.14.0

# For GPU inference
pip install onnxruntime-gpu>=1.14.0

# Common dependencies
pip install numpy opencv-python tqdm
```

#### Option 2: AXEngine (Hardware-specific)

```bash
conda create -n bevformer_inference python=3.8
conda activate bevformer_inference

# Install axengine (provided by hardware vendor)
# Follow vendor-specific installation instructions

# Common dependencies
pip install numpy opencv-python tqdm
```

## Export ONNX Model

**Dataset Preparation**: For exporting ONNX models, you need to prepare the training dataset (NuScenes). Please refer to the [Dataset Preparation Guide](./docs/prepare_dataset.md) for detailed instructions on downloading and preparing the NuScenes dataset.

**Model Modifications for AXERA NPU**: To adapt the BEVFormer model for AXERA NPU deployment, we have made several modifications to facilitate ONNX export and subsequent conversion to AXModel format. These modifications ensure compatibility with AXERA hardware and optimize the model structure for efficient inference. The exported ONNX model is specifically prepared for conversion using AXERA tools.

### Step 1: Prepare Checkpoint

Download or place your pre-trained BEVFormer checkpoint file in the `ckpts/` directory:

```bash
mkdir -p ckpts
# Place your checkpoint file here, e.g., bevformer_tiny_epoch_24.pth
```

**Note**: Pre-trained checkpoints and pre-converted models are available at [HuggingFace](https://huggingface.co/AXERA-TECH/bevformer). You can download checkpoint files and place them in the `ckpts/` directory.

### Step 2: Verify Model Requirements

**Models with DCN (require mmcv C++ extensions):**
- `bevformer_small` - Uses DCN in ResNet-101 backbone
- `bevformer_base` - Uses DCN in ResNet-101 backbone

**Models without DCN:**
- `bevformer_tiny` - Uses ResNet-50 without DCN

If you're exporting a model with DCN, ensure mmcv C++ extensions are properly installed (see Step 3 in Environment Setup).

### Step 3: Run Export Script

```bash
# Activate export environment
conda activate bevformer_onnx_export

# Set PYTHONPATH
export PYTHONPATH=/path/to/mmdetection3d:$PYTHONPATH

# Run export
python tools/export_onnx.py \
    --config projects/configs/bevformer/bevformer_tiny.py \
    --checkpoint ckpts/bevformer_tiny_epoch_24.pth \
    --output bevformer_tiny_onnxsim.onnx
```

**Arguments:**
- `--config`: Path to model configuration file
- `--checkpoint`: Path to PyTorch checkpoint file
- `--output`: Output ONNX model path

**Output:**
- `bevformer_tiny_onnxsim_raw.onnx`: Raw exported ONNX model (before simplification)
- `bevformer_tiny_onnxsim.onnx`: Simplified ONNX model (final output)

**What the script does:**
1. Loads the PyTorch model from checkpoint
2. Applies model modifications for AXERA NPU compatibility (see details below)
3. Exports to ONNX format with static shapes
4. Fixes GridSample operators (converts from `com.microsoft` to standard ONNX)
5. Fixes Reshape operators (converts dynamic shapes to constants)
6. Simplifies the model using `onnxsim`

**Model Modifications for AXERA NPU Compatibility:**
The export script automatically applies several modifications to ensure the exported ONNX model is compatible with AXERA NPU:
- **Static Shape Conversion**: Converts all dynamic shapes to static shapes for better hardware compatibility
- **Operator Optimization**: Optimizes operator types to match AXERA NPU supported operations
- **Tensor Layout Adjustment**: Adjusts tensor layouts and dimensions for efficient inference on AXERA hardware
- **Output Naming**: Ensures all intermediate outputs are properly named and accessible for quantization

**Note**: The exported ONNX model has been specifically prepared for AXERA NPU deployment. All shapes are static, making it suitable for quantization and conversion to AXModel format using AXERA tools (e.g., Pulsar2).

## Extract Data for Quantization

Before quantizing the model, you need to extract input data that will be used for calibration.

### Step 1: Prepare Input Data

**Download Sample Data:**

You can download pre-extracted sample data from HuggingFace:
- **Download Link**: [https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data](https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data)

Alternatively, you can prepare your own input data organized in the following structure:

```
extracted_data/
├── scene_001/
│   ├── img_000000.png
│   ├── img_000001.png
│   ├── ...
│   ├── meta_000000.json
│   ├── meta_000001.json
│   └── ...
├── scene_002/
│   └── ...
└── ...
```

Each `meta_*.json` file should contain:
```json
{
    "lidar2img": [[4x4 matrix], ...],  // List of 4x4 transformation matrices
    "can_bus": [18 float values]       // CAN bus data
}
```

### Step 2: Create Inference Config

The inference config JSON file contains model parameters needed for inference. You can generate it automatically from the PyTorch config file, or create it manually.

#### Method 1: Automatic Extraction (Recommended)

Use the `extract_config.py` script to automatically extract configuration from the PyTorch config file:

```bash
# Activate export environment
conda activate bevformer_onnx_export

# Set PYTHONPATH
export PYTHONPATH=/path/to/mmdetection3d:$PYTHONPATH

# Extract config for tiny model
python tools/extract_config.py \
    projects/configs/bevformer/bevformer_tiny.py \
    --output inference_config.json

# Or for small model
python tools/extract_config.py \
    projects/configs/bevformer/bevformer_small.py \
    --output inference_config_small.json
```

**What the script extracts:**
- `bev_h`, `bev_w`, `embed_dims` (from transformer configuration)
- `num_query`, `num_classes`
- `pc_range`, `post_center_range`
- `img_norm_cfg` (mean, std, to_rgb)
- `class_names`

**Output format:**
```json
{
  "model": {
    "bev_h": 50,
    "bev_w": 50,
    "num_query": 900,
    "num_classes": 10,
    "transformer": {
      "embed_dims": 256,
      "bev_h": 50,
      "bev_w": 50
    },
    "bbox_coder": {
      "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
      "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
      "max_num": 300,
      "num_classes": 10
    }
  },
  "img_norm": {
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
    "to_rgb": true
  },
  "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
  "dataset": {
    "class_names": [
      "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
      "motorcycle", "bicycle", "pedestrian", "traffic_cone"
    ]
  }
}
```

#### Method 2: Manual Creation

If automatic extraction is not available, you can manually create the JSON file. Here are the parameter values for different models:

| Model | bev_h | bev_w | embed_dims | img_norm (mean/std) |
|-------|-------|-------|------------|---------------------|
| **tiny** | 50 | 50 | 256 | [123.675, 116.28, 103.53] / [58.395, 57.12, 57.375] |
| **small** | 150 | 150 | 256 | [103.530, 116.280, 123.675] / [1.0, 1.0, 1.0] |
| **base** | 200 | 200 | 256 | [103.530, 116.280, 123.675] / [1.0, 1.0, 1.0] |

**Example for tiny model:**
```json
{
  "model": {
    "bev_h": 50,
    "bev_w": 50,
    "num_query": 900,
    "num_classes": 10,
    "transformer": {
      "embed_dims": 256,
      "bev_h": 50,
      "bev_w": 50
    },
    "bbox_coder": {
      "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
      "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
      "max_num": 300,
      "num_classes": 10
    }
  },
  "img_norm": {
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
    "to_rgb": true
  },
  "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
  "dataset": {
    "class_names": [
      "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
      "motorcycle", "bicycle", "pedestrian", "traffic_cone"
    ]
  }
}
```

**Verify the config file:**
```bash
# Check if the config file is valid JSON
cat inference_config.json | python -m json.tool

# Ensure it contains required fields
python -c "
import json
with open('inference_config.json') as f:
    config = json.load(f)
    assert 'model' in config, 'Missing model section'
    assert 'transformer' in config['model'], 'Missing transformer section'
    assert 'bev_h' in config['model']['transformer'], 'Missing bev_h'
    assert 'bev_w' in config['model']['transformer'], 'Missing bev_w'
    assert 'embed_dims' in config['model']['transformer'], 'Missing embed_dims'
    print('Config file is valid')
"
```

### Step 3: Run Data Extraction

```bash
# Activate export environment
conda activate bevformer_onnx_export

# Set PYTHONPATH
export PYTHONPATH=/path/to/mmdetection3d:$PYTHONPATH

# Run extraction
python tools/extract_quantization_data.py \
    inference_config.json \
    extracted_data \
    bevformer_tiny_onnxsim.onnx \
    --output-dir quantization_data \
    --device cuda:0
```

**Note**: If you don't have `extracted_data`, you can download sample data from [HuggingFace](https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data).

**Arguments:**
- `config_json`: Path to inference config JSON file
- `input_data_dir`: Directory containing extracted data (images and meta files)
- `onnx_model`: Path to ONNX model
- `--output-dir`: Output directory for quantization data (default: `./quantization_data`)
- `--device`: Device for ONNX inference (default: `cuda:0`)
- `--start-scene`: Start scene index (default: 0)
- `--end-scene`: End scene index (default: None, processes all scenes)

**Output Structure:**
```
quantization_data/
├── scene_001/
│   ├── frame_0000_img.npy
│   ├── frame_0000_lidar2img.npy
│   ├── frame_0000_can_bus.npy
│   ├── frame_0000_prev_bev.npy
│   ├── frame_0000_meta.json
│   └── ...
└── ...
```

**What the script does:**
1. Loads images and metadata from `extracted_data`
2. Preprocesses images (resize, normalize)
3. Calculates CAN bus deltas between consecutive frames
4. Runs ONNX inference to generate `bev_embed` (which becomes `prev_bev` for the next frame)
5. Saves all inputs as `.npy` files for quantization

**Data Validation**: You can use the `prepare_calibration_data.sh` script to validate your extracted data. The script will check data integrity, verify file structures, and ensure all required files are present before proceeding with quantization data preparation.

## Prepare Calibration Dataset

The calibration dataset is a small subset of quantization data used for model quantization.

**Data Validation**: The `prepare_calibration_data.sh` script also performs data validation during the preparation process. It checks data integrity, verifies file structures, and ensures all required files (images, meta files, etc.) are properly formatted before proceeding with quantization data extraction.

### Step 1: Run Calibration Data Preparation Script

```bash
# The script can be run from any directory
bash tools/prepare_calibration_data.sh \
    --config-json inference_config.json \
    --input-data-dir extracted_data \
    --onnx-model bevformer_tiny_onnxsim.onnx \
    --num-samples 32 \
    --device cuda:0
```

**Note**: If you don't have `extracted_data`, you can download sample data from [HuggingFace](https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data).

**Arguments:**
- `--config-json`: Path to inference config JSON (default: `PROJECT_DIR/inference_config.json`)
- `--input-data-dir`: Input data directory (default: `PROJECT_DIR/extracted_data`)
- `--onnx-model`: ONNX model path (default: `PROJECT_DIR/bevformer_tiny_fixed.onnx`)
- `--num-samples`: Number of samples to select (default: 32)
- `--device`: Device for ONNX inference (default: `cuda:0`)

**What the script does:**
1. Validates input data structure and file integrity
2. Runs `extract_quantization_data.py` to generate quantization data
3. Verifies extracted quantization data files
4. Randomly selects the specified number of samples
5. Organizes samples into temporary directory with consistent naming
6. Creates `.tar.gz` archives:
   - `calibration_img.tar.gz`
   - `calibration_lidar2img.tar.gz`
   - `calibration_can_bus.tar.gz`
   - `calibration_prev_bev.tar.gz`
7. Moves all archives to `calibration/` directory
8. Cleans up temporary files

**Data Validation Features:**
- Verifies that input data directory contains required scene folders
- Checks for required files (images, meta JSON files) in each scene
- Validates quantization data extraction by checking output directory structure
- Ensures all calibration data files are properly formatted before archiving

**Output:**
```
calibration/
├── calibration_img.tar.gz
├── calibration_lidar2img.tar.gz
├── calibration_can_bus.tar.gz
└── calibration_prev_bev.tar.gz
```

Each archive contains sequentially numbered files (`1.npy`, `2.npy`, ...) for easy processing by quantization tools.

## Run Inference

### ONNX Runtime Inference

#### Step 1: Prepare Data

**Download Sample Data:**

You can download pre-extracted sample data from HuggingFace:
- **Download Link**: [https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data](https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data)

Alternatively, ensure you have extracted data in the following structure:

```
inference_data/
├── scene_001/
│   ├── img_000000.png
│   ├── meta_000000.json
│   └── ...
└── ...
```

#### Step 2: Run Inference

```bash
# Activate inference environment
conda activate bevformer_inference

# Run inference
python tools/inference_onnx.py \
    bevformer_tiny_onnxsim.onnx \
    inference_config.json \
    inference_data \
    --output-dir inference_results \
    --score-thr 0.1 \
    --device cuda:0
```

**Arguments:**
- `onnx_model`: Path to ONNX model
- `config_json`: Path to inference config JSON file
- `data_dir`: Directory containing extracted data
- `--output-dir`: Output directory for results (default: `./inference_results`)
- `--score-thr`: Score threshold for detection (default: 0.1)
- `--device`: Device for inference (default: `cuda:0`)
- `--fps`: Video FPS for visualization (default: 3)
- `--start-scene`: Start scene index (default: 0)
- `--end-scene`: End scene index (default: None)

**Output:**
- Detection results in JSON format
- Visualization images/videos (if enabled)

### AXEngine Inference

#### Step 1: Quantize Model

First, you need to quantize the ONNX model using AXEngine tools. Refer to AXEngine documentation for quantization steps.

#### Step 2: Run Inference

```bash
# Activate inference environment
conda activate bevformer_inference

# Run inference
python tools/inference_axmodel.py \
    bevformer_tiny.axmodel \
    inference_config.json \
    extracted_data \
    --output-dir inference_results_axmodel \
    --score-thr 0.1
```

**Note**: If you don't have `extracted_data`, you can download sample data from [HuggingFace](https://huggingface.co/AXERA-TECH/bevformer/tree/main/inference_data).

**Arguments:**
- `model`: Path to AXModel file
- `config_json`: Path to inference config JSON file
- `data_dir`: Directory containing extracted data
- `--output-dir`: Output directory for results (default: `./inference_results_extracted`)
- `--score-thr`: Score threshold for detection (default: 0.1)
- `--fps`: Video FPS for visualization (default: 3)
- `--start-scene`: Start scene index (default: 0)
- `--end-scene`: End scene index (default: None)

**Note**: AXEngine inference does not require `--device` argument as it uses hardware-specific execution providers.

## Troubleshooting

### Issue: ModuleNotFoundError for mmdet3d

**Solution**: Set PYTHONPATH to point to mmdetection3d directory:
```bash
export PYTHONPATH=/path/to/mmdetection3d:$PYTHONPATH
```

### Issue: ONNX export fails with PyTorch 1.10.2

**Error**: `RuntimeError: eraseOutput: Assertion 'outputs_[i]->uses().empty()' failed` during `onnxsim.simplify()`

**Solution**: Upgrade to PyTorch >= 2.2.0. PyTorch 1.10.2 has known issues with ONNX export that cause incorrect model structure.

### Issue: DCN export fails - mmcv C++ extensions not available

**Error**: `AttributeError: 'NoneType' object has no attribute 'modulated_deform_conv_forward'` or `RuntimeError: Cannot export model with DCN: mmcv C++ extensions are not available`

**Solution**: 
1. Reinstall mmcv-full with C++ extensions:
   ```bash
   pip uninstall mmcv mmcv-full
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html
   ```

2. Or compile from source:
   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   MMCV_WITH_OPS=1 pip install -e .
   ```

3. Verify installation:
   ```bash
   python -c "from mmcv.ops import ModulatedDeformConv2d; print('OK')"
   ```

**Note**: This error only occurs for models using DCN (e.g., `bevformer_small`, `bevformer_base`). Models without DCN (e.g., `bevformer_tiny`) do not require mmcv C++ extensions.

### Issue: CUDA out of memory during export

**Solution**: 
- Reduce batch size in the export script
- Use a smaller model variant (e.g., `bevformer_tiny` instead of `bevformer_base`)
- Ensure no other processes are using GPU memory

### Issue: Cannot find calibration data files

**Solution**: 
- Ensure `extract_quantization_data.py` completed successfully
- Check that `quantization_data/` directory exists and contains scene subdirectories
- Verify file permissions

### Issue: AXEngine inference fails

**Error**: `RuntimeError: Failed to initialize axcl runtime`

**Solution**: 
- Ensure AXEngine drivers are properly installed
- Check that hardware device is accessible
- Verify AXModel file is compatible with your hardware version

### Issue: Shape mismatch errors

**Solution**: 
- Ensure all input data has consistent shapes
- Verify `inference_config.json` matches model configuration
- Check that image preprocessing matches training configuration
- Regenerate config file using `extract_config.py` to ensure correctness

### Issue: Config file missing required fields

**Error**: `KeyError: 'transformer'` or similar errors when loading config

**Solution**: 
1. Regenerate the config file using `extract_config.py`:
   ```bash
   python tools/extract_config.py \
       projects/configs/bevformer/bevformer_tiny.py \
       --output inference_config.json
   ```

2. Verify the config file contains all required fields:
   - `model.transformer.bev_h`
   - `model.transformer.bev_w`
   - `model.transformer.embed_dims`

3. Ensure the config file matches the model you're using (tiny/small/base)

## File Structure

```
bevformer_onnx_export/
├── tools/
│   ├── export_onnx.py              # Export PyTorch model to ONNX
│   ├── extract_quantization_data.py # Extract data for quantization
│   ├── inference_onnx.py            # ONNX Runtime inference
│   ├── inference_axmodel.py         # AXEngine inference
│   └── prepare_calibration_data.sh  # Prepare calibration dataset
├── projects/
│   ├── configs/                     # Model configurations
│   └── mmdet3d_plugin/              # Custom plugins
├── ckpts/                           # Checkpoint files
├── extracted_data/                  # Input data (images + metadata)
├── quantization_data/               # Extracted quantization data
├── calibration/                     # Calibration dataset archives
├── inference_results/                # Inference results
└── inference_config.json            # Inference configuration
```

## Additional Resources

- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/)
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [BEVFormer Paper](https://arxiv.org/abs/2203.17270)

## License

This project follows the license of the original BEVFormer repository.

## Citation

If you use this codebase, please cite the original BEVFormer paper:

```bibtex
@article{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```

