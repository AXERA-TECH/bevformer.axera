[English](./README_EN.md) | [简体中文](./README.md)

# BEVFormer Inference

BEVFormer DEMO on Axera

## Support Platform

- [x] AX650
- [ ] AX637

## Project Structure

```
bevformer/
├── CMakeLists.txt          # Build configuration
├── build650.sh             # Build script for AX650
├── build637.sh             # Build script for AX637
├── README.md               # This file
├── toolchains/
│   └── aarch64-none-linux-gnu.toolchain.cmake  # Cross-compilation toolchain
├── include/                # Header files
│   ├── bevformer_common.hpp
│   ├── data_loader.hpp
│   ├── preprocess.hpp
│   ├── postprocess.hpp
│   ├── visualization.hpp
│   ├── utils.hpp
│   └── timer.hpp
└── src/                    # Source files
│   ├── main.cpp
│   ├── data_loader.cpp
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   ├── visualization.cpp
│   └── utils.cpp
├── script/                 # Python reference implementation
│   └── inference_axmodel.py  # Python inference script using axengine
└── bevformer_onnx_export/ 
```

## Dependencies

- OpenCV (>= 3.0)
- AXERA BSP (msp/out directory) - chip-specific (AX650 or AX637)
- CMake (>= 3.13)
- C++14 compiler
- jsoncpp (optional, for JSON parsing - can use simplified parser)
- Cross-compilation toolchain (for x86_64 hosts targeting aarch64)

## Building

### Automated Build (Recommended)

The project provides separate build scripts for different chip types:

#### For AX650:
```bash
./build650.sh
```

#### For AX637:
```bash
./build637.sh
```

The build scripts will automatically:
1. Check and verify system dependencies (cmake, wget, unzip, tar, git, make)
2. Download and setup OpenCV library for aarch64
3. Clone and setup BSP SDK for the target chip
4. Download and setup cross-compilation toolchain (for x86_64 hosts)
5. Configure CMake with the appropriate chip type and build the project

**Note**: 
- On first run, the script will download ~500MB of dependencies. Subsequent runs will reuse cached files.
- Build outputs are stored in separate directories: `build_ax650/` and `build_ax637/`
- Each build script is dedicated to its specific chip type (build650.sh for AX650, build637.sh for AX637)

### Manual Build

If you prefer to build manually:

```bash
# For AX650
mkdir build_ax650 && cd build_ax650
cmake -DBSP_MSP_DIR=/path/to/ax650/msp/out -DAXERA_TARGET_CHIP=ax650 ..
make -j$(nproc)

# For AX637
mkdir build_ax637 && cd build_ax637
cmake -DBSP_MSP_DIR=/path/to/ax637/msp/out -DAXERA_TARGET_CHIP=ax637 ..
make -j$(nproc)
```

#### Manual Dependency Setup

1. **OpenCV**: Download from [here](https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-aarch64-linux-gnu-gcc-7.5.0.zip) and extract to `3rdparty/`
2. **BSP SDK**: 
   - For AX650: Clone from `https://github.com/AXERA-TECH/ax650n_bsp_sdk.git`
   - For AX637: Clone from `https://github.com/AXERA-TECH/ax637_bsp_sdk.git` (comming in the future...)
3. **Toolchain**: Download from [ARM](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz) and extract

#### CMake Configuration Options

- `AXERA_TARGET_CHIP`: Target chip type (ax650 or ax637, default: ax650)
- `BSP_MSP_DIR`: Path to BSP msp/out directory
- `CMAKE_TOOLCHAIN_FILE`: Path to cross-compilation toolchain file (for cross-compilation)


## Usage

```bash
./bevformer_inference <model> <config_json> <data_dir> [options]
```

### Arguments

- `model`: Path to BEVFormer AXModel file (.axmodel)
- `config_json`: Path to configuration JSON file
- `data_dir`: Path to extracted data directory (should contain scene_index.json and scene folders)

### Options

- `--output-dir <dir>`: Output directory (default: ./inference_results_extracted)
- `--score-thr <float>`: Score threshold (default: 0.1)
- `--fps <int>`: Video FPS (default: 3)
- `--start-scene <int>`: Start scene index (default: 0)
- `--end-scene <int>`: End scene index (default: all)

### Example

```bash
./bevformer_inference \
    models/bevformer.axmodel \
    config/bevformer_config.json \
    data/extracted_data \
    --output-dir ./results \
    --score-thr 0.3 \
    --fps 3
```

## Python Reference Implementation

The project also includes a Python reference implementation in `script/inference_axmodel.py` that uses the `axengine` Python library for inference. This script serves as:

- **Reference Implementation**: A working Python version that demonstrates the complete inference pipeline
- **Validation Tool**: Can be used to verify C++ implementation correctness by comparing outputs
- **Development Aid**: Easier to modify and debug during development

### Python Script Usage

```bash
python3 script/inference_axmodel.py <model> <config_json> <data_dir> [options]
```

**Arguments and Options** (same as C++ version):
- `model`: Path to BEVFormer AXModel file (.axmodel)
- `config_json`: Path to configuration JSON file
- `data_dir`: Path to extracted data directory
- `--output-dir <dir>`: Output directory (default: ./inference_results_extracted)
- `--score-thr <float>`: Score threshold (default: 0.1)
- `--fps <int>`: Video FPS (default: 3)
- `--start-scene <int>`: Start scene index (default: 0)
- `--end-scene <int>`: End scene index (default: all)

### Python Dependencies

```bash
opencv-python numpy axengine tqdm
```

### Example

```bash
python3 script/inference_axmodel.py \
    models/bevformer.axmodel \
    config/bevformer_config.json \
    data/extracted_data \
    --output-dir ./results \
    --score-thr 0.3 \
    --fps 3
```

**Note**: The Python script produces the same output format as the C++ version, making it easy to compare results and validate the C++ implementation.

## Data Format

The data directory should have the following structure:

```
data_dir/
├── scene_index.json
└── scene_xxx/
    ├── meta_000000.json
    ├── cam_00_000000.png
    ├── cam_01_000000.png
    ├── cam_02_000000.png
    ├── cam_03_000000.png
    ├── cam_04_000000.png
    └── cam_05_000000.png
    ...
```

## Model and Dataset

### Pre-converted Models

Pre-converted AXModel files are available for download at:
- **Models**: [https://huggingface.co/AXERA-TECH/bevformer](https://huggingface.co/AXERA-TECH/bevformer)
- **Sample Dataset**: [https://huggingface.co/AXERA-TECH/bevformer](https://huggingface.co/AXERA-TECH/bevformer) 
- **Inference Json File**: [https://huggingface.co/AXERA-TECH/bevformer](https://huggingface.co/AXERA-TECH/bevformer)

### Model Conversion

If you want to convert the model yourself, please refer to the `bevformer_onnx_export` folder, which contains:

- ONNX export scripts
- Quantization dataset preparation scripts
- Model modifications for AXERA chip compatibility
- ONNX model inference with onnxruntime tool(inference_onnx.py) 

**Requirements for model conversion:**
1. Follow the instructions in the `bevformer_onnx_export` folder
2. Configure the environment according to the requirements
3. Prepare the corresponding dataset

**ONNX to AXModel conversion:**
For converting ONNX models to AXModel format, please refer to the [Pulsar2 documentation](https://pulsar2-docs.readthedocs.io/en/latest/index.html).

## Output

### Output Structure

After running inference, the program generates the following output structure:

```
<output_dir>/
├── <scene_id>/
│   ├── images/
│   │   ├── frame_000000.png    # Frame 0 visualization
│   │   ├── frame_000001.png    # Frame 1 visualization
│   │   └── ...
│   └── <scene_id>_result.avi   # Combined video file
└── ...
```

**Output Files:**
- **`images/frame_XXXXXX.png`**: Individual frame visualization images showing:
  - 6 camera views with 3D bounding boxes projected onto each image
  - BEV (Bird's Eye View) map with 3D detections visualized from top-down perspective
- **`<scene_id>_result.avi`**: Video file containing all frames concatenated together (MJPG codec)

### Example Output

Running the inference command:

```bash
./bevformer_inference ax650/compiled.axmodel config.json inference_data/ --output-dir results
```

**Console Output:**

```
Loaded configuration:
  bev_h=200, bev_w=200, embed_dims=256
  max_num=100, score_threshold=0.1
  pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3]
  post_center_range=[-61.2, -61.2, -10, 61.2, 61.2, 10]

Found 2 scenes by directory scan
Total scenes to process: 2
  - 325cef682f064c55a255f2625c533b75: 41 frames
  - fcbccedd61424f1b85dcbf8f897f9754: 40 frames

Processing scene 1/2: 325cef682f064c55a255f2625c533b75 (41 frames)
Processing: [========================================] 100% [41/41] 55.1fps, ETA: 00:00

Performance Statistics (Scene: 325cef682f064c55a255f2625c533b75, 41 frames):
  Load:       avg=107.22 ms
  Preprocess: avg=29.59 ms
  Inference:  avg=89.83 ms (min=89.60, max=90.05)
  Postprocess: avg=1.86 ms
  Visualize:  avg=153.40 ms
  Total:      avg=744.28 ms (min=701.33, max=795.70), FPS=1.34

Created video with 41 frames using codec MJPG (AVI): results/325cef682f064c55a255f2625c533b75/325cef682f064c55a255f2625c533b75_result.avi
Scene 325cef682f064c55a255f2625c533b75 completed
```

**Performance Statistics Explanation:**
- **Load**: Time to load camera images from disk (optimized with parallel loading using OpenMP)
- **Preprocess**: Image preprocessing time (normalization, resizing)
- **Inference**: NPU model inference time (typically ~90ms per frame on AX650)
- **Postprocess**: Detection decoding and filtering time
- **Visualize**: Time to render 3D boxes and create visualization images (optimized with parallel processing)
- **Total**: End-to-end processing time per frame
- **FPS**: Effective frames per second (inverse of average total time)

### Visualization Results

Each frame visualization includes:
- **6 Camera Views** (top row): Front-right, front, front-lift, back-right, back, back-left cameras with projected 3D bounding boxes
- **BEV Map** (bottom): Top-down view showing detected objects with bounding boxes, labels, and direction arrows

![Visualization Example](./asset/output.gif)


## Notes

- The current implementation uses a simplified JSON parser. For production use, consider integrating jsoncpp library.
- Make sure the BSP directory contains the required libraries (ax_engine, ax_interpreter, ax_sys, ax_ivps).
- The model inputs should match: img, can_bus, lidar2img, prev_bev
- The model outputs should include: cls_scores, bbox_preds, bev_embed

## Technical discussion

- Github issues
- QQ Group: 139953715
