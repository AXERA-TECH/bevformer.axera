#!/bin/bash
# BEVFormer Calibration Data Preparation Script
# Usage: prepare_calibration_data.sh [OPTIONS]
# Options:
#   --config-json PATH       Path to inference config JSON (default: PROJECT_DIR/inference_config.json)
#   --input-data-dir PATH    Input data directory (default: PROJECT_DIR/extracted_data)
#   --onnx-model PATH        ONNX model path (default: PROJECT_DIR/bevformer_tiny_fixed.onnx)
#   --num-samples N          Number of samples to select (default: 32)
#   --device DEVICE          Device for ONNX inference (default: cuda:0)
#   -h, --help               Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="$SCRIPT_DIR"
while [ ! -f "$PROJECT_DIR/tools/extract_quantization_data.py" ] && [ "$PROJECT_DIR" != "/" ]; do
    PROJECT_DIR="$(dirname "$PROJECT_DIR")"
done


if [ ! -f "$PROJECT_DIR/tools/extract_quantization_data.py" ]; then
    echo "Error: Cannot find project root directory (looking for tools/extract_quantization_data.py)"
    echo "  Searched from: $SCRIPT_DIR"
    exit 1
fi


CONFIG_JSON="$PROJECT_DIR/inference_config.json"
INPUT_DATA_DIR="$PROJECT_DIR/extracted_data"
ONNX_MODEL="$PROJECT_DIR/bevformer_tiny_fixed.onnx"
OUTPUT_DIR="$PROJECT_DIR/quantization_data"
NUM_SAMPLES=32
DEVICE="cuda:0"


while [[ $# -gt 0 ]]; do
    case $1 in
        --config-json)
            CONFIG_JSON="$2"
            shift 2
            ;;
        --input-data-dir)
            INPUT_DATA_DIR="$2"
            shift 2
            ;;
        --onnx-model)
            ONNX_MODEL="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            head -n 8 "$0" | tail -n 7
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== BEVFormer Calibration Data Preparation Script ===${NC}"
echo ""

echo -e "${YELLOW}[1/4] Extracting quantization data...${NC}"
cd "$PROJECT_DIR"
python tools/extract_quantization_data.py \
    "$CONFIG_JSON" \
    "$INPUT_DATA_DIR" \
    "$ONNX_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Data extraction failed, $OUTPUT_DIR directory does not exist"
    exit 1
fi

echo -e "${GREEN}Data extraction completed${NC}"
echo ""

echo -e "${YELLOW}[2/4] Collect and randomly select $NUM_SAMPLES samples...${NC}"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

all_files=()
scene_dirs=()

for scene_dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$scene_dir" ] && [ "$(basename "$scene_dir")" != "." ] && [ "$(basename "$scene_dir")" != ".." ]; then
        scene_name=$(basename "$scene_dir")
        scene_dirs+=("$scene_dir")

        frame_indices=($(ls "$scene_dir"/frame_*_img.npy 2>/dev/null | \
            sed 's/.*frame_\([0-9]*\)_img\.npy/\1/' | sort -n))

        for frame_idx in "${frame_indices[@]}"; do
            all_files+=("$scene_dir|$frame_idx")
        done
    fi
done

total_files=${#all_files[@]}
echo "  Total samples found: $total_files"

if [ $total_files -lt $NUM_SAMPLES ]; then
    echo "Warning: Total samples ($total_files) less than required ($NUM_SAMPLES), using all data"
    NUM_SAMPLES=$total_files
fi

if command -v shuf &> /dev/null; then
    selected_indices=($(seq 0 $((total_files-1)) | shuf -n $NUM_SAMPLES | sort -n))
else
    selected_indices=($(python3 -c "
import random
indices = list(range($total_files))
random.shuffle(indices)
selected = sorted(indices[:$NUM_SAMPLES])
print(' '.join(map(str, selected)))
"))
fi

echo "  Randomly selected $NUM_SAMPLES samples"
echo ""

echo -e "${YELLOW}[3/4] Preparing calibration data files...${NC}"

for i in "${!selected_indices[@]}"; do
    idx=${selected_indices[$i]}
    file_info=${all_files[$idx]}
    scene_dir=$(echo "$file_info" | cut -d'|' -f1)
    frame_idx=$(echo "$file_info" | cut -d'|' -f2)
    output_idx=$((i + 1))
    cp "$scene_dir/frame_${frame_idx}_img.npy" "$TEMP_DIR/img_${output_idx}.npy"
    cp "$scene_dir/frame_${frame_idx}_lidar2img.npy" "$TEMP_DIR/lidar2img_${output_idx}.npy"
    cp "$scene_dir/frame_${frame_idx}_can_bus.npy" "$TEMP_DIR/can_bus_${output_idx}.npy"
    cp "$scene_dir/frame_${frame_idx}_prev_bev.npy" "$TEMP_DIR/prev_bev_${output_idx}.npy"
done

echo -e "${GREEN}Calibration data ready${NC}"
echo ""

echo -e "${YELLOW}[4/4] Creating tar.gz packages...${NC}"
cd "$TEMP_DIR"

CALIBRATION_DIR="$PROJECT_DIR/calibration"
mkdir -p "$CALIBRATION_DIR"

if [ ! -f img_1.npy ]; then
    echo "Error: No calibration data files found in $TEMP_DIR"
    exit 1
fi

echo "  Creating calibration_img.tar.gz..."
tar -czf "$CALIBRATION_DIR/calibration_img.tar.gz" \
    --transform "s|img_\([0-9]*\)\.npy|\1.npy|" \
    img_*.npy || {
    echo "Error: Failed to create calibration_img.tar.gz"
    exit 1
}

echo "  Creating calibration_lidar2img.tar.gz..."
tar -czf "$CALIBRATION_DIR/calibration_lidar2img.tar.gz" \
    --transform "s|lidar2img_\([0-9]*\)\.npy|\1.npy|" \
    lidar2img_*.npy || {
    echo "Error: Failed to create calibration_lidar2img.tar.gz"
    exit 1
}

echo "  Creating calibration_can_bus.tar.gz..."
tar -czf "$CALIBRATION_DIR/calibration_can_bus.tar.gz" \
    --transform "s|can_bus_\([0-9]*\)\.npy|\1.npy|" \
    can_bus_*.npy || {
    echo "Error: Failed to create calibration_can_bus.tar.gz"
    exit 1
}

echo "  Creating calibration_prev_bev.tar.gz..."
tar -czf "$CALIBRATION_DIR/calibration_prev_bev.tar.gz" \
    --transform "s|prev_bev_\([0-9]*\)\.npy|\1.npy|" \
    prev_bev_*.npy || {
    echo "Error: Failed to create calibration_prev_bev.tar.gz"
    exit 1
}

echo -e "${GREEN}tar.gz files creation finished${NC}"
echo ""

echo "Tar package content:"
echo "  calibration_img.tar.gz:"
tar -tzf "$CALIBRATION_DIR/calibration_img.tar.gz" | head -3
echo "    ... (total $NUM_SAMPLES files)"
echo ""

echo -e "${YELLOW}[5/5] Cleaning up temporary data...${NC}"
rm -rf "$OUTPUT_DIR"
echo -e "${GREEN}Cleanup finished${NC}"
echo ""

echo -e "${GREEN}=== Finished ===${NC}"
echo "Calibration data files saved in: $CALIBRATION_DIR"
echo "  - calibration_img.tar.gz"
echo "  - calibration_lidar2img.tar.gz"
echo "  - calibration_can_bus.tar.gz"
echo "  - calibration_prev_bev.tar.gz"
echo ""
echo "Each tar contains $NUM_SAMPLES samples, named in order (1.npy, 2.npy, 3.npy, ...)"

