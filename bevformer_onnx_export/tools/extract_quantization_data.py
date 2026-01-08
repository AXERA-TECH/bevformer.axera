#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Extract quantization dataset with prev_bev')
    parser.add_argument('config_json', help='JSON config file path')
    parser.add_argument('input_data_dir', help='input data directory (extracted_data)')
    parser.add_argument('onnx_model', help='ONNX model path')
    parser.add_argument('--output-dir', default='./quantization_data', help='output directory')
    parser.add_argument('--device', default='cuda:0', help='device for ONNX inference')
    parser.add_argument('--start-scene', type=int, default=0, help='start scene index')
    parser.add_argument('--end-scene', type=int, default=None, help='end scene index (None for all)')
    return parser.parse_args()


def load_onnx_model(onnx_path, device='cuda:0'):
    """Load ONNX model"""
    available_providers = ort.get_available_providers()
    
    providers = []
    if 'cuda' in device.lower() and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    return session


def load_config_from_json(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def preprocess_image(img_path, img_norm_cfg, target_size=(480, 800)):
    """Preprocess image: load, resize, normalize
    
    Args:
        img_path: path to image file
        img_norm_cfg: normalization config with 'mean', 'std', 'to_rgb'
        target_size: (H, W) target size
    
    Returns:
        img: (C, H, W) normalized numpy array, float32
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    # Convert BGR to RGB if needed
    if img_norm_cfg.get('to_rgb', True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))  # (W, H)
    
    # Convert to float and normalize
    img = img.astype(np.float32)
    mean = np.array(img_norm_cfg.get('mean', [123.675, 116.28, 103.53]), dtype=np.float32)
    std = np.array(img_norm_cfg.get('std', [58.395, 57.12, 57.375]), dtype=np.float32)
    
    img = (img - mean) / std
    
    # Convert to (C, H, W)
    img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    return img


def load_data_from_extracted(data_dir, scene_name, frame_idx):
    """Load data from extracted_data directory (PNG images + JSON meta)
    
    Args:
        data_dir: extracted_data directory path
        scene_name: scene name (scene token)
        frame_idx: frame index (sample index)
    
    Returns:
        img: (1, N, C, H, W) numpy array, preprocessed
        lidar2img: (1, N, 4, 4) numpy array
        can_bus: (1, 18) numpy array
        meta: dict with metadata
    """
    scene_dir = osp.join(data_dir, scene_name)
    
    # Load meta
    meta_path = osp.join(scene_dir, f'meta_{frame_idx:06d}.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Get normalization config
    img_norm_cfg = meta.get('img_norm_cfg', {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True
    })
    
    # Get image shape
    img_shape = meta.get('img_shape', [[480, 800, 3]] * 6)
    target_size = (img_shape[0][0], img_shape[0][1])  # (H, W)
    
    # Load images for all cameras
    num_cams = meta.get('num_cams', 6)
    imgs = []
    for cam_idx in range(num_cams):
        img_path = osp.join(scene_dir, f'cam_{cam_idx:02d}_{frame_idx:06d}.png')
        img = preprocess_image(img_path, img_norm_cfg, target_size)
        imgs.append(img)
    
    # Stack images: (N, C, H, W) -> (1, N, C, H, W)
    img = np.stack(imgs, axis=0)  # (N, C, H, W)
    img = img[np.newaxis, ...]  # (1, N, C, H, W)
    
    # Load lidar2img: (N, 4, 4) -> (1, N, 4, 4)
    lidar2img = np.array(meta['lidar2img'], dtype=np.float32)  # (N, 4, 4)
    lidar2img = lidar2img[np.newaxis, ...]  # (1, N, 4, 4)
    
    # Load can_bus: (18,) -> (1, 18)
    can_bus = np.array(meta['can_bus'], dtype=np.float32)  # (18,)
    can_bus = can_bus[np.newaxis, ...]  # (1, 18)
    
    return img, lidar2img, can_bus, meta


def main():
    args = parse_args()
    
    # Load configuration from JSON
    config = load_config_from_json(args.config_json)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ONNX model
    onnx_session = load_onnx_model(args.onnx_model, args.device)
    
    # Get model parameters from config
    transformer_cfg = config['model']['transformer']
    bev_h = transformer_cfg['bev_h']
    bev_w = transformer_cfg['bev_w']
    embed_dims = transformer_cfg['embed_dims']
    
    # Load scene index
    scene_index_path = osp.join(args.input_data_dir, 'scene_index.json')
    with open(scene_index_path, 'r') as f:
        scene_index_data = json.load(f)
    
    scenes_dict = scene_index_data['scenes']
    scene_names = list(scenes_dict.keys())
    
    end_scene = args.end_scene if args.end_scene is not None else len(scene_names)
    end_scene = min(end_scene, len(scene_names))
    
    prev_frame_info = {
        'prev_bev': None,
        'scene_token': None,
        'prev_pos': np.zeros(3, dtype=np.float32),
        'prev_angle': 0.0,
    }
    
    # Process all scenes
    for scene_idx in range(args.start_scene, end_scene):
        scene_name = scene_names[scene_idx]
        scene_info = scenes_dict[scene_name]
        sample_indices = scene_info['samples']
        num_frames = len(sample_indices)
        
        print(f"Processing scene {scene_idx+1}/{len(scene_names)}: {scene_name} ({num_frames} frames)")
        
        # Reset prev_bev for new scene
        if scene_name != prev_frame_info['scene_token']:
            prev_frame_info['prev_bev'] = None
            prev_frame_info['prev_pos'] = np.zeros(3, dtype=np.float32)
            prev_frame_info['prev_angle'] = 0.0
        
        prev_frame_info['scene_token'] = scene_name
        
        # Create output scene directory
        output_scene_dir = osp.join(args.output_dir, scene_name)
        os.makedirs(output_scene_dir, exist_ok=True)
        
        # Process all frames in this scene
        for local_idx, frame_idx in enumerate(tqdm(sample_indices, desc=f"Scene {scene_name}")):
            # Load data from extracted_data (PNG images + JSON meta)
            img, lidar2img, can_bus, meta = load_data_from_extracted(args.input_data_dir, scene_name, frame_idx)
            
            # Process can_bus (compute delta)
            if can_bus.ndim == 3:
                curr_can_bus_np = can_bus[0, 0]
            else:
                curr_can_bus_np = can_bus[0]
            
            tmp_pos = curr_can_bus_np[:3].copy()
            tmp_angle = curr_can_bus_np[-1]
            
            delta_can_bus_np = curr_can_bus_np.copy()
            
            if prev_frame_info['prev_bev'] is not None and prev_frame_info['scene_token'] == scene_name:
                delta_can_bus_np[:3] -= prev_frame_info['prev_pos']
                delta_can_bus_np[-1] -= prev_frame_info['prev_angle']
            else:
                delta_can_bus_np[:3] = 0.0
                delta_can_bus_np[-1] = 0.0
            
            prev_frame_info['prev_pos'] = tmp_pos
            prev_frame_info['prev_angle'] = tmp_angle
            
            # Prepare prev_bev
            prev_bev_input = next((inp for inp in onnx_session.get_inputs() if inp.name == 'prev_bev'), None)
            expected_shape = (bev_h * bev_w, 1, embed_dims)
            if prev_bev_input is not None:
                expected_shape = list(prev_bev_input.shape)
                for i, dim in enumerate(expected_shape):
                    if isinstance(dim, str) or dim < 0:
                        expected_shape[i] = (bev_h * bev_w, 1, embed_dims)[i] if i < 3 else 1
                expected_shape = tuple(expected_shape)
            
            if prev_frame_info['prev_bev'] is None:
                prev_bev = np.zeros(expected_shape, dtype=np.float32)
            else:
                prev_bev = prev_frame_info['prev_bev']
                if prev_bev.shape != expected_shape and len(prev_bev.shape) == 3:
                    prev_bev = prev_bev.reshape(expected_shape)
            
            # Prepare ONNX inputs
            img_np = img.astype(np.float32)
            lidar2img_np = lidar2img.astype(np.float32)
            can_bus_np = delta_can_bus_np.reshape(1, -1).astype(np.float32)
            
            input_names = [inp.name for inp in onnx_session.get_inputs()]
            onnx_inputs = {}
            for name in input_names:
                if name == 'img':
                    onnx_inputs['img'] = img_np
                elif name == 'can_bus':
                    onnx_inputs['can_bus'] = can_bus_np
                elif name == 'lidar2img':
                    onnx_inputs['lidar2img'] = lidar2img_np
                elif name == 'prev_bev':
                    onnx_inputs['prev_bev'] = prev_bev
            
            # Run inference to get bev_embed
            onnx_outputs = onnx_session.run(None, onnx_inputs)
            bev_embed, _, _ = onnx_outputs
            
            # Update prev_bev for next frame
            prev_frame_info['prev_bev'] = bev_embed
            
            # Save data for quantization
            # Note: Save the prev_bev used as input (not the output bev_embed)
            # The prev_bev for frame N is the bev_embed from frame N-1
            
            # Save img
            img_output_path = osp.join(output_scene_dir, f'frame_{local_idx:04d}_img.npy')
            np.save(img_output_path, img_np)
            
            # Save lidar2img
            lidar2img_output_path = osp.join(output_scene_dir, f'frame_{local_idx:04d}_lidar2img.npy')
            np.save(lidar2img_output_path, lidar2img_np)
            
            # Save can_bus (use delta_can_bus for quantization)
            can_bus_output_path = osp.join(output_scene_dir, f'frame_{local_idx:04d}_can_bus.npy')
            np.save(can_bus_output_path, can_bus_np)
            
            # Save prev_bev (this is the input prev_bev for this frame)
            prev_bev_output_path = osp.join(output_scene_dir, f'frame_{local_idx:04d}_prev_bev.npy')
            np.save(prev_bev_output_path, prev_bev)
            
            # Save meta (optional, for reference)
            meta_output_path = osp.join(output_scene_dir, f'frame_{local_idx:04d}_meta.json')
            with open(meta_output_path, 'w') as f:
                json.dump(meta, f, indent=2)
        
        # Save scene index
        scene_index_output = {
            'scene_name': scene_name,
            'num_frames': len(sample_indices),
            'frame_indices': list(range(len(sample_indices))),
        }
        scene_index_output_path = osp.join(output_scene_dir, 'scene_index.json')
        with open(scene_index_output_path, 'w') as f:
            json.dump(scene_index_output, f, indent=2)
        
        print(f"Scene {scene_name}: {len(sample_indices)} frames saved")
    
    # Save global index
    processed_scenes = []
    for scene_idx in range(args.start_scene, end_scene):
        scene_name = scene_names[scene_idx]
        scene_info = scenes_dict[scene_name]
        processed_scenes.append({
            'name': scene_name,
            'num_frames': len(scene_info['samples'])
        })
    
    global_index_output = {
        'num_scenes': len(processed_scenes),
        'scenes': processed_scenes,
    }
    global_index_output_path = osp.join(args.output_dir, 'global_index.json')
    with open(global_index_output_path, 'w') as f:
        json.dump(global_index_output, f, indent=2)
    
if __name__ == '__main__':
    main()

