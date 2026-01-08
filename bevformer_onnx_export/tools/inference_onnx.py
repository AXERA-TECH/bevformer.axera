#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='BEVFormer ONNX Inference from Extracted Data')
    parser.add_argument('onnx_model', help='ONNX model path')
    parser.add_argument('config_json', help='JSON config file path')
    parser.add_argument('data_dir', help='extracted data directory (inference_data)')
    parser.add_argument('--output-dir', default='./inference_results', help='output directory')
    parser.add_argument('--score-thr', type=float, default=0.1, help='score threshold')
    parser.add_argument('--device', default='cuda:0', help='device for ONNX inference')
    parser.add_argument('--fps', type=int, default=3, help='video fps')
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
    img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    return img


def load_data(data_dir, scene_name, frame_idx):
    """Load data
    
    Args:
        data_dir: data directory path
        scene_name: scene name (scene token)
        frame_idx: frame index (sample index)
    
    Returns:
        img: (1, N, C, H, W) numpy array
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

CLASS_COLORS = {
    0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 0, 255), 3: (0, 165, 255),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 128), 7: (255, 165, 0),
    8: (0, 0, 255), 9: (128, 128, 128),
}


def denormalize_bbox_np(normalized_bboxes, pc_range):
    """Denormalize bbox using numpy only"""
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    
    rot = np.arctan2(rot_sine, rot_cosine)
    
    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]
    
    w = np.exp(w)
    l = np.exp(l)
    h = np.exp(h)
    
    if normalized_bboxes.shape[-1] > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)
    else:
        denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot], axis=-1)
    return denormalized_bboxes

def decode_bboxes_custom_np(all_cls_scores, all_bbox_preds, pc_range, post_center_range, max_num=100, score_threshold=None, num_classes=10):
    """Custom bbox decode function"""
    # Use output from the last decoder layer
    all_cls_scores = all_cls_scores[-1]  # (bs, num_query, num_classes)
    all_bbox_preds = all_bbox_preds[-1]  # (bs, num_query, 10)
    
    batch_size = all_cls_scores.shape[0]
    predictions_list = []
    
    for i in range(batch_size):
        cls_scores = all_cls_scores[i]  # (num_query, num_classes)
        bbox_preds = all_bbox_preds[i]  # (num_query, 10)
        
        # Apply sigmoid
        cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        
        # TopK selection
        cls_scores_flat = cls_scores.reshape(-1)
        topk_indices = np.argsort(cls_scores_flat)[::-1][:max_num]
        scores = cls_scores_flat[topk_indices]
        labels = topk_indices % num_classes
        bbox_index = topk_indices // num_classes
        bbox_preds = bbox_preds[bbox_index]
        
        # Denormalize bbox
        final_box_preds = denormalize_bbox_np(bbox_preds, pc_range)  # (max_num, 9)
        final_scores = scores
        final_preds = labels
        
        # Apply score threshold
        if score_threshold is not None:
            thresh_mask = final_scores > score_threshold
            tmp_score = score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = np.ones(len(final_scores), dtype=bool)
                    break
                thresh_mask = final_scores >= tmp_score
        else:
            thresh_mask = np.ones(len(final_scores), dtype=bool)
        
        # Apply post processing range filtering
        if post_center_range is not None:
            post_center_range_arr = np.array(post_center_range)
            mask = (final_box_preds[..., :3] >= post_center_range_arr[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= post_center_range_arr[3:]).all(1)
            mask &= thresh_mask
            
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
        else:
            boxes3d = final_box_preds[thresh_mask]
            scores = final_scores[thresh_mask]
            labels = final_preds[thresh_mask]
        
        predictions_list.append({
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        })
    
    return predictions_list


def get_bboxes_custom_np(preds_dicts, pc_range, post_center_range, max_num=100, score_threshold=None, num_classes=10):
    """Custom get_bboxes function"""
    # Decode bounding boxes
    preds_list = decode_bboxes_custom_np(
        preds_dicts['all_cls_scores'],
        preds_dicts['all_bbox_preds'],
        pc_range,
        post_center_range,
        max_num,
        score_threshold,
        num_classes
    )
    
    num_samples = len(preds_list)
    ret_list = []
    
    for i in range(num_samples):
        preds = preds_list[i]
        bboxes = preds['bboxes']
        
        if len(bboxes) == 0:
            ret_list.append((
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64)
            ))
            continue
        
        # Adjust z coordinate: convert center z to bottom center z
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        
        # Shrink box dimensions: multiply w, l, h by 0.9 to fix oversized boxes
        bboxes[:, 3:6] = bboxes[:, 3:6] * 0.9
        
        scores = preds['scores']
        labels = preds['labels']
        
        ret_list.append((bboxes, scores, labels))
    
    return ret_list


def format_bbox_result_np(bboxes, scores, labels):
    return {
        'boxes_3d': bboxes,
        'scores_3d': scores,
        'labels_3d': labels
    }


def rotation_3d_in_axis_np(points, angles, axis=2):
    """Rotate points by angles according to axis"""
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    
    if axis == 2 or axis == -1:
        # Rotate around z-axis
        # Build rotation matrix: (N, 3, 3)
        N = len(angles)
        rot_mat = np.zeros((N, 3, 3), dtype=points.dtype)
        rot_mat[:, 0, 0] = rot_cos
        rot_mat[:, 0, 1] = -rot_sin
        rot_mat[:, 0, 2] = zeros
        rot_mat[:, 1, 0] = rot_sin
        rot_mat[:, 1, 1] = rot_cos
        rot_mat[:, 1, 2] = zeros
        rot_mat[:, 2, 0] = zeros
        rot_mat[:, 2, 1] = zeros
        rot_mat[:, 2, 2] = ones
        
        # Apply rotation: (N, M, 3) @ (N, 3, 3) -> (N, M, 3)
        return np.einsum('aij,ajk->aik', points, rot_mat)
    else:
        raise ValueError(f'Only axis=2 (z-axis) is supported for LiDAR boxes')


def compute_bbox_corners_np(bboxes):
    """Compute 8 corners of 3D bbox"""
    if len(bboxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    
    dtype = bboxes.dtype
    
    # Extract bbox parameters
    centers = bboxes[:, :3]  # (N, 3) [x, y, z] - the bottom center
    w = bboxes[:, 3:4]  # width (y direction)
    l = bboxes[:, 4:5]  # length (x direction)
    h = bboxes[:, 5:6]  # height (z direction)
    dims = np.concatenate([l, w, h], axis=1)  # (N, 3) [x_size, y_size, z_size] = [l, w, h]
    yaws = bboxes[:, 6]  # (N,) yaw angle
    
    # Key fix: offset yaw by -80 degrees
    yaws = yaws - (np.pi / 2.0 - np.pi / 18.0)
    
    # Generate corners
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1).astype(dtype)
    
    # Rearrange to [0, 1, 3, 2, 4, 5, 7, 6]
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    
    # Use relative origin [0.5, 0.5, 0] (bottom center)
    corners_norm = corners_norm - np.array([0.5, 0.5, 0], dtype=dtype)
    
    # Scale corners: dims is [x_size, y_size, z_size]
    corners = dims[:, np.newaxis, :] * corners_norm[np.newaxis, :, :]  # (N, 8, 3)
    
    # Rotate around z-axis
    corners = rotation_3d_in_axis_np(corners, yaws, axis=2)
    
    # Translate to center point
    corners += centers[:, np.newaxis, :]
    
    return corners


def draw_bbox3d_on_img_custom_np(bboxes, raw_img, lidar2img_rt, color=(0, 255, 0), thickness=2):
    """Custom 3D bbox drawing"""
    img = raw_img.copy()
    
    if len(bboxes) == 0:
        return img
    
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    if not isinstance(lidar2img_rt, np.ndarray):
        lidar2img_rt = np.array(lidar2img_rt)
    
    lidar2img_rt = lidar2img_rt.reshape(4, 4)
    
    # Compute corners
    corners_3d = compute_bbox_corners_np(bboxes)  # (N, 8, 3)
    
    num_bbox = corners_3d.shape[0]
    
    # Project to 2D
    corners_3d_flat = corners_3d.reshape(-1, 3)  # (N*8, 3)
    ones = np.ones((corners_3d_flat.shape[0], 1), dtype=np.float32)
    pts_4d = np.concatenate([corners_3d_flat, ones], axis=-1)  # (N*8, 4)
    
    # Project
    pts_2d = pts_4d @ lidar2img_rt.T  # (N*8, 4)
    
    # Perspective division
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    # Reshape to (N, 8, 2)
    imgfov_pts_2d = pts_2d[:, :2].reshape(num_bbox, 8, 2)
    
    # Draw 12 lines
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(np.int32)
        for start, end in line_indices:
            pt1 = (int(corners[start, 0]), int(corners[start, 1]))
            pt2 = (int(corners[end, 0]), int(corners[end, 1]))
            # Check if points are within image range
            h, w = img.shape[:2]
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    return img.astype(np.uint8)


def post_process_outputs_np(all_cls_scores, all_bbox_preds, config, score_thr=0.1):
    bbox_coder = config['model']['bbox_coder']
    pc_range = bbox_coder['pc_range']
    post_center_range = bbox_coder['post_center_range']
    max_num = bbox_coder['max_num']
    score_threshold = bbox_coder.get('score_threshold', None)
    num_classes = bbox_coder['num_classes']
    
    preds_dicts = {
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds
    }
    
    bbox_list = get_bboxes_custom_np(
        preds_dicts, pc_range, post_center_range,
        max_num, score_threshold, num_classes
    )
    
    results = []
    for bboxes, scores, labels in bbox_list:
        # Set class score thresholds
        class_score_thrs = {
            0: 0.3,  # Car
            1: 0.3,  # Truck
            2: 0.3,  # Construction vehicle
            3: 0.3,  # Bus
            4: 0.3,  # Trailer
            5: 0.3,  # Barrier
            6: 0.3,  # Motorcycle
            7: 0.3,  # Bicycle
            8: 0.3, # Pedestrian
            9: 0.3,  # Traffic cone
        }
        default_thr = score_thr
        
        keep_indices = []
        for i in range(len(scores)):
            cls_id = int(labels[i])
            thr = class_score_thrs.get(cls_id, default_thr)
            if scores[i] > thr:
                keep_indices.append(i)
        
        if len(keep_indices) == 0:
            results.append(format_bbox_result_np(
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64)
            ))
            continue
        
        keep_indices = np.array(keep_indices, dtype=np.int64)
        bboxes = bboxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # Circle NMS
        dist_thrs = {
            0: 2.0, 1: 3.0, 2: 2.5, 3: 4.0, 4: 3.0,
            5: 1.0, 6: 1.5, 7: 1.0, 8: 0.5, 9: 0.3,
        }
        
        if len(scores) > 0:
            keep_nms = circle_nms_np(bboxes, scores, labels, dist_thrs)
            if len(keep_nms) > 0:
                bboxes = bboxes[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]
            else:
                results.append(format_bbox_result_np(
                    np.zeros((0, 9), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64)
                ))
                continue
        
        results.append(format_bbox_result_np(bboxes, scores, labels))
    
    return results


def circle_nms_np(bboxes, scores, labels, dist_thrs):
    if len(bboxes) == 0:
        return np.array([], dtype=np.int64)
    
    keep = []
    order = np.argsort(scores)[::-1]
    bboxes = bboxes[order]
    scores = scores[order]
    labels = labels[order]
    
    pts = bboxes[:, :2]
    labels_np = labels
    
    suppressed = np.zeros(len(bboxes), dtype=bool)
    
    for i in range(len(bboxes)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        
        curr_cls = int(labels_np[i])
        radius = dist_thrs.get(curr_cls, 1.0)
        
        if i + 1 < len(bboxes):
            dists = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            idx_to_suppress = np.where(
                (dists < radius) & (labels_np[i+1:] == curr_cls)
            )[0]
            suppressed[i+1:][idx_to_suppress] = True
    
    return np.array(keep, dtype=np.int64)


def denormalize_img_np(img_array, img_norm_cfg):
    """Denormalize image array (C, H, W) to (H, W, C) BGR"""
    mean = np.array(img_norm_cfg.get('mean', [123.675, 116.28, 103.53]))
    std = np.array(img_norm_cfg.get('std', [58.395, 57.12, 57.375]))
    
    # (C, H, W) RGB -> (H, W, C) RGB
    if img_array.ndim == 3:
        img = img_array.transpose(1, 2, 0)
    else:
        img = img_array
    img = (img * std + mean)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def draw_bev_map(bboxes, labels, scores, pc_range, bev_size=(800, 800), score_thr=0.1):
    """Draw BEV (Bird's Eye View) map with detections
    
    Args:
        bboxes: (N, 9) numpy array, format: [x, y, z, w, l, h, yaw, vx, vy]
        labels: (N,) numpy array, class labels
        scores: (N,) numpy array, detection scores
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_size: (width, height) of BEV image
        score_thr: score threshold
    
    Returns:
        bev_img: (H, W, 3) numpy array, BEV visualization
    """
    bev_w, bev_h = bev_size # BEV image size
    bev_img = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 255  # White background
    
    # Draw grid
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Draw grid lines
    grid_color = (200, 200, 200)  # Light gray grid lines
    for i in range(-5, 6):
        x = x_min + (i + 5) * x_range / 10
        y = y_min + (i + 5) * y_range / 10
        # Vertical lines (y direction in LiDAR -> x direction in image)
        img_x = int((y - y_min) / y_range * bev_w)
        if 0 <= img_x < bev_w:
            cv2.line(bev_img, (img_x, 0), (img_x, bev_h), grid_color, 1)
        # Horizontal lines (x direction in LiDAR -> y direction in image, flipped)
        img_y = int((x_max - x) / x_range * bev_h)
        if 0 <= img_y < bev_h:
            cv2.line(bev_img, (0, img_y), (bev_w, img_y), grid_color, 1)
    
    # Draw center lines (ego vehicle position) - darker on white background
    center_x = int((0 - y_min) / y_range * bev_w)
    center_y = int((x_max - 0) / x_range * bev_h)
    cv2.line(bev_img, (center_x, 0), (center_x, bev_h), (150, 150, 150), 2)
    cv2.line(bev_img, (0, center_y), (bev_w, center_y), (150, 150, 150), 2)
    
    
    ego_length_px = 30  # pixels (representing ~4.5m, along x-axis rightward)
    ego_width_px = 12   # pixels (representing ~1.8m, along y-axis downward)
    
    ego_corners_local = np.array([
        [ego_length_px//2, -ego_width_px//2],   # front-top (head)
        [ego_length_px//2, ego_width_px//2],    # front-bottom
        [-ego_length_px//2, ego_width_px//2],   # back-bottom
        [-ego_length_px//2, -ego_width_px//2],  # back-top
    ], dtype=np.float32)
    

    rotation_angle_90 = np.pi / 2  # 90 degrees in radians
    cos_rot_90 = np.cos(rotation_angle_90)
    sin_rot_90 = np.sin(rotation_angle_90)
    rot_mat_90 = np.array([[cos_rot_90, -sin_rot_90], [sin_rot_90, cos_rot_90]])
    
    ego_corners_rotated_90 = ego_corners_local @ rot_mat_90.T
    
    ego_corners_rotated = ego_corners_rotated_90 @ rot_mat_90.T
    
    # Translate to image coordinates (center position)
    ego_corners = []
    for corner in ego_corners_rotated:
        corner_img_x = int(center_x + corner[0])
        corner_img_y = int(center_y + corner[1])
        ego_corners.append([corner_img_x, corner_img_y])
    ego_corners = np.array(ego_corners, dtype=np.int32)
    
    # Draw filled rectangle
    cv2.fillPoly(bev_img, [ego_corners], (0, 0, 255))  # Red filled
    cv2.polylines(bev_img, [ego_corners], True, (0, 0, 0), 2)  # Black outline
    

    arrow_length = ego_length_px // 2
    initial_direction = np.array([1.0, 0.0])  
    arrow_dir_rotated_90 = initial_direction @ rot_mat_90.T
    arrow_dir_rotated = arrow_dir_rotated_90 @ rot_mat_90.T
    arrow_end_x = int(center_x + arrow_length * arrow_dir_rotated[0])
    arrow_end_y = int(center_y + arrow_length * arrow_dir_rotated[1])
    cv2.arrowedLine(bev_img, (center_x, center_y), (arrow_end_x, arrow_end_y),
                   (0, 0, 0), 3, tipLength=0.3)  # Black arrow
    
    if len(bboxes) == 0:
        return bev_img
    
    if score_thr > 0:
        mask = scores > score_thr
        bboxes = bboxes[mask]
        labels = labels[mask]
        scores = scores[mask]
    
    if len(bboxes) == 0:
        return bev_img
    
    default_color = (255, 255, 255)
    

    for i in range(len(bboxes)):
        box = bboxes[i]
        label = int(labels[i])
        score = float(scores[i])
        color = CLASS_COLORS.get(label, default_color)
        
        x, y, z = box[0], box[1], box[2]  # center position
        w, l, h = box[3], box[4], box[5]  # width, length, height
        yaw = box[6]  # yaw angle
        
        yaw = yaw - np.pi / 2.0  # Subtract 90 degrees (counterclockwise)
        
        # Convert to image coordinates
        # Note: In LiDAR coordinate, x is forward, y is left, z is up
        # In BEV image (top-down view):
        #   - x (forward) -> image y (downward, flipped)
        #   - y (left) -> image x (rightward)
        # So: img_x = (y - y_min) / y_range * bev_w
        #     img_y = (x_max - x) / x_range * bev_h  (flip x to get top-down view)
        img_x = int((y - y_min) / y_range * bev_w)
        img_y = int((x_max - x) / x_range * bev_h)  # Flip x for top-down view
        
        # Skip if outside image
        if not (0 <= img_x < bev_w and 0 <= img_y < bev_h):
            continue
        
        # Calculate box dimensions in image space
        box_w_px = int(w / x_range * bev_w)
        box_l_px = int(l / y_range * bev_h)
        
        # Draw rotated rectangle
        # Calculate 4 corners of the box in LiDAR coordinates
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Box corners relative to center (in LiDAR frame: x forward, y left)
        corners_local = np.array([
            [l/2, w/2],   # front-right
            [l/2, -w/2],  # front-left
            [-l/2, -w/2], # back-left
            [-l/2, w/2]   # back-right
        ])
        
        # Rotate corners
        rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_rotated = corners_local @ rot_mat.T
        
        # Translate to world coordinates and convert to image space
        corners_img = []
        for corner in corners_rotated:
            corner_x = x + corner[0]  # x in LiDAR (forward)
            corner_y = y + corner[1]  # y in LiDAR (left)
            corner_img_x = int((corner_y - y_min) / y_range * bev_w)  # y -> img_x
            corner_img_y = int((x_max - corner_x) / x_range * bev_h)  # x -> img_y (flipped)
            corners_img.append([corner_img_x, corner_img_y])
        
        corners_img = np.array(corners_img, dtype=np.int32)
        
        # Draw filled polygon (semi-transparent on white background)
        overlay = bev_img.copy()
        cv2.fillPoly(overlay, [corners_img], color)
        cv2.addWeighted(overlay, 0.5, bev_img, 0.5, 0, bev_img)
        # Draw outline (black on white background)
        cv2.polylines(bev_img, [corners_img], True, (0, 0, 0), 2)
        
        # Draw direction arrow (forward direction) - black on white
        # In LiDAR: forward is +x, left is +y
        # In BEV image: x -> img_y (flipped), y -> img_x
        # So rotation: img_x += sin(yaw) * length, img_y -= cos(yaw) * length
        arrow_length = max(box_l_px // 2, 10)
        arrow_end_x = int(img_x + arrow_length * sin_yaw)   # y component -> img_x
        arrow_end_y = int(img_y - arrow_length * cos_yaw)  # x component -> img_y (flipped)
        cv2.arrowedLine(bev_img, (img_x, img_y), (arrow_end_x, arrow_end_y),
                       (0, 0, 0), 2, tipLength=0.3)  # Black arrow
        
        # Draw center point
        cv2.circle(bev_img, (img_x, img_y), 3, (0, 0, 0), -1)  # Black center point
    
    # Rotate BEV map counterclockwise by 90 degrees (map only, not text)
    center = (bev_w // 2, bev_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)  # 90 degrees counterclockwise
    bev_img = cv2.warpAffine(bev_img, rotation_matrix, (bev_w, bev_h), borderValue=(255, 255, 255))
    
    # Flip horizontally to fix mirror effect
    bev_img = cv2.flip(bev_img, 1)  # 1 for horizontal flip
    
    text = 'BEV Map'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = bev_w - text_width - 10  
    text_y = text_height + 10  
    cv2.putText(bev_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    return bev_img


def visualize_results_np(img, result, lidar2img, img_norm_cfg, class_names, score_thr=0.3, pc_range=None):
    num_cams = img.shape[1] if img.ndim == 5 else 1
    raw_imgs = [denormalize_img_np(img[0, cam_idx], img_norm_cfg) for cam_idx in range(num_cams)]
    boxes_3d = result.get('boxes_3d')
    scores_3d = result.get('scores_3d')
    labels_3d = result.get('labels_3d')
    vis_imgs = []
    boxes_3d_for_bev = labels_3d_for_bev = scores_3d_for_bev = None

    if boxes_3d is not None and len(boxes_3d) > 0:
        mask = (scores_3d > score_thr) if (score_thr > 0 and scores_3d is not None) else np.ones_like(scores_3d, dtype=bool)
        if np.any(mask):
            boxes_3d = boxes_3d[mask]
            scores_3d = scores_3d[mask]
            labels_3d = labels_3d[mask]
        boxes_3d_for_bev = boxes_3d.copy()
        labels_3d_for_bev = labels_3d.copy()
        scores_3d_for_bev = scores_3d.copy()
        for cam_idx, vis_img in enumerate(raw_imgs):
            vis_img = vis_img.copy()
            if lidar2img.shape[1] > cam_idx:
                cam_lidar2img = lidar2img[0, cam_idx]
                for box, label in zip(boxes_3d, labels_3d):
                    color = CLASS_COLORS.get(int(label), (255, 255, 255))
                    try:
                        vis_img = draw_bbox3d_on_img_custom_np(box[None], vis_img, cam_lidar2img, color=color, thickness=2)
                    except Exception:
                        pass
            vis_imgs.append(vis_img)
    else:
        vis_imgs = raw_imgs

    if pc_range is None:
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    if boxes_3d_for_bev is not None and len(boxes_3d_for_bev) > 0:
        bev_size = (vis_imgs[0].shape[0], vis_imgs[0].shape[0]) if vis_imgs else (800, 800)
        bev_img = draw_bev_map(boxes_3d_for_bev, labels_3d_for_bev, scores_3d_for_bev, pc_range, bev_size=bev_size, score_thr=score_thr)
    else:
        bev_size = (vis_imgs[0].shape[0], vis_imgs[0].shape[0]) if vis_imgs else (800, 800)
        bev_img = np.full((bev_size[1], bev_size[0], 3), 255, np.uint8)
        cv2.putText(bev_img, 'BEV Map (No Detections)', (10, bev_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if len(vis_imgs) == 6:
        target_height = max(img.shape[0] for img in vis_imgs)
        resized_imgs = [img if img.shape[0] == target_height else cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height)) for img in vis_imgs]

        reordered_imgs = [
            resized_imgs[2], resized_imgs[0], resized_imgs[1],  
            cv2.flip(resized_imgs[4], 1), cv2.flip(resized_imgs[3], 1), cv2.flip(resized_imgs[5], 1)  
        ]
        top_row = np.hstack(reordered_imgs[:3])
        bottom_row = np.hstack(reordered_imgs[3:])
        left_side = np.vstack([top_row, bottom_row])
        bev_img = cv2.resize(bev_img, (int(bev_img.shape[1] * left_side.shape[0] / bev_img.shape[0]), left_side.shape[0]))
        vis_img = np.hstack([left_side, bev_img])
    elif len(vis_imgs) > 1:
        target_height = max(img.shape[0] for img in vis_imgs)
        resized_imgs = [img if img.shape[0] == target_height else cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height)) for img in vis_imgs]
        if bev_img.shape[0] != target_height:
            bev_img = cv2.resize(bev_img, (int(bev_img.shape[1] * target_height / bev_img.shape[0]), target_height))
        vis_img = np.hstack([np.hstack(resized_imgs), bev_img])
    else:
        cam_img = vis_imgs[0] if vis_imgs else bev_img
        if bev_img.shape[0] != cam_img.shape[0]:
            bev_img = cv2.resize(bev_img, (int(bev_img.shape[1] * cam_img.shape[0] / bev_img.shape[0]), cam_img.shape[0]))
        vis_img = np.hstack([cam_img, bev_img]) if vis_imgs else bev_img

    return vis_img


def create_video_from_images(image_dir, output_video_path, fps=3):
    import subprocess
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        return
    
    first_img = cv2.imread(osp.join(image_dir, image_files[0]))
    if first_img is None:
        return
    
    height, width = first_img.shape[:2]
    
    max_width, max_height = 1920, 1080
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        width, height = int(width * scale), int(height * scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for img_file in tqdm(image_files, desc=f"Creating video: {osp.basename(output_video_path)}"):
        img_path = osp.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video_writer.write(img)
    
    video_writer.release()

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
    scene_index_path = osp.join(args.data_dir, 'scene_index.json')
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
    
    scene_results = defaultdict(list)
    
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
        
        # Process all frames in this scene
        for local_idx, frame_idx in enumerate(tqdm(sample_indices, desc=f"Scene {scene_name}")):
            # Load data
            img, lidar2img, can_bus, meta = load_data(args.data_dir, scene_name, frame_idx)
            
            # Process can_bus (compute delta)
            curr_can_bus_np = can_bus[0]  # (18,)
            
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
            
            # Run inference
            onnx_outputs = onnx_session.run(None, onnx_inputs)
            bev_embed, all_cls_scores, all_bbox_preds = onnx_outputs
            
            prev_frame_info['prev_bev'] = bev_embed
            
            # Post-process
            results = post_process_outputs_np(
                all_cls_scores, all_bbox_preds, config, args.score_thr
            )
            
            # Visualize
            img_norm_cfg = config['img_norm']
            class_names = config['dataset']['class_names']
            pc_range = config['model']['bbox_coder']['pc_range']
            vis_img = visualize_results_np(
                img, results[0], lidar2img, img_norm_cfg, class_names, args.score_thr, pc_range=pc_range
            )
            
            scene_results[scene_name].append({
                'frame_idx': local_idx,
                'result': results[0],
                'vis_img': vis_img,
                'meta': meta
            })
    
    # Save results
    for scene_name, frames in tqdm(scene_results.items(), desc="Save scene results"):
        scene_dir = osp.join(args.output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        images_dir = osp.join(scene_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        for local_idx, frame_data in enumerate(frames):
            vis_img = frame_data['vis_img']
            
            if vis_img is None:
                continue
            
            if not isinstance(vis_img, np.ndarray):
                vis_img = np.array(vis_img)
            
            if vis_img.dtype != np.uint8:
                vis_img = (vis_img * 255).astype(np.uint8) if vis_img.max() <= 1.0 else vis_img.astype(np.uint8)
            
            if len(vis_img.shape) == 3 and vis_img.shape[0] in (1, 3):
                vis_img = vis_img.transpose(1, 2, 0)
            
            if vis_img.shape[0] > 0 and vis_img.shape[1] > 0:
                cv2.imwrite(osp.join(images_dir, f'frame_{local_idx:06d}.png'), vis_img)
        
        create_video_from_images(images_dir, osp.join(scene_dir, f'{scene_name}_result.mp4'), args.fps)
        print(f"✓ Scene {scene_name}: {len(frames)} frames, video: {osp.join(scene_dir, f'{scene_name}_result.mp4')}")


if __name__ == '__main__':
    main()

