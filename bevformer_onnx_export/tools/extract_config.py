#!/usr/bin/env python3
import argparse
import json
from mmcv import Config


def extract_config(cfg_path, output_path):

    cfg = Config.fromfile(cfg_path)
    

    config_dict = {}
    
    # model configuration
    if 'model' in cfg:
        model_cfg = cfg.model
        if 'pts_bbox_head' in model_cfg:
            bbox_head = model_cfg.pts_bbox_head
            config_dict['model'] = {
                'bev_h': bbox_head.get('bev_h', 200),
                'bev_w': bbox_head.get('bev_w', 200),
                'num_query': bbox_head.get('num_query', 900),
                'num_classes': bbox_head.get('num_classes', 10),
            }
            
            # Transformer configuration
            if 'transformer' in bbox_head:
                transformer = bbox_head.transformer
                config_dict['model']['transformer'] = {
                    'embed_dims': transformer.get('embed_dims', 256),
                    'bev_h': transformer.get('bev_h', bbox_head.get('bev_h', 200)),
                    'bev_w': transformer.get('bev_w', bbox_head.get('bev_w', 200)),
                }
            
            # BBox coder configuration
            if 'bbox_coder' in bbox_head:
                bbox_coder = bbox_head.bbox_coder
                config_dict['model']['bbox_coder'] = {
                    'pc_range': bbox_coder.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                    'post_center_range': bbox_coder.get('post_center_range', [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]),
                    'max_num': bbox_coder.get('max_num', 300),
                    'num_classes': bbox_coder.get('num_classes', 10),
                }
    
    # dataset configuration
    if 'data' in cfg and 'test' in cfg.data:
        test_cfg = cfg.data.test
        if isinstance(test_cfg, dict):
            config_dict['dataset'] = {
                'class_names': test_cfg.get('classes', [
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                ]),
            }
        elif isinstance(test_cfg, list) and len(test_cfg) > 0:
            config_dict['dataset'] = {
                'class_names': test_cfg[0].get('classes', [
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                ]),
            }
    
    # image normalization configuration (from pipeline)
    if 'img_norm_cfg' in cfg:
        config_dict['img_norm'] = {
            'mean': cfg.img_norm_cfg.get('mean', [123.675, 116.28, 103.53]),
            'std': cfg.img_norm_cfg.get('std', [58.395, 57.12, 57.375]),
            'to_rgb': cfg.img_norm_cfg.get('to_rgb', True),
        }
    else:
        # default value
        config_dict['img_norm'] = {
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True,
        }
    
    # Point cloud range
    if 'point_cloud_range' in cfg:
        config_dict['point_cloud_range'] = cfg.point_cloud_range
    else:
        config_dict['point_cloud_range'] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # save as JSON
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"configuration saved to: {output_path}")
    print(f"configuration content:")
    print(json.dumps(config_dict, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract config for inference')
    parser.add_argument('config', help='config file path such as ./projects/configs/bevformer/bevformer_tiny.py')
    parser.add_argument('--output', default='./inference_config.json', help='output JSON file path')
    return parser.parse_args()


def main():
    args = parse_args()
    extract_config(args.config, args.output)


if __name__ == '__main__':
    main()

