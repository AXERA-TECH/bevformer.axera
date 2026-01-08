# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)

@parse_args('v','v','i','i','b')
def grid_sampler(g, input, grid, mode_enum, padding_mode_enum, align_corners):
    mode_str = ['bilinear', 'nearest', 'bicubic'][mode_enum]
    padding_str = ['zeros', 'border', 'reflection'][padding_mode_enum]
    return g.op('com.microsoft::GridSample',input,grid,mode_s=mode_str,padding_mode_s=padding_str,align_corners_i=align_corners)


def export_onnx(model, data_loader, output_file):
    register_custom_op_symbolic("::grid_sampler", grid_sampler, 13)
    model.eval()

    for data in data_loader:
        
        with torch.no_grad():
            # img: (B, N, C, H, W)
            img = data['img'][0].data[0].float()
            if img.dim() == 4:
                img = img.unsqueeze(0)  # (1, N, C, H, W)
            
            # lidar2img: (B, N, 4, 4)
            lidar2img = torch.from_numpy(np.array(data['img_metas'][0].data[0][0]['lidar2img'])).float()
            if lidar2img.dim() == 3:  # (N, 4, 4)
                lidar2img = lidar2img.unsqueeze(0)  # (1, N, 4, 4)
            elif lidar2img.dim() == 2:  # (4, 4)
                lidar2img = lidar2img.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
            
            # can_bus: (B, 18)
            can_bus = torch.from_numpy(data['img_metas'][0].data[0][0]['can_bus']).float()
            if can_bus.dim() == 1:  # (18,)
                can_bus = can_bus.unsqueeze(0)  # (1, 18)
            

            if hasattr(model, 'module'):
                export_model = model.module
            else:
                export_model = model
            
            # prev_bev: (bev_h*bev_w, B, embed_dims)
            pts_bbox_head = export_model.pts_bbox_head
            bev_h = pts_bbox_head.bev_h
            bev_w = pts_bbox_head.bev_w
            embed_dims = pts_bbox_head.embed_dims
            B = img.shape[0]
            prev_bev = torch.zeros((bev_h * bev_w, B, embed_dims), dtype=img.dtype, device=img.device)
            
            if not hasattr(export_model, 'forward_onnx'):
                raise RuntimeError("Model does not have forward_onnx method. Please ensure BEVFormer detector has forward_onnx implemented.")
            
            class ONNXExportWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, img, can_bus, lidar2img, prev_bev):
                    return self.model.forward_onnx(img, can_bus, lidar2img, prev_bev)
            
            wrapped_model = ONNXExportWrapper(export_model)
            inputs = (img, can_bus, lidar2img, prev_bev)

            raw_output_file = output_file.replace('.onnx', '_raw.onnx')
            torch.onnx.export(
                wrapped_model,
                inputs,
                raw_output_file,
                export_params=True,
                do_constant_folding=False,
                verbose=False,
                opset_version=13,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                input_names=['img', 'can_bus', 'lidar2img', 'prev_bev'],
                output_names=['bev_embed', 'outputs_classes', 'outputs_coords'],
                dynamic_axes={
                    'img': {0: 'batch_size'},
                    'can_bus': {0: 'batch_size'},
                    'lidar2img': {0: 'batch_size'},
                    'prev_bev': {0: 'bev_h*bev_w', 1: 'batch_size'},
                } if False else None,  
            )

            import onnx
            from onnx import helper
            from onnxsim import simplify
            
            onnx_model = onnx.load(raw_output_file)
            
            grid_sample_fixed = False
            for node in onnx_model.graph.node:
                if node.op_type == "GridSample" and node.domain == "com.microsoft":
                    node.domain = ""
                    new_attrs = []
                    for attr in node.attribute:
                        if attr.name == "mode_s":
                            new_attr = helper.make_attribute("mode", attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s)
                            new_attrs.append(new_attr)
                        elif attr.name == "padding_mode_s":
                            new_attr = helper.make_attribute("padding_mode", attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s)
                            new_attrs.append(new_attr)
                        elif attr.name == "align_corners_i":
                            new_attr = helper.make_attribute("align_corners", attr.i)
                            new_attrs.append(new_attr)
                        else:
                            new_attrs.append(attr)
                    node.ClearField("attribute")
                    node.attribute.extend(new_attrs)
                    grid_sample_fixed = True
                    print(f"Converted com.microsoft::GridSample to standard GridSample: {node.name}")
            
            if grid_sample_fixed:
                while len(onnx_model.opset_import) > 0:
                    onnx_model.opset_import.pop()
                onnx_model.opset_import.extend([helper.make_opsetid("", 16)])
                print("Upgraded ONNX opset to 16")
            
            onnx.save(onnx_model, raw_output_file)
            print(f"Updated ONNX file saved: {raw_output_file}")
            
            reshape_fixed = False
            value_info_dict = {}
            for vi in onnx_model.graph.value_info:
                value_info_dict[vi.name] = vi
            for vi in onnx_model.graph.input:
                value_info_dict[vi.name] = vi
            for vi in onnx_model.graph.output:
                value_info_dict[vi.name] = vi
            
            initializer_dict = {init.name: init for init in onnx_model.graph.initializer}
            
            for node in onnx_model.graph.node:
                if node.op_type == "Reshape" and len(node.input) > 1:
                    shape_input = node.input[1]
                    
                    is_constant = shape_input in initializer_dict
                    
                    if not is_constant:
                        if node.output[0] in value_info_dict:
                            output_vi = value_info_dict[node.output[0]]
                            output_shape = []
                            can_fix = True
                            
                            for dim in output_vi.type.tensor_type.shape.dim:
                                if dim.dim_value > 0:
                                    output_shape.append(dim.dim_value)
                                else:
                                    can_fix = False
                                    print(f"Warning: Reshape {node.name} has dynamic dimension, skipping fix")
                                    break
                            
                            if can_fix and len(output_shape) > 0 and all(s > 0 for s in output_shape):
                                shape_initializer = helper.make_tensor(
                                    name=shape_input,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(output_shape)],
                                    vals=output_shape
                                )
                                onnx_model.graph.initializer.append(shape_initializer)
                                initializer_dict[shape_input] = shape_initializer
                                reshape_fixed = True
                                print(f"Fixed Reshape {node.name}: shape parameter {shape_input} -> {output_shape}")
            
            if reshape_fixed:
                onnx.save(onnx_model, raw_output_file)
                print(f"Reshape-fixed ONNX file saved: {raw_output_file}")
            
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, output_file)
 
            print(f"Simplified ONNX file has been saved in {output_file}")
            exit()
            return {0:'1'}

def single_onnx_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.forward_onnx(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
