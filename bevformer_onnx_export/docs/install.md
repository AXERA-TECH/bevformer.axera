# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
# conda create -n open-mmlab python=3.8 -y
conda create -n open-mmlab -c conda-forge python=3.8 -y
conda activate open-mmlab
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c conda-forge gcc_linux-64=10 gxx_linux-64=10 libgcc-ng=10
conda install -c conda-forge python=3.8
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**f. Install Detectron2 and Timm.**
```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.21 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
pip install "numpy>=1.21,<1.24"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**g. Clone BEVFormer.**
```
git clone https://github.com/fundamentalvision/BEVFormer.git
```

**h. Prepare pretrained models.**
```shell
cd bevformer
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

note: this pretrained model is the same model used in [detr3d](https://github.com/WangYueFt/detr3d)
