# Part2Object

By [Cheng Shi](https://chengshiest.github.io/), Yulin Zhang, Bin Yang, [Jiajin Tang](https://toneyaya.github.io/), [Yuexin Ma](https://yuexinma.me/) and
[Sibei Yang](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

The official PyTorch implementation of the "Part2Object: Hierarchical Unsupervised 3D
Instance Segmentation".

If you find Part2Object useful in your research, please consider citing:
```
inproceedings{
  shi2024Part2Object,
  title={Part2Object: Hierarchical Unsupervised 3D Instance Segmentation},
}
```

# README structure
- [Installation](#Installation) - setting up a conda environment
- [Data Preprocessing](#Data_Preprocessing) - we primarily use the ScanNet dataset, we have to preprocess them to get aligned point clouds and 2D images
- [Pseudo Mask Generation](#Pseudo_Mask_Generation) - we generate pseudo masks using self-supervised features and extract them for self-training
- [Self-Training and Data-efficient](#Self-Training) - we mostly follow the training procedure of Mask3D, but we use the pseudo masks, noise robust losses, self-training iterations, and a class-agnostic evaluation

# Roadmap
- [x] Installation
- [ ] Data download and Preprocessing
- [ ] Pseudo Mask Generation
- [ ] Upload Pseudo Mask Result
- [ ] Self-Training
- [ ] Upload Pretrained Models
- [ ] Evaluation


# Installation
<div id=Installation>

We follow [Mask3D](https://github.com/JonasSchult/Mask3D) to install our environment. 

### Dependencies
The main dependencies of the project are the following:
```yaml
python: 3.10.9
cuda: 11.3
```
You can set up a conda environment as follows
```
# Some users experienced issues on Ubuntu with an AMD CPU
# Install libopenblas-dev (issue #115, thanks WindWing)
# sudo apt-get install libopenblas-dev

export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

conda env create -f environment.yml

conda activate mask3d_cuda113

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

mkdir third_party
cd third_party

git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

cd ..
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

cd ../../pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2
```


### Main Result and Available Resources 

| Methods     | AP25 | AP50 | mAP  |            |
| ----------- | ---- | ---- | ---- | ---------- |
| Part2Object | 55.1 | 26.8 | 12.6 | [result]() |




# Self-Training and Data-efficient
<div id=Self-Training>

| Methods     | AP50 / self-training (0% data) | AP50 / 1% data | AP50 / 5% data | AP50 / 10% data | AP50 / 20% data |
| ----------- | ------------------------------ | -------------- | -------------- | --------------- | --------------- |
| Part2Object | 32.6                           | 44.1           | 64.2           | 68.0            | 72.1            |
|             | [weight]()                     | [weight]()     | [weight]()     | [weight]()      | [weight]()      |


## Acknowledgement

We thank [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.28.1) for their valuable code bases.


