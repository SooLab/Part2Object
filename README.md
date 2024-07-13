# Part2Object

By [Cheng Shi](https://chengshiest.github.io/), Yulin Zhang, Bin Yang, [Jiajin Tang](https://toneyaya.github.io/), [Yuexin Ma](https://yuexinma.me/) and
[Sibei Yang](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

The official PyTorch implementation of the "Part2Object: Hierarchical Unsupervised 3D
Instance Segmentation".



# README structure
- [Installation](#Installation)
- [Self-Training and Data-efficient](#Self-Training)
- [Main Result and Available Resources](#resource)

# Roadmap
- [x] Installation
- [x] Data download and Preprocessing
- [x] Pseudo Mask Generation
- [x] Upload Pseudo Mask Result
- [x] Self-Training
- [x] Upload Pretrained Models


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

conda activate part2object

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







# Self-Training and Data-efficient
<div id=Self-Training>

You can download our generated pseudo-labels [here](https://drive.google.com/file/d/19lsRVYrE3rgTObndUnTq-MSb4nO72BLK/view?usp=sharing) or generate by yourself with [our code](https://github.com/ChengShiest/Part2Object/tree/main/pseudo_mask_gen).

### Train &  Evaluation
To train or test the results of Part2Object, modify the file paths appropriately and run the following scripts.
```bash
sh scripts/scannet/scannet_val.sh
```
### Train data efficient model
After getting the base model trained with pseudo-labeling, you can train the data efficient model by modifying the following script appropriately.
```bash
sh scripts/scannet/scannet_df.sh
```


# Main Result and Available Resources 
<div id=resource>

### Pseudo Label
| Methods     | AP25 | AP50 | mAP  |            |
| ----------- | ---- | ---- | ---- | ---------- |
| Part2Object | 55.1 | 26.8 | 12.6 | [result](https://drive.google.com/file/d/19lsRVYrE3rgTObndUnTq-MSb4nO72BLK/view?usp=sharing) |


# Model

| Methods     | AP50 /  (0% data)|  | AP50 / 1% data|   | AP50 / 5% data|   | AP50 / 10% data|   | AP50 / 20% data|   |
| ----------- | ------------------------------| -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | --------------- | --------------- |
| Part2Object | 32.6           | [weight](https://drive.google.com/file/d/19lsRVYrE3rgTObndUnTq-MSb4nO72BLK/view?usp=sharing)                | 44.1  | [weight](https://drive.google.com/file/d/16Q7KUbr8GSj0psnYGHQId7TN2k6zuIYr/view?usp=sharing)             | 64.2         | [weight](https://drive.google.com/file/d/1ZaOwSOs9m4QyvlSS779s6JZebBGGjqGo/view?usp=sharing)     | 68.0        | [weight](https://drive.google.com/file/d/1uOOcdTPTir9DxQjlSKjc_zekR5wb_vAu/view?usp=sharing)          | 72.1       | [weight](https://drive.google.com/file/d/19lsRVYrE3rgTObndUnTq-MSb4nO72BLK/view?usp=sharing)          |



## Acknowledgement

We thank [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.28.1) for their valuable code bases.


If you find Part2Object useful in your research, please consider citing:
```
inproceedings{
  shi2024Part2Object,
  title={Part2Object: Hierarchical Unsupervised 3D Instance Segmentation},
}
```
