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
- [ ] Installation
- [ ] Data download and preprocessing
- [ ] Upload processed data
- [ ] Pseudo Mask Generation
- [ ] Self-Training
- [ ] Upload pretrained models
- [ ] Evaluation


# Installation
<div id=Installation>

### Conda

```
# create conda environment
conda create -n Part2Object python=3.8 -y
conda activate Part2Object

# install pytorch (other versions may also work)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# other requirements
git clone https://github.com/ChengShiest/Part2Object.git
cd Part2Object
pip install -r requirements.txt
```

# Data Preprocessing
<div id=Data_Preprocessing>




# Pseudo Mask Generation
<div id=Pseudo_Mask_Generation>



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


