# Part2Object

By [Cheng Shi](https://chengshiest.github.io/), Yulin Zhang, Bin Yang, [Jiajin Tang](https://toneyaya.github.io/), [Yuexin Ma](https://yuexinma.me/) and
[Sibei Yang](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

The official PyTorch implementation of the "Part2Object: Hierarchical Unsupervised 3D
Instance Segmentation".

# Main results
### [ScanNet v2](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap)

| Dataset | AP | AP_50 | AP_25 | Config | Checkpoint :floppy_disk: | Scores :chart_with_upwards_trend: | Visualizations :telescope:
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ScanNet val  | 55.2 | 73.7 | 83.5 | [config](scripts/scannet/scannet_val.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_val.ckpt) | [scores](./docs/detailed_scores/scannet_val.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/scannet/val/)
| ScanNet test | 56.6 | 78.0 | 87.0 | [config](scripts/scannet/scannet_benchmark.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_benchmark.ckpt) | [scores](http://kaldir.vc.in.tum.de/scannet_benchmark/result_details?id=1081) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/scannet/test/)

### [S3DIS](http://buildingparser.stanford.edu/dataset.html) (pretrained on ScanNet train+val)
# Installation
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

# Usage

# Citing Part2Object

If you find Part2Object useful in your research, please consider citing:
```
inproceedings{
  shi2024Part2Object,
  title={Part2Object: Hierarchical Unsupervised 3D Instance Segmentation},
}
```