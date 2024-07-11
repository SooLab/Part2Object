
# Data Preprocessing
<div id=Data_Preprocessing>

Download the ScanNet dataset from  [here](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation)  , and move it to ./data

### Apply VCCS Algorithm

```bash
python vccs/data_prepare_ScanNet.py --data_path "PATH_TO_RAW_SCANNET_DATASET"
```
This code will preprcocess ScanNet and put it under ./data/ScanNet/processed

```bash
python vccs/initialSP_prepare_ScanNet.py
```
This code will construct init segments on ScanNet and put it under ./data/ScanNet/initial_superpoints

### Get DINO features of Pointclouds
Download DINO model and put it in ./DINO 
```bash
python project_feature.py
```
This code will get the DINO features for each point cloud and store them in ./DINO_point_feats

### Get Point Distances 
```bash
python get_dis_matrix.py
```
This code calculates the shortest distance between the initial segments and stores it in ./dis_matrixes_initseg

### Get Bbox Prior
```bash
python get bbox prior/scene_frame_ncut.py --scannet_frames data/Scannetv2/frames_square --immedia_data_dir PATH_TO_STORE_INTERMEDIATE_RESULTS

python get bbox prior/scene_frame_merge.py --immedia_data_dir PATH_TO_STORE_INTERMEDIATE_RESULTS --scannet_frames data/Scannetv2/frames_square --val_data_dir data/Scannetv2 --output_dir DIR_TO_OUTPUT
```
This code will calculate Bbox Prior and store it in ./DIR_TO_OUTPUT


# Pseudo Mask Generation
<div id=Pseudo_Mask_Generation>

### Part2Object : hierarchical clustering

You can obtain the hierarchical clustering results using `topk_merge.py`. To do this, you need to specify the file path to the results produced in the Data Preprocessing section and also specify an output directory.

Here's an example:

```bash
python topk_merge.py --input /path/to/preprocessed/data --output /path/to/output/directory
```

Replace `/path/to/preprocessed/data` with the path to your preprocessed data and `/path/to/output/directory` with the path to the directory where you want to save the results.

### Post Processing

You can use `post_pro.py` for post-processing to eliminate noise points. In addition to specifying the `input` and `output` parameters as with `topk_merge.py`, you also need to specify `output-processed` to store the post-processing results. Here's how you can do it:

Here's an example:

```bash
python post_pro.py --input /path/to/input/data --output /path/to/output/directory --output-processed /path/to/post-processed/results
```
Replace `/path/to/input/data` with the path to your input data, `/path/to/output/directory` with the path to the directory where you want to save the intermediate results, and `/path/to/post-processed/results` with the path to the directory where you want to save the post-processing results.
