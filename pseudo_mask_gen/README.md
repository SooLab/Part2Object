
# Data Preprocessing
<div id=Data_Preprocessing>




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