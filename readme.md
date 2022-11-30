Official GitHub repo for paper __Influence-Based Mini-Batching for Graph Neural Networks__ accepted at LoG conference 2022.

## Environment setup

Officially we run our code with a single card Nvidia GTX1080Ti. Note that new GPUs e.g. RTX 3090Ti may not support older cuda version.

We set up an anaconda environment with python 3.7

```angular2html
conda create -n ibmb python=3.7
conda activate ibmb
```

We use pytorch 1.8.1 with cudatoolkit 10.2

`conda install pytorch==1.8.1 cudatoolkit=10.2 -c pytorch`

Installation of PyG 1.7.0

```
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl  # please do not use older versions!
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.7.0
```

Feel free to use other versions of PyG, but please be aware that some class keywords might be different. 

Other packages

```angular2html
pip install ogb
pip install sacred
pip install seml
pip install parallel-sort==0.0.3  # please do not use new versions
pip install python-tsp
```