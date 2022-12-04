# Influence-Based Mini-Batching

<p align="center">
<img src="https://user-images.githubusercontent.com/9202783/205024258-fa39efe0-0e50-4959-92f5-59f10462bddb.png" width="600">
</p>

Reference implementation of influence-based mini-batching (IBMB), as proposed in:

[**Influence-Based Mini-Batching for Graph Neural Networks**](https://www.cs.cit.tum.de/daml/ibmb)  
Johannes Gasteiger*, Chendi Qian*, Stephan Günnemann  
Published at LoG 2022 (oral)

*Both authors contributed equally. 

## Environment setup

The experiments in the paper were run on an Nvidia GTX1080Ti. Note that newer GPUs might not support this version of CUDA.

We set up an Anaconda environment with Python 3.7

```angular2html
conda create -n ibmb python=3.7
conda activate ibmb
```

We use pytorch 1.8.1 with cudatoolkit 10.2

`conda install pytorch==1.8.1 cudatoolkit=10.2 -c pytorch`

Install PyG 1.7.0

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
pip install psutil
```

## Replicate experiments

To replicate Table 7 and Figure 3, run with the yaml files under `configs/main_exps/`, e.g.

`python run_ogbn.py with configs/main_exps/gcn/arxiv/ppr_based.yaml`

To replicate Figure 2, run with the yaml files under `configs/infer` and `configs/full_infer`. We recommend using [seml](https://github.com/TUM-DAML/seml) to tune the hyperparameters in large scale. 

See the paper's appendix for more information about tuning hyperparameters of IBMB and the baselines. 
