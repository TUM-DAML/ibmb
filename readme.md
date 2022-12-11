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

The experiments in the paper were run on an Nvidia GTX1080Ti. If you have other GPUs, please check your GPU and cuda compatibility. 

We set up an Anaconda environment with Python 3.8

```angular2html
conda create -n ibmb python=3.8
conda activate ibmb
```

We use pytorch 1.10.1 with cudatoolkit 11.3

`conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge`

Install PyG 2.0.3. Feel free to use other versions, just be careful of some updates in classes and keywords.

```
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl  # please do not use older versions < 0.6.12
pip install torch-geometric==2.0.3
```

Other packages

```angular2html
pip install ogb
pip install sacred
pip install seml
pip install parallel-sort==0.0.3  # please do not use new versions
pip install python-tsp
pip install psutil
pip install numba
```

## Run experiments

We provide a couple of demo configs to play with.

To replicate Table 7 and Figure 3, run with the yaml files under `configs/main_exps/`, e.g.

`python run_ogbn.py with configs/main_exps/gcn/arxiv/ibmb_node.yaml`

To replicate Figure 2, run with the yaml files under `configs/infer`. You have to provide grid search parameters yourself. We recommend using [seml](https://github.com/TUM-DAML/seml) to tune the hyperparameters in large scale. e.g.

`python infer.py with configs/infer/gcn/arxiv/ibmb_node.yaml`

See the paper's appendix for more information about tuning hyperparameters of IBMB and the baselines. 

For the largest dataset `ogbn-papers100M`, we recommend you to run the notebook `dataprocess_100m.ipynb` first then proceed with `run_papers100m.py`. In order to perform full graph inference on such large dataset, we provide some tricks of tensor chunking, see `run_papers100m_full_infer.py` for more details. You need at least 256GB RAM for the `ogbn-papers100M` experiments. 