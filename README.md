# Master thesis: Scalable Neural Networks for Solving Many-Electron Schrödinger Equations

## Install dependencies

```
conda create -n pesnet python=3.9

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install -e .

conda install cudatoolkit cudnn

conda install -c conda-forge cudatoolkit-dev

pip install pyscf==2.0.1

conda env config vars set LD_LIBRARY_PATH=/nfs/homedirs/qian/miniconda3/envs/pesnet/lib

pip install ipywidgets
```

## run code

```angular2html
python train.py with configs/pesnet_eanet.yaml
```