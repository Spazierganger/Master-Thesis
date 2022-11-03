# Potential Energy Surface Network (PESNet)
Reference implementation of PESNet as proposed in

Ab-Initio Potential Energy Surfaces by Pairing GNNs with Neural Wave Functions <br/>
by Nicholas Gao, Stephan Günnemann


## Run the code
First install [JAX](https://github.com/google/jax) and the correct [CUDA Toolkit](https://anaconda.org/anaconda/cudatoolkit) and [CUDNN](https://anaconda.org/anaconda/cudnn), then this package via
```bash
pip install -e .
```
You can now train a model, e.g., H2, via a config file
```bash
python train.py with configs/systems/h2.yaml print_progress=True
```
You can overwrite parameters either by [CLI](https://sacred.readthedocs.io/en/stable/command_line.html) or via the config file.
All progress is tracked on tensorboard.

### Running multiple experiments
To run multiple experiments, we recommend to use the [`seml`](https://github.com/TUM-DAML/seml) library.
However, we also provide `YAML` files to train each system.


## Reproducing results from the paper
### H4+:
```bash
python train.py with configs/systems/h4plus.yaml print_progress=True
```
### Hydrogen rectangle:
```bash
python train.py with configs/systems/h4.yaml print_progress=True
```
### Hydrogen chain:
```bash
python train.py with configs/systems/h10.yaml print_progress=True
```
### Nitrogen molecule:
```bash
python train.py with configs/systems/n2.yaml print_progress=True pesnet.ferminet_params.determinants=32
```
### Cyclobutadiene:
```bash
python train.py with configs/systems/cyclobutadiene.yaml \
    print_progress=True \
    pesnet.ferminet_params.determinants=32 \
    pesnet.ferminet_params.hidden_dims='[[512, 32, True], [512, 32, True], [512, 32, True], [512, 32, True]]'
```

