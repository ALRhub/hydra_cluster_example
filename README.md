# Stamp Forming Simulation
This is the source code for simulations of stamp forming processes using Graph Network Simulators.

## Installation
Clone this repository. Then setup a python virtual environment with python 3.11. We recommend using [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), but conda or venv should also work.

Inside the environment, install Pytorch using

```pip3 install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

Then install the other torch dependencies using

```
pip install --upgrade torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install --upgrade torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install --upgrade torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install --upgrade torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Finally, install this package itself using

```pip install -e .```

when you are in the main folder of this repository.
### Internal Repositories
The following repositories are required to run the code:
- [HMPN](https://github.com/ALRhub/hmpn.git)
- [hydra_list_sweeper](https://github.com/ALRhub/hydra_list_sweeper.git)

To install them, run
```
git clone https://www.github.com/ALRhub/HMPN.git ../hmpn

pip install -e ../hmpn
```


### Troubleshooting
- If the following error occurs:
`ModuleNotFoundError: No module named 'torch'`
Then install torch together with torchvision and torchaudio using
`pip3 install torch torchvision torchaudio`
- If you have the problem
` fatal error: Python.h: No such file or directory`
Then install the package `python3.10-dev` using `sudo apt-get install python3.10-dev`


## License
"Stamp Forming Simulation" is open-sourced under the [MIT license](LICENSE).
