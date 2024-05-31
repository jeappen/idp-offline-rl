[//]: # (<p align="center"><img align="center" width="300px" src="assets/logo.png"></p>)

# [ICML 2024] Kernelized Stein Discrepancy for Offline Reinforcement Learning

This code builds off d3rlpy (MIT License), a framework for Offline Reinforcement Learning.
We share the same dataset handling functionality.
Follow the instructions below to set up the environment:

- Create a new python 3.8 environment from Miniconda with the requirements

```
conda create -y -n offrl python=3.8 
conda activate offrl
conda install -y numpy pandas seaborn cython==0.29.21
pip install -r requirements.txt
pip install -e .
```

- Add this environment to your jupyter notebook

```
conda install -y -c anaconda ipykernel
python -m ipykernel install --user --name offrl --display-name "Python (offrl)"
```

- Now you can run the scripts and notebooks in this repo from the tutorials
- Run pytest from the root to check your installation (first disable tqdm to avoid conflicts)
  ```
  TQDM_DISABLE=1 && pytest tests/lcb
  ```
- All code for the main algorithms in the paper are in [d3rlpy/lcb](d3rlpy/lcb).

### File Structure
```
.
├── ...
├── tests                   # Test files
│   ├── lcb                 # Basic functionality tests for paper algos
│   .
├── d3rlpy                  # All source files for d3rlpy (with some minor edits)
│   ├── lcb                 # Our code for the paper algorithms
│   .
├── tutorials               # Scripts and notebooks (front-end)
│   ├── run_*.py            # Scripts with options to choose data type for each environment
│   ├── PlotPickle.ipynb    # Edit paths and use to plot graphs once results are generated
│   .
└── ...
```

## Scripts

- DeepSea, PriorMDP, Random MDP experiments : [run_mdp.py](tutorials/run_mdp.py)
- FrozenLake experiments : [run_frozen.py](tutorials/run_frozen.py)
- Portfolio experiments : [run_portfolio.py](tutorials/run_portfolio.py)

### Example Usage

```
conda activate offrl
# Run the random MDP experiments with the random policy dataset
python run_mdp.py --run-mode all --extra-exp-prefix run_random_test --dataset rand
# Run the random MDP Q-learning experiments with the easy dataset
python run_mdp.py --run-mode qling --extra-exp-prefix run_random_test --dataset easy

```

## Notebooks

- Offline RL tools : [tutorials/OfflineRL_random.ipynb](tutorials/OfflineRL_random.ipynb)
- Plotting : [tutorials/PlotPickle.ipynb](tutorials/PlotPickle.ipynb)

## Misc.

For more example usage refer the ipython notebook  [tutorials/OfflineRL_frozenlake.ipynb](tutorials/OfflineRL_frozenlake.ipynb).

## Bibtex

```bibtex
@inproceedings{koppel2024informationdirected,
title={Information-Directed Pessimism for Offline Reinforcement Learning},
author={Koppel, Alec and Bhatt, Sujay and Jiacheng Guo and Joe Eappen and Mengdi Wang and Ganesh, Sumitra},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
}
```


---
## D3RLPY Old README

check the [old README](https://github.com/jeappen/idp-offline-rl/blob/a11b15db1e60d47296fdc7e7880fe1165803138a/README.md) for the original README.md for d3rlpy.
